import os
import random
from collections import deque, namedtuple
from typing import Tuple, Optional

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.model import BaseQNet
from src.game.buffer import PrioritizedReplayBuffer

# -------------------------
# DQN Agent w/ Double DQN + PER
# -------------------------
class DQNAgentPER:
    def __init__(
        self,
        env: gym.Env,
        q_net: BaseQNet,
        target_net: BaseQNet,
        device: Optional[torch.device] = None,
        buffer_size: int = 16384,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        sync_every: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.05,
        epsilon_decay: int = 10000,
        update_every: int = 1,
        tau: float = 1.0,
        grad_clip: Optional[float] = None,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100000,
        seed: Optional[int] = None,
    ):
        self.env = env
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.q_net = q_net.to(self.device)
        self.target_net = target_net.to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # PER buffer (capacity rounded to power of two)
        self.replay = PrioritizedReplayBuffer(buffer_size, device=self.device, alpha=per_alpha)
        self.batch_size = batch_size
        self.gamma = gamma
        self.sync_every = sync_every
        self.update_every = update_every
        self.tau = tau
        self.grad_clip = grad_clip

        # epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0

        # PER beta schedule
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames

        # bookkeeping
        self.loss_fn = nn.MSELoss(reduction="none")  # we'll handle reduction with IS weights
        self.train_steps = 0

        if seed is not None:
            self.seed(seed)

    def seed(self, s: int):
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        try:
            self.env.seed(s)
        except Exception:
            pass

    def epsilon(self):
        """Linear epsilon decay"""
        return max(self.epsilon_final, self.epsilon_start - (self.epsilon_start - self.epsilon_final) * (self.total_steps / self.epsilon_decay))

    def beta_by_frame(self, frame_idx: int):
        """Anneal beta from per_beta_start to 1.0 over per_beta_frames"""
        return min(1.0, self.per_beta_start + (1.0 - self.per_beta_start) * (frame_idx / max(1, self.per_beta_frames)))

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if (not eval_mode) and random.random() < self.epsilon():
            return self.env.action_space.sample()
        self.q_net.eval()
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_net(s)
            action = int(torch.argmax(q_values, dim=1).item())
        self.q_net.train()
        return action

    def push_transition(self, s, a, r, s_next, done):
        self.replay.push(s, a, r, s_next, done)

    def compute_td_loss_per(self):
        if len(self.replay) < self.batch_size:
            return None, None, None  # no loss yet

        beta = self.beta_by_frame(self.total_steps)
        states, actions, rewards, next_states, dones, idxs, weights = self.replay.sample(self.batch_size, beta=beta)

        # current Q(s,a)
        q_values = self.q_net(states)
        q_val = q_values.gather(1, actions)  # (batch,1)

        # Double DQN target:
        # - use q_net (online) to select argmax actions for next_states
        # - use target_net to evaluate those actions
        with torch.no_grad():
            # online network chooses best actions for next_states
            next_q_online = self.q_net(next_states)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)  # (batch,1)

            # target network evaluates those actions
            next_q_target = self.target_net(next_states)
            next_q_val = next_q_target.gather(1, next_actions)  # (batch,1)

            # compute target q
            target_q = rewards + (1.0 - dones) * (self.gamma * next_q_val)

        td_errors = (q_val - target_q).squeeze(1).detach().cpu().numpy()  # for priorities update

        # element-wise loss weighted by importance-sampling weights
        loss_element = self.loss_fn(q_val, target_q)  # (batch,1)
        loss_weighted = (weights * loss_element).mean()

        # optimize
        self.optimizer.zero_grad()
        loss_weighted.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # update priorities in buffer (absolute TD error)
        new_priorities = np.abs(td_errors) + 1e-6
        self.replay.update_priorities(idxs, new_priorities)

        return loss_weighted.item(), td_errors, idxs

    def soft_update_target(self):
        if self.tau >= 1.0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q_state": self.q_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(data["q_state"])
        self.target_net.load_state_dict(data["target_state"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.total_steps = data.get("total_steps", 0)

    def train(
        self,
        num_episodes: int = 200,
        max_steps_per_episode: Optional[int] = None,
        log_every: int = 10,
        render: bool = False,
    ):
        episode_returns = []
        for ep in range(1, num_episodes + 1):
            state, _info = self.env.reset()
            ep_return = 0.0
            done = False
            steps = 0
            while not done:
                action = self.select_action(state, eval_mode=False)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done_flag = bool(terminated or truncated)

                # store
                self.push_transition(state, action, reward, next_state, done_flag)

                # bookkeeping
                state = next_state
                ep_return += reward
                steps += 1
                self.total_steps += 1

                # training step
                if self.total_steps % self.update_every == 0:
                    loss, td_errors, idxs = self.compute_td_loss_per()
                    if loss is not None:
                        self.train_steps += 1

                # sync target periodically
                if self.total_steps % self.sync_every == 0:
                    self.soft_update_target()

                if render:
                    self.env.render()

                if done_flag or (max_steps_per_episode is not None and steps >= max_steps_per_episode):
                    done = True

            episode_returns.append(ep_return)
            if ep % log_every == 0 or ep == 1:
                avg = sum(episode_returns[-log_every:]) / len(episode_returns[-log_every:])
                print(f"[EP {ep}/{num_episodes}] steps {self.total_steps} | episode_return {ep_return:.2f} | avg_last{log_every} {avg:.2f} | eps {self.epsilon():.3f} | replay_len {len(self.replay)}")

        return episode_returns