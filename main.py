from src.game.env import HEX
from src.models.model import MLP_QNet
from src.game.agent import DQNAgentPER
import numpy as np

if __name__ == "__main__":
    env = HEX(grid_size=11, render_mode="matrix")

    obs_space = env.observation_space
    n_actions = env.action_space.n

    if len(obs_space.shape) == 1:
        input_dim = obs_space.shape[0]
        q_net = MLP_QNet(input_dim=input_dim, n_actions=n_actions)
        target_net = MLP_QNet(input_dim=input_dim, n_actions=n_actions)
    else:
        flat = int(np.prod(obs_space.shape))
        q_net = MLP_QNet(input_dim=flat, n_actions=n_actions)
        target_net = MLP_QNet(input_dim=flat, n_actions=n_actions)

    agent = DQNAgentPER(
        env=env,
        q_net=q_net,
        target_net=target_net,
        buffer_size=16384,
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        sync_every=1000,
        epsilon_start=1.0,
        epsilon_final=0.02,
        epsilon_decay=15000,
        update_every=1,
        tau=1.0,
        grad_clip=10.0,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_frames=50000,
        seed=42,
    )

    # quick smoke train (short)
    returns = agent.train(num_episodes=50, max_steps_per_episode=500, log_every=10, render=False)
    print("Done. Last returns:", returns[-5:])