import os
import random
from collections import namedtuple
from typing import Optional, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# SumTree (for PER)
# -------------------------
class SumTree:
    """
    Binary SumTree for efficient prioritized sampling.
    Stores priorities in a binary tree and transitions in a data array.
    """
    def __init__(self, capacity: int):
        assert capacity > 0 and (capacity & (capacity - 1)) == 0, "capacity must be power of two"
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)  # binary tree
        self.data = [None] * capacity  # circular buffer
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, p: float, data):
        if data.state is not None:
            tree_idx = self.write + self.capacity - 1
            self.data[self.write] = data
            self.update(tree_idx, p)
            self.write = (self.write + 1) % self.capacity
            self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, tree_idx: int, p: float):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        self._propagate(tree_idx, change)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]

    def __len__(self):
        return self.n_entries

# -------------------------
# Prioritized Replay Buffer
# -------------------------
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class PrioritizedReplayBuffer:
    """
    Proportional Prioritized Experience Replay with SumTree.
    Exposes push(), sample(), update_priorities()
    """
    def __init__(self, capacity: int, device: torch.device, alpha: float = 0.6):
        # capacity must be power of two for this SumTree implementation convenience
        # If user passes non-power-of-two, we round up to next power of two.
        pow2 = 1
        while pow2 < capacity:
            pow2 *= 2
        self.capacity = pow2
        self.device = device
        self.alpha = alpha  # how much prioritization is used (0 = uniform, 1 = full)
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0  # new transitions get max priority so they are sampled at least once

    def push(self, state, action, reward, next_state, done):
        data = Transition(state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, data)

    def sample(self, batch_size: int, beta: float = 0.4):
        assert len(self.tree) > 0, "Cannot sample from empty buffer"
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            if data is not None: # TODO: find what is causing the None and correct
                idxs.append(idx)
                priorities.append(p)
                batch.append(data)

        # convert to tensors
        states = torch.stack([torch.as_tensor(b.state, dtype=torch.float32) for b in batch]).to(self.device)
        actions = torch.as_tensor([b.action for b in batch], dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.as_tensor([b.reward for b in batch], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.stack([torch.as_tensor(b.next_state, dtype=torch.float32) for b in batch]).to(self.device)
        dones = torch.as_tensor([float(b.done) for b in batch], dtype=torch.float32).unsqueeze(1).to(self.device)

        # compute importance-sampling weights
        total = self.tree.total()
        probs = np.array(priorities) / total
        # avoid zero
        probs = np.maximum(probs, 1e-8)
        N = len(self.tree)
        weights = (N * probs) ** (-beta)
        weights = weights / (weights.max() + 1e-8)  # normalize by max weight for stability
        weights = torch.as_tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones, idxs, weights

    def update_priorities(self, idxs, priorities):
        # priorities: list or array of new priority values (td error absolute + eps)
        for idx, p in zip(idxs, priorities):
            p_adj = (abs(p) + 1e-6) ** self.alpha
            if p_adj <= 0:
                p_adj = 1e-6
            self.tree.update(idx, float(p_adj))
            self.max_priority = max(self.max_priority, float(p_adj))

    def __len__(self):
        return len(self.tree)
    
    def clear(self):
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0