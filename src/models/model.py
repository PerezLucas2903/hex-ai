import os
import random
from collections import deque, namedtuple
from typing import Tuple, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ------------------------------
# Base Q-network Interface
# ------------------------------
class BaseQNet(nn.Module):
    """
    Base class / interface for Q-networks.
    Your custom architectures should subclass this and implement forward(self, x)
    which returns Q-values of shape (batch, n_actions) when x is (batch, ...).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward() returning Q-values.")

# ------------------------------
# Example architectures
# ------------------------------
class MLP_QNet(BaseQNet):
    """Simple MLP for vector observations (flattened)."""
    def __init__(self, input_dim: int, n_actions: int, hidden_dims=(128, 128)):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        size = x.size()
        x = x.reshape(size[0],-1)
        return self.net(x)


class Conv_QNet(BaseQNet):
    """
    Small conv net for grid observations (C, H, W) or (H, W) single-channel.
    Output: Q-values (batch, n_actions).
    Assumes later you flatten conv features and use a final linear to n_actions.
    """
    def __init__(self, input_shape: Tuple[int, int], n_actions: int, channels=(32, 64)):
        # input_shape: (H, W) or (C, H, W) -> we'll accept (H,W) and add channel=1
        super().__init__()
        if len(input_shape) == 3:
            self.in_channels = input_shape[0]
            h, w = input_shape[1], input_shape[2]
        else:
            self.in_channels = 1
            h, w = input_shape

        conv_layers = []
        in_ch = self.in_channels
        for ch in channels:
            conv_layers.append(nn.Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1))
            conv_layers.append(nn.ReLU())
            in_ch = ch
        self.conv = nn.Sequential(*conv_layers)

        # compute conv output size with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, h, w)
            conv_out = self.conv(dummy)
            conv_flat = int(np.prod(conv_out.shape[1:]))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_flat, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        # Accept either (batch, H, W) or (batch, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # add channel
        convf = self.conv(x)
        q = self.head(convf)
        return q
