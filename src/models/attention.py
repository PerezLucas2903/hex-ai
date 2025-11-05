import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.model import BaseQNet
from typing import Tuple, Optional
import numpy as np

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.softmax = nn.Softmax(dim=-1)

        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.positional_enconding = nn.Linear(2, dim)

    def forward(self, x):
        batch_size, height, width,n_dim = x.size()
        if n_dim != self.dim:
            raise ValueError(f"Input channels {n_dim} do not match attention dimension {self.dim}")

        # Positional Encoding for height and width
        pos_enc_h = torch.arange(height, device=x.device).unsqueeze(1).repeat(1, width)
        pos_enc_w = torch.arange(width, device=x.device).unsqueeze(0).repeat(height, 1)
        pos_enc = torch.stack((pos_enc_h, pos_enc_w), dim=-1).float()  # (H, W, 2)
        pos_enc = self.positional_enconding(pos_enc)  # (H, W, C)
        x = x + pos_enc.unsqueeze(0)  # (B, H, W, C)

        # Compute attention
        x_flat = x.view(batch_size, height * width, n_dim)  # (B, H*W, C)
        attn_weights = self.softmax(torch.bmm(x_flat, x_flat.transpose(1, 2)))  # (B, H*W, H*W)
        out = torch.bmm(attn_weights, x_flat)  # (B, H*W, C)

        # Reshape back to (B, C, H, W)
        out = self.linear(out)

        # Residual connection
        out = out.view(batch_size, height, width, n_dim) + x

        # Normalization
        out = F.layer_norm(out, normalized_shape=[height, width, n_dim])

        return out
    

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.softmax = nn.Softmax(dim=-1)

        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, x, context):
        batch_size, height, width,n_dim = x.size()
        if n_dim != self.dim:
            raise ValueError(f"Input channels {n_dim} do not match attention dimension {self.dim}")

        # Compute attention
        x_flat = x.view(batch_size, height * width, n_dim)  # (B, H*W, C)
        context_flat = context.view(batch_size, -1, n_dim)  # (B, N, C)
        attn_weights = self.softmax(torch.bmm(x_flat, context_flat.transpose(1, 2)))  # (B, H*W, N)
        out = torch.bmm(attn_weights, context_flat)  # (B, H*W, C)

        # Reshape back to (B, C, H, W)
        out = self.linear(out)

        # Residual connection
        out = out.view(batch_size, height, width, n_dim) + x

        # Normalization
        out = F.layer_norm(out, normalized_shape=[height, width, n_dim])

        return out

class Attention_QNet(BaseQNet):
    def __init__(self, n_attention_layers: int, n_dim: int):
        super().__init__()
        self.embed = nn.Embedding(3, n_dim)
        self.attention_layers = nn.ModuleList([Attention(n_dim) for _ in range(n_attention_layers)])

        self.advantage_layer = nn.Sequential(
            nn.Linear(n_dim, n_dim),
            nn.ReLU(),
            nn.Linear(n_dim, 1)
        )

        self.value_x = nn.Parameter(torch.randn(n_dim))
        self.value_layer = nn.Sequential(
            nn.Linear(n_dim, n_dim),
            nn.ReLU(),
            nn.Linear(n_dim, 1)
        )
        self.cross_attention = CrossAttention(n_dim)

    def forward(self, x):
        x = x.long()
        x = self.embed(x)  # (B, H, W, C)

        for attn_layer in self.attention_layers:
            x = attn_layer(x)

        # Advantage calculation
        advantage = self.advantage_layer(x)  # (B, H, W, 1)
        advantage = advantage.reshape(x.size(0), -1)  # (B, H*W)

        # Value calculation
        value_x = self.cross_attention(self.value_x.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(x.size(0),1,1,1), x)  # (B, 1, 1, C)
        value = self.value_layer(value_x)  # (B, 1)
        value = value.reshape(x.size(0), 1)  # (B, 1)

        # q-value calculation
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)  # (B, H*W)

        # Tanh activation to keep q-values bounded
        q = nn.Tanh()(q)

        return q