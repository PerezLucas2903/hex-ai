import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.model import BaseQNet
from typing import Tuple, Optional
import numpy as np

class ResNet(nn.Module):
    def __init__(self, in_channels, out_in_channels, kernel_size,stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.out = nn.Sequential(nn.Conv2d(in_channels, out_in_channels, kernel_size, padding=padding, stride=stride),
                                 nn.BatchNorm2d(out_in_channels),
                                 nn.ReLU())

    def forward(self, x):
        return self.out(x + self.cnn(x))
    
class ResNet_QNet(BaseQNet):
    def __init__(self, input_shape: Tuple[int, int], n_actions: int, channels=(4,8,16,32), kernel_size=3):
        super().__init__()
        if len(input_shape) == 3:
            self.in_channels = input_shape[0]
            h, w = input_shape[1], input_shape[2]
        else:
            self.in_channels = 1
            h, w = input_shape

        resnet_layers = []
        in_ch = self.in_channels
        for ch in channels:
            resnet_layers.append(ResNet(in_ch,ch, kernel_size, stride=1))
            in_ch = ch
        self.resnet = nn.Sequential(*resnet_layers)

        # compute resnet output size with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, h, w)
            resnet_out = self.resnet(dummy)
            resnet_flat = int(np.prod(resnet_out.shape[1:]))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(resnet_flat, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.nn.functional.one_hot(x.long(),num_classes=3).float()
        x = x.permute(0,3,1,2)[:,1:]

        resnetf = self.resnet(x)
        q = self.head(resnetf)
        return q