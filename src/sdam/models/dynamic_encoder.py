from __future__ import annotations

import torch
from torch import nn


class DynamicEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dynamic_dim: int) -> None:
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels, dynamic_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 5:
            raise ValueError("obs must have shape [B, T, C, H, W]")
        if obs.shape[1] < 2:
            raise ValueError("obs time dimension must be at least 2")
        diffs = (obs[:, 1:] - obs[:, :-1]).abs()
        batch, steps, channels, height, width = diffs.shape
        encoded = self.net(diffs.reshape(batch * steps, channels, height, width))
        return encoded.reshape(batch, steps, self.dynamic_dim)
