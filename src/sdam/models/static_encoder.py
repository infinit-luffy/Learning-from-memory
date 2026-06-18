from __future__ import annotations

import torch
from torch import nn


class StaticEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, static_dim: int) -> None:
        super().__init__()
        self.static_dim = static_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels, static_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 5:
            raise ValueError("obs must have shape [B, T, C, H, W]")
        if obs.shape[1] < 2:
            raise ValueError("obs time dimension must be at least 2")
        background_estimate = obs.mean(dim=1)
        return self.net(background_estimate)
