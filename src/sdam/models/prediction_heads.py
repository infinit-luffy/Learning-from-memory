from __future__ import annotations

import torch
from torch import nn


class PositionVelocityHead(nn.Module):
    def __init__(self, memory_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.net = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, memory: torch.Tensor) -> dict[str, torch.Tensor]:
        if memory.ndim != 2 or memory.shape[-1] != self.memory_dim:
            raise ValueError("memory must have shape [B, memory_dim]")
        prediction = self.net(memory)
        return {"position": prediction[:, :2], "velocity": prediction[:, 2:]}
