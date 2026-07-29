"""Gumbel-Softmax slot router: slow (background) vs. fast (foreground)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotRouter(nn.Module):
    """Small MLP + Gumbel-Softmax that assigns each slot to {slow, fast}.

    Output g is a one-hot (hard=True forward) with straight-through gradients.
    Convention: index 0 = slow, index 1 = fast.
    """

    def __init__(self, slot_dim: int = 128, hidden: int = 64, tau_init: float = 1.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )
        # Register tau as buffer so it moves with .to(device) but is not a parameter.
        self.register_buffer("tau", torch.tensor(float(tau_init)))

    def forward(self, slots: torch.Tensor, hard: bool = True):
        """slots: (B, K, D). Return g: (B, K, 2), logits: (B, K, 2)."""
        logits = self.mlp(slots)
        g = F.gumbel_softmax(logits, tau=float(self.tau.item()), hard=hard, dim=-1)
        return g, logits

    @torch.no_grad()
    def anneal(self, factor: float = 0.9995, tau_min: float = 0.3) -> float:
        self.tau.mul_(factor).clamp_(min=tau_min)
        return float(self.tau.item())
