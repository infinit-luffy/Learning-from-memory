"""Slot Attention (Locatello et al., 2020) tailored for DINOv2 patch tokens."""
from __future__ import annotations

import torch
import torch.nn as nn


class SlotAttention(nn.Module):
    """Iterative slot attention with competitive softmax over slots."""

    def __init__(
        self,
        num_slots: int = 16,
        slot_dim: int = 128,
        input_dim: int = 384,
        iters: int = 3,
        hidden_mlp: int = 256,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_ff = nn.LayerNorm(slot_dim)

        # Learnable slot initialization
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.02)
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(input_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp, slot_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, input_dim). Return slots: (B, K, slot_dim)."""
        B = x.shape[0]
        x_n = self.norm_input(x)
        k, v = self.to_k(x_n), self.to_v(x_n)

        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)

        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))
            # Competitive softmax over slots (dim=1) — the object-centric bit.
            attn_logits = torch.einsum("bkd,bnd->bkn", q, k) * self.scale
            attn = attn_logits.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bkn,bnd->bkd", attn, v)

            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(B, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots
