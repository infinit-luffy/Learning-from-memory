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

    def sample_init(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """Sample slot queries from ``mu, sigma`` params. Exposed so paired
        forward passes can share the same random draw, eliminating the
        stochastic-init noise that would otherwise dominate temporal-variance
        signals (see docs/paper_section_III_method.md §III.D.3, discussion of
        the shared-init motivation).
        """
        device = device or self.slots_mu.device
        dtype = dtype or self.slots_mu.dtype
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(batch_size, self.num_slots, -1)
        eps = torch.randn(mu.shape, device=device, dtype=dtype)
        return mu + sigma * eps

    def forward(
        self, x: torch.Tensor, slots_init: torch.Tensor | None = None
    ) -> torch.Tensor:
        """x: (B, N, input_dim). Return slots: (B, K, slot_dim).

        If ``slots_init`` is given, use it verbatim as the starting slot
        queries (skip stochastic sampling). Otherwise, sample a fresh init
        via ``sample_init``. Passing a shared init across two forward passes
        makes the two outputs directly comparable per slot index.
        """
        B = x.shape[0]
        x_n = self.norm_input(x)
        k, v = self.to_k(x_n), self.to_v(x_n)

        if slots_init is None:
            slots = self.sample_init(B, device=x.device, dtype=x.dtype)
        else:
            assert slots_init.shape == (B, self.num_slots, self.slot_dim), (
                f"slots_init shape {tuple(slots_init.shape)} "
                f"!= expected ({B}, {self.num_slots}, {self.slot_dim})"
            )
            slots = slots_init

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
