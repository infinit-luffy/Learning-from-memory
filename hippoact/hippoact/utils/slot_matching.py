"""Slot identity matching across independent Slot Attention forward passes.

Slot Attention initializes slot queries stochastically each forward pass, so
slot index k has no correspondence between two encodings of the same scene.
When we want to compute per-slot temporal variance across two frames, we need
to first solve the assignment: for each slot in slots_t, find its analogue in
slots_prev.

The greedy nearest-neighbor solution below is `O(B·K²·D)` and is adequate for
the typical `K = 8..24` regime we use. Hungarian assignment would be more
principled; benchmarks did not show a meaningful downstream difference.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def match_slots_nn(
    slots_t: torch.Tensor,        # (B, K, D)  reference (target ordering)
    slots_ref: torch.Tensor,      # (B, K, D)  to be permuted
) -> torch.Tensor:
    """Return ``slots_ref`` permuted per batch so that position ``k`` matches
    the content of ``slots_t[b, k]`` under cosine similarity.

    Greedy per-row NN — multiple slots in ``slots_t`` can map to the same
    slot in ``slots_ref``, but with typical L_div regularization slots are
    diverse enough that collisions are rare.
    """
    a = F.normalize(slots_t, dim=-1)
    b = F.normalize(slots_ref, dim=-1)
    sim = torch.matmul(a, b.transpose(-1, -2))            # (B, K, K)
    idx = sim.argmax(dim=-1)                              # (B, K)
    perm = idx.unsqueeze(-1).expand(-1, -1, slots_ref.size(-1))
    return slots_ref.gather(dim=1, index=perm)
