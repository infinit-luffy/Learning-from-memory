"""Slot-level background swap (P3).

Efficient in-batch approximation: for each sample, replace slots marked as
'slow' with the slots from a permuted batch-mate whose slots are also 'slow'
at the same slot index. We keep the fast slots of the source sample intact.
"""
from __future__ import annotations

import torch


def slot_swap_in_batch(
    slots: torch.Tensor,        # (B, K, D)
    fast_mask: torch.Tensor,    # (B, K)  1 = fast, 0 = slow
    perm: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (slots_swapped, swap_indicator) where indicator says whether the
    swap actually replaced anything for each (b, k) (both A and B slow).
    """
    B, K, D = slots.shape
    if perm is None:
        perm = torch.randperm(B, device=slots.device)
    slow_mask = 1.0 - fast_mask
    # A slot is genuinely swapped iff both source and target treat it as slow.
    swap_indicator = slow_mask * slow_mask[perm]                       # (B, K)
    # Replace: keep original where swap_indicator == 0, else take permuted.
    take_perm = swap_indicator.unsqueeze(-1)                           # (B, K, 1)
    slots_swapped = slots * (1.0 - take_perm) + slots[perm] * take_perm
    return slots_swapped, swap_indicator
