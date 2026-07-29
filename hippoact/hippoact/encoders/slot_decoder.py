"""DINOSAUR-style spatial-broadcast slot decoder over DINOv2 features."""
from __future__ import annotations

import torch
import torch.nn as nn


class SlotFeatureDecoder(nn.Module):
    """Reconstruct DINOv2 patch tokens from slots via spatial broadcast + MLP.

    Each slot proposes a full-image feature map and a mixture weight per patch;
    slots then compose via a softmax over slot index.
    """

    def __init__(
        self,
        slot_dim: int = 128,
        out_dim: int = 384,
        num_patches: int = 196,
        hidden: int = 256,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, slot_dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim + 1),  # +1 alpha channel
        )

    def forward(self, slots: torch.Tensor):
        """slots: (B, K, D). Return (recon, alpha) with alpha over slot dim.

        recon: (B, N, out_dim)
        alpha: (B, K, N)  softmax over slot dim
        """
        B, K, D = slots.shape
        N = self.num_patches
        # Broadcast slot to each patch position; add positional embedding.
        s = slots.unsqueeze(2).expand(-1, -1, N, -1) + self.pos_emb.unsqueeze(1)
        out = self.mlp(s)                                  # (B, K, N, out_dim+1)
        feats, mask_logits = out[..., :-1], out[..., -1]
        alpha = mask_logits.softmax(dim=1)                 # softmax over K
        recon = (feats * alpha.unsqueeze(-1)).sum(dim=1)   # (B, N, out_dim)
        return recon, alpha
