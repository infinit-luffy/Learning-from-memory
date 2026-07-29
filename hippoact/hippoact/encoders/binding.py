"""Cross-modal binding: Transformer encoder over (fast_slots × proprio) time window."""
from __future__ import annotations

import torch
import torch.nn as nn


class BindingTransformer(nn.Module):
    """Bind fast slots and proprioception into a single episodic code c_t.

    Slow slots are excluded via a boolean key-padding mask — this is what
    makes the slot-swap augmentation (P3) sound: policy-consuming pathways
    truly never see background slot content.
    """

    def __init__(
        self,
        slot_dim: int = 128,
        proprio_dim: int = 32,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        t_window: int = 4,
        num_slots: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.t_window = t_window
        self.num_slots = num_slots
        self.d_model = d_model

        self.slot_proj = nn.Linear(slot_dim, d_model)
        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        # (time × slot-index) positional embeddings; the proprio "extra" slot
        # gets the K-th positional slot index.
        self.pos_time = nn.Parameter(torch.randn(1, t_window, 1, d_model) * 0.02)
        self.pos_slot = nn.Parameter(torch.randn(1, 1, num_slots + 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        fast_slots_seq: torch.Tensor,     # (B, T, K, slot_dim)
        proprio_seq: torch.Tensor,        # (B, T, proprio_dim)
        fast_mask: torch.Tensor,          # (B, T, K)  1 = keep, 0 = ignore (slow)
    ) -> torch.Tensor:
        B, T, K, _ = fast_slots_seq.shape
        assert T == self.t_window and K == self.num_slots

        s = self.slot_proj(fast_slots_seq) + self.pos_time + self.pos_slot[:, :, :K]
        p = self.proprio_proj(proprio_seq).unsqueeze(2) \
            + self.pos_time + self.pos_slot[:, :, K:K + 1]
        # Concat along the "token per timestep" dim (proprio as extra slot).
        tokens = torch.cat([s, p], dim=2)                       # (B, T, K+1, d)
        tokens = tokens.reshape(B, T * (K + 1), -1)

        proprio_mask = torch.ones(B, T, 1, device=fast_mask.device, dtype=fast_mask.dtype)
        mask = torch.cat([fast_mask, proprio_mask], dim=2).reshape(B, T * (K + 1))
        # key_padding_mask: True = ignored.
        key_padding_mask = mask < 0.5

        h = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        h = self.out_norm(h)
        valid = (~key_padding_mask).float().unsqueeze(-1)
        c = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return c  # (B, d_model)
