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


class VisionBindingEncoder(nn.Module):
    """W2.1 E2E-1: trainable, permutation-equivariant readout over stacked slots.

    This is §III.E's binding transformer in its vision-only form (TODO §R4.10
    ruling).  Two deliberate differences from `BindingTransformer` above, both
    forced by measurement:

    * **No proprio token.**  On locomotion, proprio is the full state
      (`proprio_only` scores 974 on walker-walk and 0.999 zero-shot
      invariance), so any Q2 claim made with it is vacuous.  Q2 is vision-only.
    * **No slot positional embedding.**  `pos_slot` would make the readout
      permutation-*sensitive*, which is the property E2E-0's flatten had.  Only
      `pos_time` remains, so the encoder is equivariant across the slot axis
      and sensitive across the frame axis -- exactly the smoke test in the
      ruling (shuffle slots -> z unchanged; shuffle frames -> z changes).

    Input is the flat vector the environment emits, so TD-MPC2 keeps running
    with `obs=state` and its own pipeline is untouched:

        per frame t:  [ slots (K*Ds, row-major) | fast mask (K) ]
        obs = concat of those T blocks, oldest frame first

    matching how the environment's frame-stack deque concatenates.

    Slow slots are dropped via `src_key_padding_mask` rather than zeroed, which
    is what makes the exclusion real: a zeroed slot is still a token a
    transformer attends to, an masked one is not.
    """

    def __init__(self, slot_dim: int = 128, num_slots: int = 16,
                 t_window: int = 3, d_model: int = 128, n_layers: int = 4,
                 n_heads: int = 4, out_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.t_window = t_window
        self.n_tokens = t_window * num_slots

        self.slot_proj = nn.Linear(slot_dim, d_model)
        # Time only -- no slot-index embedding, by design (see class docstring).
        self.pos_time = nn.Parameter(torch.randn(1, t_window, 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, out_dim)

    @property
    def in_dim(self) -> int:
        return self.n_tokens * self.slot_dim + self.n_tokens

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """(..., in_dim) -> (..., out_dim).  Leading dims are free so the same
        module serves `act()` (B, D) and `update()` (T, B, D)."""
        lead = obs.shape[:-1]
        x = obs.reshape(-1, obs.shape[-1])
        B, T, K = x.shape[0], self.t_window, self.num_slots

        blk = x.view(B, T, K * self.slot_dim + K)
        slots = blk[..., :K * self.slot_dim].view(B, T, K, self.slot_dim)
        mask = blk[..., K * self.slot_dim:]                     # (B,T,K)

        tokens = self.slot_proj(slots) + self.pos_time          # (B,T,K,d)
        tokens = tokens.reshape(B, T * K, -1)
        key_padding = (mask.reshape(B, T * K) < 0.5)

        # A fully-masked row makes softmax over an empty set -> NaN, which
        # would propagate silently through the whole world model.  Measured
        # fast-slot count is 6-15 of 16 per frame so this should never fire,
        # but "should never" is not a guarantee worth a NaN.
        #
        # Written branch-free on purpose: a Python `if x.any():` plus an
        # in-place index assignment is data-dependent control flow, which
        # segfaulted inside inductor's cudagraph backward (torch 2.x,
        # `cudagraph_trees.py:_backward_impl`).  Falling back to
        # `compile=false` would have hidden the cause and cost throughput;
        # this keeps the guard and stays capturable.
        all_masked = key_padding.all(dim=1, keepdim=True)
        key_padding = key_padding & ~all_masked

        h = self.out_norm(self.encoder(tokens, src_key_padding_mask=key_padding))
        valid = (~key_padding).to(h.dtype).unsqueeze(-1)
        c = (h * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        return self.out_proj(c).reshape(*lead, -1)
