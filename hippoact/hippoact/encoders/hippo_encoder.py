"""Top-level HippoAct encoder: raw obs sequence → structured latent state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from hippoact.encoders.binding import BindingTransformer
from hippoact.encoders.dinov2 import DinoV2Encoder
from hippoact.encoders.slot_attention import SlotAttention
from hippoact.encoders.slot_decoder import SlotFeatureDecoder
from hippoact.encoders.slot_router import SlotRouter


@dataclass
class EncoderOutput:
    slots: torch.Tensor            # (B, K, D_s) — last frame slots
    fast_slots: torch.Tensor       # (B, K, D_s) — slow positions zeroed
    fast_mask: torch.Tensor        # (B, K)      — 1 if fast
    c: Optional[torch.Tensor] = None            # (B, D_c)  — episodic code
    slot_recon: Optional[torch.Tensor] = None   # (B, N, D_v)
    slot_alpha: Optional[torch.Tensor] = None   # (B, K, N)
    router_logits: Optional[torch.Tensor] = None
    slots_seq: Optional[torch.Tensor] = None    # (B, T, K, D_s) — all frames
    fast_mask_seq: Optional[torch.Tensor] = None  # (B, T, K)


class HippoActEncoder(nn.Module):
    """The full HippoAct visual encoder.

    Pipeline: DINOv2 (frozen) → SlotAttention → SlotRouter → BindingTransformer.
    """

    def __init__(
        self,
        num_slots: int = 16,
        slot_dim: int = 128,
        proprio_dim: int = 32,
        c_dim: int = 128,
        t_window: int = 4,
        image_size: int = 224,
        patch_size: int = 14,
        dino_model: str = "dinov2_vits14",
        binding_layers: int = 4,
        binding_heads: int = 4,
        binding_dropout: float = 0.1,
        slot_iters: int = 3,
        slot_hidden: int = 256,
        router_hidden: int = 64,
        gumbel_tau_init: float = 1.0,
        slot_query_mode: str = "sampled",   # {"sampled", "learned"}
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.t_window = t_window
        self.c_dim = c_dim

        self.dino = DinoV2Encoder(
            model_name=dino_model, image_size=image_size, patch_size=patch_size
        )
        self.slot_attn = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=self.dino.feat_dim,
            iters=slot_iters,
            hidden_mlp=slot_hidden,
            learned_queries=(slot_query_mode == "learned"),
        )
        self.slot_decoder = SlotFeatureDecoder(
            slot_dim=slot_dim,
            out_dim=self.dino.feat_dim,
            num_patches=self.dino.n_patches,
        )
        self.router = SlotRouter(
            slot_dim=slot_dim, hidden=router_hidden, tau_init=gumbel_tau_init
        )
        self.binding = BindingTransformer(
            slot_dim=slot_dim,
            proprio_dim=proprio_dim,
            d_model=c_dim,
            n_layers=binding_layers,
            n_heads=binding_heads,
            t_window=t_window,
            num_slots=num_slots,
            dropout=binding_dropout,
        )

    # ---- Per-frame API -------------------------------------------------

    def encode_frame(self, img: torch.Tensor, decode: bool = True,
                     slots_init: torch.Tensor | None = None) -> EncoderOutput:
        """Encode a single frame. img: (B, 3, H, W).

        `decode=False` skips the slot decoder. Stage-1 needs `recon`/`alpha`
        for its reconstruction and connectivity losses; downstream RL does not
        — the router reads slot vectors only. The decoder is not a minor
        addition: it materialises (B, K, N, D_v), i.e. 6 GB at B=1024, and
        dominates the per-batch cost. Skipping it is exact, not an
        approximation, whenever the caller ignores recon/alpha.
        """
        feats = self.dino(img)                              # (B, N, D_v)
        # `slots_init` makes the encoder a deterministic function of the frame.
        # In `sampled` query mode the slots are separated only by the noise in
        # `sample_init`, so without a fixed init the same frame encodes
        # differently every call — see adapters/slot_features.py.
        slots = self.slot_attn(feats, slots_init=slots_init)   # (B, K, D_s)
        if decode:
            recon, alpha = self.slot_decoder(slots)         # (B, N, D_v), (B, K, N)
        else:
            recon, alpha = None, None
        g, logits = self.router(slots)                      # g: (B, K, 2)
        fast_mask = g[..., 1]                               # (B, K)  0/1
        fast_slots = slots * fast_mask.unsqueeze(-1)
        return EncoderOutput(
            slots=slots,
            fast_slots=fast_slots,
            fast_mask=fast_mask,
            slot_recon=recon,
            slot_alpha=alpha,
            router_logits=logits,
        )

    # ---- Sequence API --------------------------------------------------

    def forward(
        self, imgs_seq: torch.Tensor, proprio_seq: torch.Tensor
    ) -> EncoderOutput:
        """Encode a sequence and produce the episodic code c_t.

        imgs_seq:    (B, T, 3, H, W)
        proprio_seq: (B, T, proprio_dim)
        """
        B, T = imgs_seq.shape[:2]
        assert T == self.t_window, f"expected T={self.t_window}, got {T}"

        outs = [self.encode_frame(imgs_seq[:, t]) for t in range(T)]
        slots_seq = torch.stack([o.slots for o in outs], dim=1)         # (B,T,K,Ds)
        fast_slots_seq = torch.stack([o.fast_slots for o in outs], dim=1)
        fast_mask_seq = torch.stack([o.fast_mask for o in outs], dim=1)  # (B,T,K)

        c = self.binding(fast_slots_seq, proprio_seq, fast_mask_seq)     # (B, c_dim)

        last = outs[-1]
        return EncoderOutput(
            slots=last.slots,
            fast_slots=last.fast_slots,
            fast_mask=last.fast_mask,
            c=c,
            slot_recon=last.slot_recon,
            slot_alpha=last.slot_alpha,
            router_logits=last.router_logits,
            slots_seq=slots_seq,
            fast_mask_seq=fast_mask_seq,
        )

    # ---- Convenience ---------------------------------------------------

    def trainable_parameters(self):
        """Iterator over parameters that should receive gradients (excludes DINOv2)."""
        for name, p in self.named_parameters():
            if p.requires_grad:
                yield name, p

    def anneal_router(self, factor: float = 0.9995, tau_min: float = 0.3) -> float:
        return self.router.anneal(factor=factor, tau_min=tau_min)
