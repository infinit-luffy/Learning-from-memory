"""HippoAct → TD-MPC2 encoder adapter (E2E-0 level).

E2E-0 contract (TODO.md W2.1): MINIMAL viable representation —
    z = MLP( flatten(S_fg) ⊕ proprio )
No binding transformer, no auxiliary losses, no temporal window. Those are
E2E-1/E2E-2 increments, added only after E2E-0 clears its criterion.

Wiring into the TD-MPC2 fork (single touch point, see PHASE2_PLAN §3.1):

    # third_party/tdmpc2/tdmpc2/common/world_model.py
    if cfg.encoder_type == "hippoact":
        from hippoact.adapters import HippoActAdapter, AdapterConfig
        self._encoder = HippoActAdapter(AdapterConfig(
            stage1_ckpt=cfg.hippoact_ckpt,
            proprio_dim=cfg.proprio_dim,     # walker: 24 (qpos 9 + qvel 9 + ...)
            latent_dim=cfg.latent_dim,       # keep TD-MPC2's own latent_dim
        ))
    else:
        self._encoder = layers.enc(cfg)      # pixel baseline untouched

Observation contract: the DCS wrapper must emit a dict
    {"rgb": (B, 3, 224, 224) float in [0,1] ImageNet-normalized,
     "state": (B, d_q) float}
TD-MPC2's replay/rollout code treats the encoder as opaque; nothing else
changes. MPPI must call the encoder ONCE per environment step (it plans in
latent space) — verify with the call-count assertion in tests below.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from hippoact.encoders.hippo_encoder import HippoActEncoder


@dataclass
class AdapterConfig:
    stage1_ckpt: str = ""            # path to Stage-1 checkpoint (required)
    proprio_dim: int = 24
    latent_dim: int = 256            # TD-MPC2 latent size — do not change
    num_slots: int = 16
    slot_dim: int = 128
    image_size: int = 224
    finetune_encoder: bool = False   # E2E-0 default: frozen Stage-1 modules
    finetune_lr_scale: float = 0.1   # if finetuning later (E2E-2+)


class HippoActAdapter(nn.Module):
    """obs dict → z. E2E-0: fast slots + proprio, nothing else."""

    def __init__(self, cfg: AdapterConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = HippoActEncoder(
            num_slots=cfg.num_slots,
            slot_dim=cfg.slot_dim,
            proprio_dim=cfg.proprio_dim,
            image_size=cfg.image_size,
        )
        if cfg.stage1_ckpt:
            payload = torch.load(Path(cfg.stage1_ckpt), map_location="cpu")
            state = payload.get("encoder", payload)
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            # binding.* keys will be missing at E2E-0 — that's expected.
            unexpected_real = [k for k in unexpected if not k.startswith("binding")]
            assert not unexpected_real, f"ckpt mismatch: {unexpected_real[:5]}"

        if not cfg.finetune_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        in_dim = cfg.num_slots * cfg.slot_dim + cfg.proprio_dim
        self.z_mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.Mish(inplace=True),          # match TD-MPC2's activation
            nn.Linear(512, cfg.latent_dim),
        )
        # Diagnostics: verify encoder isn't re-invoked inside planning.
        self._call_count = 0

    def forward(self, obs: dict) -> torch.Tensor:
        """obs: {"rgb": (B,3,H,W), "state": (B,d_q)} → z: (B, latent_dim)."""
        self._call_count += 1
        rgb, state = obs["rgb"], obs["state"]

        enc_ctx = torch.no_grad() if not self.cfg.finetune_encoder else _nullcontext()
        with enc_ctx:
            out = self.encoder.encode_frame(rgb)
            # Deployment-consistent deterministic routing (argmax, not Gumbel).
            fast_mask = (out.router_logits.argmax(dim=-1) == 1).float()  # (B, K)
            fast_slots = out.slots * fast_mask.unsqueeze(-1)             # (B, K, D)

        z = self.z_mlp(torch.cat([fast_slots.flatten(1), state], dim=-1))
        return z

    # -- health metrics for wandb ---------------------------------------

    @torch.no_grad()
    def routing_stats(self, obs: dict) -> dict:
        out = self.encoder.encode_frame(obs["rgb"])
        fast = (out.router_logits.argmax(dim=-1) == 1).float()
        return {
            "adapter/fast_ratio": fast.mean().item(),
            "adapter/slot_norm": out.slots.norm(dim=-1).mean().item(),
        }


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
