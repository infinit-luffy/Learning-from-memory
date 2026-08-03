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

        arch = {}
        payload = None
        if cfg.stage1_ckpt:
            payload = torch.load(Path(cfg.stage1_ckpt), map_location="cpu",
                                 weights_only=False)
            # Stage-1 checkpoints embed the architecture they were trained with
            # (read back from the live modules, so it cannot drift from the
            # config file). Build from that rather than from our own defaults —
            # a silent num_slots/slot_dim mismatch would otherwise surface only
            # as bad returns, days later.
            arch = payload.get("encoder_arch", {}) or {}
            if arch.get("dino_is_mock", False):
                raise ValueError(
                    f"{cfg.stage1_ckpt} was trained with MockDinoV2Encoder "
                    "(random frozen CNN) — refusing to use it for E2E."
                )

        self.num_slots = int(arch.get("num_slots", cfg.num_slots))
        self.slot_dim = int(arch.get("slot_dim", cfg.slot_dim))
        self.image_size = int(arch.get("image_size", cfg.image_size))

        # The encoder's own proprio width belongs to the binding transformer,
        # which E2E-0 does not use — but it must match the checkpoint or the
        # load fails. Stage-1 trains with the config default (32) while the
        # DMC env supplies 24, so read the trained width off the tensor rather
        # than assuming either. `cfg.proprio_dim` (the env's) is what z_mlp uses.
        enc_proprio_dim = cfg.proprio_dim
        if payload is not None:
            w = payload.get("encoder", payload).get("binding.proprio_proj.weight")
            if w is not None:
                enc_proprio_dim = int(w.shape[1])

        self.encoder = HippoActEncoder(
            num_slots=self.num_slots,
            slot_dim=self.slot_dim,
            proprio_dim=enc_proprio_dim,
            image_size=self.image_size,
            slot_iters=int(arch.get("slot_iters", 3)),
            dino_model=str(arch.get("dino_model", "dinov2_vits14")),
        )
        # DinoV2Encoder falls back to a random frozen CNN when torch.hub fails
        # (a transient network error is enough). Anything built on that mock is
        # meaningless, and the failure is otherwise invisible — refuse here
        # rather than let a two-day RL run produce numbers nobody can trust.
        # An *explicit* HIPPOACT_FORCE_MOCK=1 is a deliberate offline/test
        # choice and is allowed; what must never pass silently is the automatic
        # fallback that a transient torch.hub error triggers.
        import os as _os
        dino = getattr(self.encoder, "dino", None)
        if (dino is not None and getattr(dino, "is_mock", False)
                and _os.environ.get("HIPPOACT_FORCE_MOCK", "0") != "1"):
            raise RuntimeError(
                "DINOv2 backbone is MockDinoV2Encoder (torch.hub load failed). "
                "Set HIPPOACT_STRICT_DINO=1 to see the underlying error; "
                "refusing to run E2E on a random backbone."
            )

        if payload is not None:
            state = payload.get("encoder", payload)
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            # binding.* keys will be missing at E2E-0 — that's expected.
            unexpected_real = [k for k in unexpected if not k.startswith("binding")]
            assert not unexpected_real, f"ckpt mismatch: {unexpected_real[:5]}"
            missing_real = [k for k in missing
                            if not k.startswith(("binding", "dino", "_backbone"))]
            assert not missing_real, f"ckpt missing weights: {missing_real[:5]}"

        if not cfg.finetune_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        # ImageNet statistics, applied on GPU. The replay buffer stores rgb as
        # uint8 (147 KB/frame vs 588 KB as float32 — 75 GB vs 301 GB at 500K
        # capacity), so normalisation cannot happen in the env wrapper.
        self.register_buffer("_img_mean",
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_img_std",
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        in_dim = self.num_slots * self.slot_dim + cfg.proprio_dim
        self.z_mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.Mish(inplace=True),          # match TD-MPC2's activation
            nn.Linear(512, cfg.latent_dim),
        )
        # Diagnostics: verify encoder isn't re-invoked inside planning.
        self._call_count = 0

    def _preprocess(self, rgb: torch.Tensor) -> torch.Tensor:
        """uint8 (B,3,H,W) in [0,255] -> ImageNet-normalised float."""
        if rgb.dtype == torch.uint8:
            rgb = rgb.float().div_(255.0)
        elif rgb.max() > 1.5:            # float but still in [0,255]
            rgb = rgb / 255.0
        return (rgb - self._img_mean) / self._img_std

    def forward(self, obs: dict) -> torch.Tensor:
        """obs: {"rgb": (B,3,H,W) uint8, "state": (B,d_q)} → z: (B, latent_dim)."""
        self._call_count += 1
        rgb, state = obs["rgb"], obs["state"].float()

        enc_ctx = torch.no_grad() if not self.cfg.finetune_encoder else _nullcontext()
        with enc_ctx:
            out = self.encoder.encode_frame(self._preprocess(rgb), decode=False)
            # Deployment-consistent deterministic routing (argmax, not Gumbel).
            fast_mask = (out.router_logits.argmax(dim=-1) == 1).float()  # (B, K)
            fast_slots = out.slots * fast_mask.unsqueeze(-1)             # (B, K, D)

        z = self.z_mlp(torch.cat([fast_slots.flatten(1), state], dim=-1))
        return z

    # -- health metrics for wandb ---------------------------------------

    @torch.no_grad()
    def routing_stats(self, obs: dict) -> dict:
        out = self.encoder.encode_frame(self._preprocess(obs["rgb"]), decode=False)
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
