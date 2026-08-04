"""The frozen half of the HippoAct→RL interface, factored out so that the two
ways of running E2E-0 cannot drift apart.

E2E-0 is  z = MLP( flatten(S_fg) ⊕ q_t )  with Stage-1 frozen. That splits into

    frozen:     frame → DINOv2 → slots → argmax route → flatten(S_fg)
    trainable:  [flatten(S_fg) ⊕ q_t] → MLP → z

Because the first half is frozen, a stored frame's features never change, so it
can be evaluated **once at environment-step time** instead of once per sampled
frame per gradient update. TD-MPC2 does one update per env step over
(horizon+1) x batch = 1024 frames, so that is a ~1000x difference: measured
3882 ms/step versus ~15 ms (`experiments/scripts/bench_e2e0.py`).

Both call sites use this class, so "precompute in the env" and "encode inside
the model" are the same function by construction, not by assertion:

  * `HippoActAdapter` — frozen extractor + trainable z_mlp inside the world
    model. Needed once the encoder joins the training graph (E2E-1/E2E-2 if
    Stage-1 is unfrozen).
  * `experiments/dcs/hippoact_slots.py` — the env emits the features directly,
    TD-MPC2 then runs completely unmodified with `obs=state`, and its own state
    encoder (which ends in SimNorm, as its dynamics/reward/Q all assume) is the
    trainable MLP.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from hippoact.encoders.hippo_encoder import HippoActEncoder

# ImageNet statistics. Frames are carried as uint8 (147 KB vs 588 KB per
# 224x224x3 frame), so normalisation happens here, on device.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class SlotFeatureExtractor(nn.Module):
    """frame (B,3,H,W) uint8 → flatten(fast_slots) (B, K*D). Always frozen."""

    def __init__(self, stage1_ckpt: str, default_proprio_dim: int = 24,
                 slot_init_seed: int | None = 0,
                 num_slots: int = 16, slot_dim: int = 128, image_size: int = 224):
        super().__init__()
        # An empty path means "random init" — only useful for shape/wiring
        # tests. Every real run passes a Stage-1 checkpoint.
        payload = (torch.load(Path(stage1_ckpt), map_location="cpu", weights_only=False)
                   if stage1_ckpt else None)

        # Stage-1 checkpoints embed the architecture read back from the live
        # modules, so it cannot drift from a config file. Build from that.
        arch = (payload.get("encoder_arch", {}) or {}) if payload else {}
        if arch.get("dino_is_mock", False):
            raise ValueError(
                f"{stage1_ckpt} was trained with MockDinoV2Encoder (a random "
                "frozen CNN) — refusing to use it."
            )
        self.num_slots = int(arch.get("num_slots", num_slots))
        self.slot_dim = int(arch.get("slot_dim", slot_dim))
        self.image_size = int(arch.get("image_size", image_size))

        # The encoder's proprio width belongs to the binding transformer, which
        # E2E-0 does not use — but it must match the checkpoint or the load
        # fails. Read it off the tensor rather than assuming.
        state = payload.get("encoder", payload) if payload else None
        w = state.get("binding.proprio_proj.weight") if state else None
        enc_proprio_dim = int(w.shape[1]) if w is not None else default_proprio_dim

        self.encoder = HippoActEncoder(
            num_slots=self.num_slots,
            slot_dim=self.slot_dim,
            proprio_dim=enc_proprio_dim,
            image_size=self.image_size,
            slot_iters=int(arch.get("slot_iters", 3)),
            dino_model=str(arch.get("dino_model", "dinov2_vits14")),
        )

        # An explicit HIPPOACT_FORCE_MOCK=1 is a deliberate offline/test choice.
        # What must never pass silently is the automatic fallback that a
        # transient torch.hub error triggers — it swaps in a random CNN and
        # everything downstream becomes meaningless without any error.
        dino = getattr(self.encoder, "dino", None)
        if (dino is not None and getattr(dino, "is_mock", False)
                and os.environ.get("HIPPOACT_FORCE_MOCK", "0") != "1"):
            raise RuntimeError(
                "DINOv2 backbone is MockDinoV2Encoder (torch.hub load failed). "
                "Set HIPPOACT_STRICT_DINO=1 to surface the underlying error."
            )

        if state is not None:
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            unexpected_real = [k for k in unexpected if not k.startswith("binding")]
            assert not unexpected_real, f"ckpt mismatch: {unexpected_real[:5]}"
            missing_real = [k for k in missing
                            if not k.startswith(("binding", "dino", "_backbone"))]
            assert not missing_real, f"ckpt missing weights: {missing_real[:5]}"

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))
        self.call_count = 0

        # --- deterministic inference -----------------------------------
        # In `sampled` slot-query mode `slots_mu` is shared across slots
        # (shape (1,1,D)); the 16 slots are separated *only* by the noise drawn
        # in `sample_init`. So the encoder is not a function: encoding one frame
        # twice gives different slots (measured max|Δ| = 8.5 on this
        # checkpoint). That breaks three things at once — the buffer would hold
        # a different representation than a re-encode produces, `act()` would
        # map one observation to different latents on different calls, and the
        # MPC would plan from a noisy latent.
        #
        # Fixing one draw makes the encoder deterministic while preserving the
        # symmetry breaking it needs. The draw is arbitrary — CP5e measured
        # that different inits give materially different decompositions
        # (alpha cosine 0.189) — so `slot_init_seed` is part of the encoder's
        # identity and must be reported alongside the checkpoint.
        self.slot_init_seed = slot_init_seed
        self._fixed_init = None
        if slot_init_seed is not None:
            g = torch.Generator(device="cpu").manual_seed(int(slot_init_seed))
            sa = self.encoder.slot_attn
            mu = sa.slots_mu.detach()
            sigma = sa.slots_logsigma.detach().exp()
            eps = torch.randn((1, sa.num_slots, sa.slot_dim), generator=g)
            self.register_buffer(
                "_fixed_init_buf",
                (mu.expand(1, sa.num_slots, -1) + sigma.expand(1, sa.num_slots, -1) * eps),
            )
            self._fixed_init = True

    @property
    def feature_dim(self) -> int:
        return self.num_slots * self.slot_dim

    def preprocess(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.dtype == torch.uint8:
            rgb = rgb.float().div(255.0)
        elif rgb.max() > 1.5:                 # float but still in [0,255]
            rgb = rgb / 255.0
        return (rgb - self._mean) / self._std

    def compute(self, rgb: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) uint8 or float → (B, num_slots*slot_dim).

        No `no_grad` here, so the fine-tuning path (E2E-2) can call it inside
        its own autograd context; `forward` is the frozen entry point.
        """
        self.call_count += 1
        # decode=False: the slot decoder's recon/alpha are only needed by
        # Stage-1's losses. It materialises (B,K,N,D_v) — 6 GB at B=1024 — and
        # the router reads slot vectors only, so skipping it is exact.
        init = (self._fixed_init_buf.expand(rgb.shape[0], -1, -1)
                if self._fixed_init else None)
        out = self.encoder.encode_frame(self.preprocess(rgb), decode=False,
                                        slots_init=init)
        # Deployment-consistent deterministic routing (argmax, not Gumbel).
        fast_mask = (out.router_logits.argmax(dim=-1) == 1).float()      # (B,K)
        fast_slots = out.slots * fast_mask.unsqueeze(-1)                 # (B,K,D)
        return fast_slots.flatten(1)

    @torch.no_grad()
    def slots_and_mask(self, rgb: torch.Tensor):
        """(B,3,H,W) -> ((B,K,D) slots, (B,K) fast mask).

        E2E-1's binding transformer needs the two separately: it *masks* slow
        slots out of attention instead of zeroing them, because a zeroed slot
        is still a token a transformer attends to.  `compute` above returns the
        gated-and-flattened form E2E-0 consumes; both come from the same
        forward pass, so the two paths cannot disagree about routing.
        """
        self.call_count += 1
        init = (self._fixed_init_buf.expand(rgb.shape[0], -1, -1)
                if self._fixed_init else None)
        out = self.encoder.encode_frame(self.preprocess(rgb), decode=False,
                                        slots_init=init)
        return out.slots, (out.router_logits.argmax(dim=-1) == 1)

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.compute(rgb)
