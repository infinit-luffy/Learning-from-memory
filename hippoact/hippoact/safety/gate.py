"""Runtime OOD safety gate.

Reconstruction residual of DINOv2 patch features from slots. Insensitive to
nuisance appearance change (lighting, mild texture) because DINOv2 features
already are; fires on structural OOD (novel objects, occlusion, camera
failure).
"""
from __future__ import annotations

from typing import Iterable

import torch

from hippoact.encoders.hippo_encoder import HippoActEncoder


class SafetyGate:
    def __init__(self, encoder: HippoActEncoder):
        self.encoder = encoder
        self.tau_safe: float = float("inf")   # set by calibrate()
        self._n_calibrated: int = 0

    @torch.no_grad()
    def _residual(self, img: torch.Tensor) -> torch.Tensor:
        """Return per-sample residual scalar (B,) for a batch of frames."""
        feats = self.encoder.dino(img)                            # (B, N, D)
        slots = self.encoder.slot_attn(feats)
        recon, _ = self.encoder.slot_decoder(slots)
        return (recon - feats).pow(2).mean(dim=(-1, -2))          # (B,)

    def calibrate(
        self,
        loader: Iterable,
        quantile: float = 0.95,
        image_key: str = "img",
        device: str = "cuda",
        max_frames: int = 5000,
    ) -> float:
        """Set tau_safe to `quantile`-th percentile of in-distribution residuals.

        `loader` yields dict-like batches with `image_key` (B,3,H,W) tensors.
        """
        self.encoder.eval()
        residuals = []
        n_seen = 0
        for batch in loader:
            img = batch[image_key].to(device) if isinstance(batch, dict) else batch.to(device)
            r = self._residual(img)
            residuals.append(r.cpu())
            n_seen += r.numel()
            if n_seen >= max_frames:
                break
        cat = torch.cat(residuals)
        self.tau_safe = float(cat.quantile(quantile).item())
        self._n_calibrated = int(cat.numel())
        return self.tau_safe

    @torch.no_grad()
    def is_ood(self, img: torch.Tensor) -> bool:
        """True if the current frame should trigger fallback.

        Accepts (3,H,W) or (1,3,H,W).
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)
        r = self._residual(img).item()
        return r > self.tau_safe

    def state_dict(self) -> dict:
        return {"tau_safe": self.tau_safe, "n_calibrated": self._n_calibrated}

    def load_state_dict(self, sd: dict) -> None:
        self.tau_safe = float(sd["tau_safe"])
        self._n_calibrated = int(sd.get("n_calibrated", 0))
