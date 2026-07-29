"""Frozen DINOv2 backbone with a mock fallback for offline / test environments."""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn


class MockDinoV2Encoder(nn.Module):
    """Deterministic mock that mimics DINOv2 patch-token output shape.

    Used in tests and environments without internet access to `torch.hub`.
    Behaves like a frozen 4-layer CNN + linear projection to `feat_dim`.
    """

    def __init__(self, image_size: int = 224, patch_size: int = 14, feat_dim: int = 384):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.n_patches = self.grid ** 2
        self.feat_dim = feat_dim

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(256, feat_dim, 3, stride=1, padding=1),
        )
        # Freeze weights to mirror real DINOv2 use.
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)  →  (B, n_patches, feat_dim)
        feats = self.stem(x)                          # (B, feat_dim, h, w)
        feats = torch.nn.functional.adaptive_avg_pool2d(feats, (self.grid, self.grid))
        return feats.flatten(2).transpose(1, 2)       # (B, n_patches, feat_dim)


class DinoV2Encoder(nn.Module):
    """Frozen DINOv2 ViT wrapper.

    Falls back to `MockDinoV2Encoder` if `torch.hub` cannot load the model
    (e.g. no internet, CI, sandbox), which lets the rest of the pipeline
    boot without the real weights. Set ``HIPPOACT_FORCE_MOCK=1`` to force it.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        image_size: int = 224,
        patch_size: int = 14,
    ):
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = patch_size
        self._backbone: Optional[nn.Module] = None
        self._is_mock: bool = False
        self.feat_dim: int = {"dinov2_vits14": 384, "dinov2_vitb14": 768,
                              "dinov2_vitl14": 1024}.get(model_name, 384)

        if os.environ.get("HIPPOACT_FORCE_MOCK", "0") == "1":
            self._install_mock()
        else:
            try:
                self._backbone = torch.hub.load(
                    "facebookresearch/dinov2", model_name, verbose=False
                )
                self._backbone.eval()
                for p in self._backbone.parameters():
                    p.requires_grad = False
            except Exception as e:  # noqa: BLE001
                print(f"[HippoAct] Falling back to MockDinoV2Encoder ({e}).")
                self._install_mock()

    def _install_mock(self) -> None:
        self._backbone = MockDinoV2Encoder(
            self.image_size, self.patch_size, self.feat_dim
        )
        self._is_mock = True

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    @property
    def n_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, n_patches, feat_dim) patch tokens."""
        if self._is_mock:
            return self._backbone(x)
        out = self._backbone.forward_features(x)
        return out["x_norm_patchtokens"]
