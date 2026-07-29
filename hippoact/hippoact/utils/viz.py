"""Slot alpha visualization utilities.

Used by the training loop and the sanity tests to eyeball slot alphas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def overlay_slot_alpha(
    img: torch.Tensor,          # (3, H, W)  in [0, 1]
    alpha: torch.Tensor,        # (K, N)      softmax over slot dim
    grid_h: int = 14,
    grid_w: int = 14,
    upsample: int = 16,
) -> np.ndarray:
    """Return a numpy array of shape (K, H, W, 3) — original blended per-slot mask."""
    import torch.nn.functional as F

    K, N = alpha.shape
    assert N == grid_h * grid_w
    a = alpha.reshape(K, 1, grid_h, grid_w)
    a = F.interpolate(a, scale_factor=upsample, mode="bilinear", align_corners=False)
    a = a.squeeze(1)                                         # (K, H, W)
    a = a / (a.amax(dim=(-1, -2), keepdim=True) + 1e-8)

    base = img.permute(1, 2, 0).cpu().numpy()                # (H, W, 3) in [0, 1]
    out = []
    for k in range(K):
        m = a[k].cpu().numpy()[..., None]                    # (H, W, 1)
        # Red overlay proportional to attention.
        red_layer = np.array([1.0, 0.0, 0.0])
        blend = base * (1 - 0.5 * m) + red_layer * (0.5 * m)
        out.append(np.clip(blend, 0, 1))
    return np.stack(out, axis=0)                             # (K, H, W, 3)


def save_slot_grid(
    img: torch.Tensor, alpha: torch.Tensor, out_path: str | Path, upsample: int = 16
) -> Path:
    """Save an 8×2 grid PNG of slot masks overlaid on the observation."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlays = overlay_slot_alpha(img, alpha, upsample=upsample)
    K = overlays.shape[0]
    cols = 8
    rows = (K + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for k in range(K):
        r, c = divmod(k, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        ax.imshow(overlays[k])
        ax.set_title(f"slot {k}")
        ax.axis("off")
    for k in range(K, rows * cols):
        r, c = divmod(k, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return out_path
