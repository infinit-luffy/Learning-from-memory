#!/usr/bin/env python3
"""L3 · S1 sanity: Slot Attention overfits a single scene in ~500 steps.

Runs in <2 minutes on any GPU. Great smoke test before starting a real run.
Uses a random synthetic image so no dataset is required.

Usage:
    python scripts/overfit_test.py
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from hippoact.encoders.hippo_encoder import HippoActEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(0)
    encoder = HippoActEncoder(image_size=args.image_size).to(args.device)
    encoder.train()

    # Synthetic scene: a few colored blobs on a textured background.
    img = _make_synthetic_scene(args.image_size).to(args.device).unsqueeze(0)  # (1,3,H,W)

    optim = torch.optim.AdamW(
        list(encoder.slot_attn.parameters()) + list(encoder.slot_decoder.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    losses = []
    t0 = time.time()
    for step in range(args.steps):
        with torch.no_grad():
            target = encoder.dino(img)
        slots = encoder.slot_attn(target)
        recon, _ = encoder.slot_decoder(slots)
        loss = F.mse_loss(recon, target.detach())

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        losses.append(loss.item())
        if step % 50 == 0:
            print(f"step {step:4d}  L_slot = {loss.item():.5f}")

    elapsed = time.time() - t0
    initial, final = losses[0], losses[-1]
    print("-" * 50)
    print(f"elapsed: {elapsed:.1f}s   initial: {initial:.4f}   final: {final:.4f}")
    print(f"ratio initial/final: {initial / max(final, 1e-8):.1f}×")
    # Pass criteria from testing_strategy.md §S1
    assert final < 0.01, f"S1 FAIL: final loss {final:.4f} > 0.01"
    assert initial > 10 * final, f"S1 FAIL: loss barely decreased"
    print("✔  S1 (overfit single image) PASSED")


def _make_synthetic_scene(size: int) -> torch.Tensor:
    """Simple textured scene: colored disks on a striped background."""
    import numpy as np

    rng = np.random.default_rng(42)
    img = np.zeros((3, size, size), dtype=np.float32)
    # Stripes background
    ys = np.linspace(0, 4 * np.pi, size)
    img[0] = 0.3 + 0.1 * np.sin(ys)[None, :]
    img[1] = 0.4
    img[2] = 0.5 + 0.1 * np.cos(ys)[None, :]

    # Colored disks
    for _ in range(4):
        cx, cy = rng.integers(20, size - 20, size=2)
        r = rng.integers(10, 25)
        col = rng.random(3)
        y, x = np.ogrid[:size, :size]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        for c in range(3):
            img[c][mask] = col[c]
    return torch.from_numpy(img).clamp(0, 1)


if __name__ == "__main__":
    main()
