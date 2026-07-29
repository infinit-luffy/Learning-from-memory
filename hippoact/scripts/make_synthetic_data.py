#!/usr/bin/env python3
"""Generate synthetic manipulation-like frames so Stage 1 can end-to-end run
without waiting for real robot data.

Each frame is a textured background with a small number of colored shapes
placed at random positions and orientations. Varies backgrounds and shape
palettes across frames so slot attention has genuine variability to learn from.

Usage:
    python scripts/make_synthetic_data.py --out data/frames/synthetic --n 5000

5000 frames takes ~90s on any modern CPU. Enough to run Stage 1 as a
pipeline sanity check (not enough to produce a strong representation).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _stripe_bg(size: int, rng: np.random.Generator) -> np.ndarray:
    """Horizontally striped background with a random hue."""
    freq = rng.uniform(1.0, 6.0)
    phase = rng.uniform(0, 2 * np.pi)
    ys = np.linspace(0, freq * np.pi, size) + phase
    stripes = 0.5 + 0.15 * np.sin(ys)[:, None]
    base_hue = rng.random(3)
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        img[..., c] = np.clip(base_hue[c] * stripes[..., 0], 0, 1)
    return img


def _checker_bg(size: int, rng: np.random.Generator) -> np.ndarray:
    """Checker-pattern background with two random colors."""
    cell = rng.integers(16, 48)
    c1, c2 = rng.random(3), rng.random(3)
    x = np.arange(size) // cell
    y = np.arange(size)[:, None] // cell
    mask = ((x + y) % 2 == 0).astype(np.float32)[..., None]
    return mask * c1 + (1 - mask) * c2


def _noise_bg(size: int, rng: np.random.Generator) -> np.ndarray:
    """Perlin-ish smooth noise background."""
    low = rng.random((size // 16, size // 16, 3))
    img = np.array(Image.fromarray((low * 255).astype(np.uint8))
                   .resize((size, size), Image.BILINEAR)) / 255.0
    return img.astype(np.float32)


BG_FNS = [_stripe_bg, _checker_bg, _noise_bg]


def _place_disks(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sprinkle a few colored disks — foreground candidates for slot attention."""
    size = img.shape[0]
    n = rng.integers(2, 6)
    y, x = np.ogrid[:size, :size]
    for _ in range(n):
        cx, cy = rng.integers(20, size - 20, size=2)
        r = rng.integers(8, 22)
        col = rng.random(3)
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        for c in range(3):
            img[mask, c] = col[c]
    return img


def _place_rect(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Optional rectangle 'gripper-like' element."""
    size = img.shape[0]
    if rng.random() < 0.5:
        w, h = rng.integers(20, 40, size=2)
        x0 = rng.integers(0, size - w)
        y0 = rng.integers(0, size - h)
        col = rng.random(3) * 0.3 + 0.1     # darker
        img[y0:y0 + h, x0:x0 + w] = col
    return img


def make_frame(size: int, rng: np.random.Generator) -> np.ndarray:
    """Return one HxWx3 float32 image in [0,1]."""
    bg_fn = BG_FNS[rng.integers(0, len(BG_FNS))]
    img = bg_fn(size, rng)
    img = _place_rect(img, rng)
    img = _place_disks(img, rng)
    return np.clip(img, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/frames/synthetic")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        frame = make_frame(args.size, rng)
        Image.fromarray((frame * 255).astype(np.uint8)).save(
            out / f"frame_{i:06d}.png"
        )
        if (i + 1) % 500 == 0:
            print(f"  wrote {i + 1} / {args.n}")

    print(f"Done. {args.n} frames in {out.resolve()}")


if __name__ == "__main__":
    main()
