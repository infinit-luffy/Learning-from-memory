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


def _paint_disks(img: np.ndarray, disks) -> np.ndarray:
    """Overlay a list of (cx, cy, r, col) disks onto an image in place."""
    size = img.shape[0]
    y, x = np.ogrid[:size, :size]
    for cx, cy, r, col in disks:
        cx = int(np.clip(cx, r, size - r))
        cy = int(np.clip(cy, r, size - r))
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        for c in range(3):
            img[mask, c] = col[c]
    return img


def _paint_rect(img: np.ndarray, rect) -> np.ndarray:
    if rect is None:
        return img
    x0, y0, w, h, col = rect
    img[y0:y0 + h, x0:x0 + w] = col
    return img


def make_clip(size: int, rng: np.random.Generator, clip_len: int) -> list[np.ndarray]:
    """Return a list of ``clip_len`` frames with:
      - background: fixed across the clip (should be routed slow)
      - rectangle 'gripper base': fixed across the clip (should be slow)
      - disks: initial position + small per-step velocity (should be fast)
    """
    bg_fn = BG_FNS[rng.integers(0, len(BG_FNS))]
    bg = bg_fn(size, rng)

    if rng.random() < 0.5:
        w, h = rng.integers(20, 40, size=2)
        x0 = rng.integers(0, size - w)
        y0 = rng.integers(0, size - h)
        rect_col = rng.random(3) * 0.3 + 0.1
        rect = (int(x0), int(y0), int(w), int(h), rect_col)
    else:
        rect = None

    n_disks = int(rng.integers(2, 6))
    disks_init = []
    velocities = []
    for _ in range(n_disks):
        cx, cy = rng.integers(30, size - 30, size=2)
        r = int(rng.integers(8, 22))
        col = rng.random(3)
        disks_init.append([float(cx), float(cy), r, col])
        vx, vy = rng.normal(0.0, 3.0, size=2)
        velocities.append((float(vx), float(vy)))

    frames = []
    for t in range(clip_len):
        img = bg.copy()
        img = _paint_rect(img, rect)
        disks_t = [
            (init[0] + v[0] * t, init[1] + v[1] * t, init[2], init[3])
            for init, v in zip(disks_init, velocities)
        ]
        img = _paint_disks(img, disks_t)
        frames.append(np.clip(img, 0, 1))
    return frames


def make_frame(size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a single scene (no temporal structure) — used only by --flat."""
    return make_clip(size, rng, clip_len=1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/frames/synthetic")
    ap.add_argument("--n-clips", type=int, default=1250,
                    help="Number of clips to generate (default 1250 × 4 = 5000 frames)")
    ap.add_argument("--clip-len", type=int, default=4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flat", action="store_true",
                    help="Legacy: dump all frames flat into --out (no temporal structure)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.flat:
        total = args.n_clips * args.clip_len
        for i in range(total):
            frame = make_frame(args.size, rng)
            Image.fromarray((frame * 255).astype(np.uint8)).save(
                out / f"frame_{i:06d}.png"
            )
            if (i + 1) % 500 == 0:
                print(f"  wrote {i + 1} / {total}")
        print(f"Done. {total} flat frames in {out.resolve()}")
    else:
        total = args.n_clips * args.clip_len
        for c in range(args.n_clips):
            clip_dir = out / f"clip_{c:06d}"
            clip_dir.mkdir(exist_ok=True)
            for t, frame in enumerate(make_clip(args.size, rng, args.clip_len)):
                Image.fromarray((frame * 255).astype(np.uint8)).save(
                    clip_dir / f"frame_{t:04d}.png"
                )
            if (c + 1) % 100 == 0:
                print(f"  wrote {c + 1} / {args.n_clips} clips  "
                      f"({(c + 1) * args.clip_len} / {total} frames)")
        print(f"Done. {args.n_clips} clips × {args.clip_len} frames = {total} "
              f"total in {out.resolve()}")


if __name__ == "__main__":
    main()
