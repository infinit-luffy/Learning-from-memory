#!/usr/bin/env python3
"""Where does E2E-0's per-step cost go?

TD-MPC2 runs one `act()` and one `update()` per environment step. `act()`
encodes a single frame; `update()` encodes (horizon+1) x batch_size frames —
1024 at the default 3/256. With a 224x224 DINOv2 ViT-S/14 that ratio is the
whole story, so measure both.

Usage:  HIPPOACT_STRICT_DINO=1 python experiments/scripts/bench_e2e0.py [ckpt]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hippoact"))
sys.path.insert(0, str(ROOT))

from hippoact.adapters.tdmpc2_adapter import HippoActAdapter, AdapterConfig  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else str(
    ROOT / "hippoact" / "outputs" / "stage1_seed1" / "ckpt_final.pt")


def bench(fn, n):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def main():
    ad = HippoActAdapter(AdapterConfig(
        stage1_ckpt=CKPT, proprio_dim=24, latent_dim=512)).cuda().eval()

    results = {}
    for B, n, tag in [(1, 20, "act(): 1 frame"),
                      (256, 5, "update(): one timestep, 256"),
                      (1024, 3, "update(): full 4x256 batch")]:
        obs = {"rgb": torch.randint(0, 255, (B, 3, 224, 224), dtype=torch.uint8,
                                    device="cuda"),
               "state": torch.zeros(B, 24, device="cuda")}
        dt = bench(lambda: ad(obs), n)
        results[B] = dt
        print(f"  {tag:<30} {dt*1000:8.1f} ms  ({B/dt:8.0f} frames/s)")

    step = results[1] + results[1024]
    print()
    print(f"  per env step = act(1) + update(1024) = {step*1000:.0f} ms "
          f"-> {1/step:.2f} SPS")
    print(f"  500K steps   = {500_000*step/3600:.0f} h = {500_000*step/86400:.1f} days")
    print()
    print(f"  update() is {results[1024]/step*100:.1f}% of the cost.")
    print("  At E2E-0 the Stage-1 encoder is FROZEN, so a stored frame's slots")
    print("  never change: encoding at insertion is exactly equal to encoding at")
    print("  sample time. Caching slots would remove DINOv2 from update()")
    print(f"  entirely (-> ~{1/results[1]:.0f} SPS bound) and shrink the buffer from")
    print("  147 KB/frame to 16x128 floats = 8 KB. Not valid for E2E-1/2, which")
    print("  fine-tune the encoder.")


if __name__ == "__main__":
    main()
