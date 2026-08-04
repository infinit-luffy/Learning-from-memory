#!/usr/bin/env python3
"""E2E-1 smoke test — the pre-registered checks from the TODO §R4.10 ruling.

    1. intra-frame slot shuffle  -> z bit-identical   (permutation equivariance)
    2. frame-order shuffle       -> z must change     (temporal encoding works)
    3. env obs shape             == encoder in_dim    (the silent-mismatch guard)
    4. env-side stacking         == a hand-built stack of the same frames
    5. throughput                -> the fast path is still ~25 SPS

Check 1 is the property E2E-0's index-flatten lacked; check 2 is what makes
the frame stack more than three copies of one frame.  Both must hold, and they
pull in opposite directions, so passing one is not evidence for the other.

Usage:
    python experiments/scripts/smoke_e2e1.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party" / "tdmpc2" / "tdmpc2"), str(ROOT),
                str(ROOT / "hippoact")]

import numpy as np
import torch

CKPT = str(ROOT / "hippoact" / "outputs" / "stage1_seed1" / "ckpt_final.pt")
T, K, DS = 3, 16, 128
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main():
    from hippoact.encoders.binding import VisionBindingEncoder

    torch.manual_seed(0)
    enc = VisionBindingEncoder(slot_dim=DS, num_slots=K, t_window=T,
                               out_dim=512).eval()
    blk = K * DS + K
    assert enc.in_dim == T * blk

    # A batch whose slots and masks are both non-degenerate.
    B = 4
    slots = torch.randn(B, T, K, DS)
    mask = (torch.rand(B, T, K) > 0.35).float()
    mask[:, :, 0] = 1.0                       # never a fully-masked frame

    def pack(sl, mk):
        return torch.cat([sl.reshape(B, T, K * DS), mk], dim=-1).reshape(B, -1)

    with torch.no_grad():
        z = enc(pack(slots, mask))

        # 1. shuffle slots *within* each frame, same permutation per frame.
        perm = torch.randperm(K)
        z_slot = enc(pack(slots[:, :, perm], mask[:, :, perm]))

        # 2. reverse the frame order.
        rev = torch.arange(T - 1, -1, -1)
        z_time = enc(pack(slots[:, rev], mask[:, rev]))

    d_slot = (z - z_slot).abs().max().item()
    d_time = (z - z_time).abs().max().item()
    check("slot shuffle leaves z unchanged", d_slot < 1e-5, f"max|Δ| = {d_slot:.2e}")
    check("frame shuffle changes z", d_time > 1e-3, f"max|Δ| = {d_time:.2e}")

    # ---- environment side -------------------------------------------------
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    from omegaconf import OmegaConf
    from experiments.dcs.dcs_env import make_env

    cfg = OmegaConf.create(dict(
        task="dcs-easy-walker-walk", obs="state", seed=1,
        hippoact_precompute=CKPT, hippoact_slot_init_seed=0,
        hippoact_include_proprio=False, hippoact_num_frames=T,
        hippoact_emit_mask=True, hippoact_image_size=224))
    env = make_env(cfg)
    obs = env.reset()

    check("env obs dim == encoder in_dim", obs.shape[0] == enc.in_dim,
          f"{obs.shape[0]} vs {enc.in_dim}")

    # 4. after reset the window holds three copies of the same frame, so the
    #    three blocks must be identical; after one step only the newest differs.
    blocks = obs.view(T, blk)
    same_at_reset = torch.allclose(blocks[0], blocks[2])
    obs1 = env.step(env.action_space.sample())[0].view(T, blk)
    slid = torch.allclose(obs1[0], blocks[1]) and torch.allclose(obs1[1], blocks[2])
    check("reset fills the window with one frame", same_at_reset)
    check("step slides the window by one", slid,
          "oldest two of t+1 == newest two of t")

    # mask entries really are 0/1 and not all-on
    m = obs1[:, K * DS:]
    check("mask is binary and non-trivial",
          bool(((m == 0) | (m == 1)).all()) and 0 < m.mean().item() < 1,
          f"fast fraction = {m.mean().item():.2f}")

    # 5. throughput
    n = 60
    t0 = time.perf_counter()
    for _ in range(n):
        env.step(env.action_space.sample())
    sps = n / (time.perf_counter() - t0)
    check("env throughput >= 20 SPS", sps >= 20, f"{sps:.1f} SPS")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
