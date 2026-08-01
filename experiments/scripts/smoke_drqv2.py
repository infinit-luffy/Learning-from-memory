#!/usr/bin/env python3
"""W1.3 pre-flight: verify the DrQ-v2 DCS integration before spending GPU-days.

Mirrors `smoke_env.py` on the TD-MPC2 side:
  1. clean and distracted envs build with identical specs;
  2. `distraction='none'` is bit-identical to DrQ-v2's stock path — if this
     fails, the clean arm is no longer the official DrQ-v2 configuration;
  3. the distraction is visible and dynamic, and does not change the reward
     (same seed + same actions must give the same reward sequence);
  4. env-only throughput.

Run from the project root with the hippoact env's python.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "drqv2"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import dmc  # noqa: E402  (from third_party/drqv2)

FRAME_STACK, ACTION_REPEAT = 3, 2


def rollout(env, actions):
    ts = env.reset()
    rewards, frames = [], []
    for a in actions:
        ts = env.step(a)
        rewards.append(float(ts.reward))
        frames.append(ts.observation.copy())
        if ts.last():
            break
    return np.array(rewards), np.array(frames)


def main():
    print("=" * 70)
    print("1. build envs")
    clean = dmc.make("walker_walk", FRAME_STACK, ACTION_REPEAT, 1)
    none_ = dmc.make("walker_walk", FRAME_STACK, ACTION_REPEAT, 1, "none")
    easy = dmc.make("walker_walk", FRAME_STACK, ACTION_REPEAT, 1, "easy")
    for tag, e in [("stock", clean), ("distraction=none", none_), ("distraction=easy", easy)]:
        spec = e.observation_spec()
        print(f"   {tag:<18} obs={spec.shape} act={e.action_spec().shape}")
    assert clean.observation_spec().shape == easy.observation_spec().shape

    print("=" * 70)
    print("2. distraction='none' must equal DrQ-v2's stock path")
    rng = np.random.default_rng(0)
    acts = [rng.uniform(-1, 1, clean.action_spec().shape).astype(np.float32)
            for _ in range(50)]
    r_stock, f_stock = rollout(dmc.make("walker_walk", FRAME_STACK, ACTION_REPEAT, 1), acts)
    r_none, f_none = rollout(dmc.make("walker_walk", FRAME_STACK, ACTION_REPEAT, 1, "none"), acts)
    dr = float(np.abs(r_stock - r_none).max())
    dp = int(np.abs(f_stock.astype(np.int32) - f_none.astype(np.int32)).max())
    ok_id = dr < 1e-6 and dp == 0
    print(f"   max |reward diff| = {dr:.3e}   max |pixel diff| = {dp}   -> "
          f"{'PASS' if ok_id else 'FAIL'}")

    print("=" * 70)
    print("3. distraction is present, dynamic, and reward-neutral")
    r_easy, f_easy = rollout(dmc.make("walker_walk", FRAME_STACK, ACTION_REPEAT, 1, "easy"), acts)
    diff_frac = float((f_stock != f_easy).mean())
    bg = float(np.abs(f_easy[10, -3:].astype(np.int32) - f_easy[11, -3:].astype(np.int32)).mean())
    cl = float(np.abs(f_stock[10, -3:].astype(np.int32) - f_stock[11, -3:].astype(np.int32)).mean())
    dr2 = float(np.abs(r_stock - r_easy).max())
    ok_bg = diff_frac > 0.5 and bg > cl and dr2 < 1e-6
    print(f"   pixels differing clean vs easy : {diff_frac:.3f}")
    print(f"   mean |frame_t - frame_t+1|     : easy {bg:.2f}  clean {cl:.2f}")
    print(f"   reward diff clean vs easy      : {dr2:.3e} (must be 0)")
    print(f"   -> {'PASS' if ok_bg else 'FAIL'}")

    print("=" * 70)
    print("4. env-only throughput")
    for tag, e in [("clean", clean), ("easy", easy)]:
        e.reset()
        n = 200
        t0 = time.time()
        for i in range(n):
            ts = e.step(acts[i % len(acts)])
            if ts.last():
                e.reset()
        dt = time.time() - t0
        print(f"   {tag:<8} {n/dt:7.1f} env-steps/s  ({1000*dt/n:.2f} ms/step)")

    print("=" * 70)
    print("RESULT:", "ALL PASS" if (ok_id and ok_bg) else "CHECK FAILURES ABOVE")
    return 0 if (ok_id and ok_bg) else 1


if __name__ == "__main__":
    sys.exit(main())
