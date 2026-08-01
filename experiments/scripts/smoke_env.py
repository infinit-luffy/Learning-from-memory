#!/usr/bin/env python3
"""W1.2 pre-flight: verify the DCS integration before spending GPU-days.

Checks, in order:
  1. clean (`walker-walk`) and DCS (`dcs-easy-walker-walk`) envs both build,
     with identical obs/action specs and episode length.
  2. `dcs-none-walker-walk` is *identical* to the official `walker-walk` path
     (same reward sequence under the same seed + action sequence).  This is the
     correctness check on the shim: if it fails, the wrapper stack diverges
     from the official pixel baseline and Table V would not be comparable.
  3. The DCS background actually changes the pixels (easy != clean), and
     changes over time (dynamic=True).
  4. Env-only stepping throughput, to separate rendering cost from RL cost.

Run from the project root with the hippoact env's python.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "tdmpc2" / "tdmpc2"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402


class Cfg:
    """Minimal stand-in for the parsed TD-MPC2 config."""

    def __init__(self, task, seed=1, obs="rgb"):
        self.task = task
        self.seed = seed
        self.obs = obs
        self.multitask = False

    def get(self, k, default=None):
        return getattr(self, k, default)


def build(task, seed=1):
    from envs import make_env
    return make_env(Cfg(task, seed))


def rollout_rewards(env, actions):
    env.reset()
    rewards, frames = [], []
    for a in actions:
        obs, r, done, info = env.step(a)
        rewards.append(float(r))
        frames.append(obs.numpy().copy())
        if done:
            break
    return np.array(rewards), np.array(frames)


def main():
    print("=" * 70)
    print("1. build envs")
    clean = build("walker-walk")
    print(f"   walker-walk           obs={tuple(clean.observation_space.shape)} "
          f"act={clean.action_space.shape} T={clean.max_episode_steps}")
    dcs_none = build("dcs-none-walker-walk")
    print(f"   dcs-none-walker-walk  obs={tuple(dcs_none.observation_space.shape)} "
          f"act={dcs_none.action_space.shape} T={dcs_none.max_episode_steps}")
    t0 = time.time()
    dcs_easy = build("dcs-easy-walker-walk")
    print(f"   dcs-easy-walker-walk  obs={tuple(dcs_easy.observation_space.shape)} "
          f"act={dcs_easy.action_space.shape} T={dcs_easy.max_episode_steps} "
          f"(build {time.time()-t0:.1f}s)")

    assert clean.observation_space.shape == dcs_easy.observation_space.shape
    assert clean.max_episode_steps == dcs_easy.max_episode_steps

    print("=" * 70)
    print("2. dcs-none must be bit-identical to the official walker-walk path")
    rng = np.random.default_rng(0)
    acts = [torch.from_numpy(rng.uniform(-1, 1, clean.action_space.shape).astype(np.float32))
            for _ in range(50)]
    r_clean, f_clean = rollout_rewards(build("walker-walk", seed=1), acts)
    r_none, f_none = rollout_rewards(build("dcs-none-walker-walk", seed=1), acts)
    dr = np.abs(r_clean - r_none).max()
    dpix = np.abs(f_clean.astype(np.int32) - f_none.astype(np.int32)).max()
    print(f"   max |reward diff| = {dr:.3e}   max |pixel diff| = {dpix}")
    ok_identity = dr < 1e-6 and dpix == 0
    print(f"   -> {'PASS' if ok_identity else 'FAIL'} (official baseline path preserved)")

    print("=" * 70)
    print("3. DCS background is present and dynamic")
    r_easy, f_easy = rollout_rewards(build("dcs-easy-walker-walk", seed=1), acts)
    # Same physics seed + same actions => walker identical, only background differs.
    diff_frac = float((f_clean != f_easy).mean())
    # last frame of the 3-frame stack, across two consecutive steps
    bg_motion = float(np.abs(f_easy[10, -3:].astype(np.int32)
                             - f_easy[11, -3:].astype(np.int32)).mean())
    clean_motion = float(np.abs(f_clean[10, -3:].astype(np.int32)
                                - f_clean[11, -3:].astype(np.int32)).mean())
    print(f"   pixels differing clean vs easy : {diff_frac:.3f}")
    print(f"   mean |frame_t - frame_t+1|     : easy {bg_motion:.2f}  clean {clean_motion:.2f}")
    print(f"   reward diff clean vs easy      : {np.abs(r_clean - r_easy).max():.3e} "
          f"(should be 0 — distraction must not change the task)")
    ok_bg = diff_frac > 0.5 and bg_motion > clean_motion
    print(f"   -> {'PASS' if ok_bg else 'FAIL'}")

    print("=" * 70)
    print("4. env-only throughput (render + physics, no RL)")
    for name, env in [("walker-walk", clean), ("dcs-easy-walker-walk", dcs_easy)]:
        env.reset()
        n = 300
        t0 = time.time()
        for i in range(n):
            _, _, done, _ = env.step(acts[i % len(acts)])
            if done:
                env.reset()
        dt = time.time() - t0
        print(f"   {name:24s} {n/dt:7.1f} env-steps/s   ({1000*dt/n:.2f} ms/step)")

    print("=" * 70)
    print("RESULT:", "ALL PASS" if (ok_identity and ok_bg) else "CHECK FAILURES ABOVE")
    return 0 if (ok_identity and ok_bg) else 1


if __name__ == "__main__":
    sys.exit(main())
