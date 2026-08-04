#!/usr/bin/env python3
"""Why does E2E-0 vision-only reach only 0.35x the pixel baseline?

`SlotFeatureExtractor.compute` returns

    fast_slots = slots * (router_logits.argmax(-1) == 1)     -> flatten (K*D,)

so a slot's 128-d block is either its value or **all zeros**, decided per frame
by a hard argmax.  Two properties of that vector decide whether TD-MPC2 can
model it at all, and neither is guaranteed by Stage-1's training objective:

  1. **Gate stability.**  If a slot's fast/slow assignment flips between
     consecutive frames, 128 of the 2048 dimensions jump to/from zero in one
     env step.  TD-MPC2's dynamics head has to predict z_{t+1} from z_t; a
     discontinuity of that size is not predictable from the action.

  2. **Slot identity stability.**  Slot attention has no ordering guarantee.
     If slot 3 binds the torso at t and a leg at t+1, `flatten` scrambles the
     vector even when the scene barely moved.

This measures both on a real rollout, and contrasts them against the frame's
own physical motion (proprio delta) as the scale of "how much actually changed".

Usage:
    python experiments/scripts/diag_slot_features.py --steps 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party" / "tdmpc2" / "tdmpc2"), str(ROOT),
                str(ROOT / "hippoact")]

import numpy as np
import torch

DEFAULT_CKPT = ROOT / "hippoact" / "outputs" / "stage1_seed1" / "ckpt_final.pt"


def build_env(difficulty, seed):
    from dm_control.suite.wrappers import action_scale
    from envs.dmcontrol import DMControlWrapper
    from experiments.dcs.dcs_env import make_dm_env

    env = make_dm_env(difficulty, "walker", "walk", seed)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)
    return DMControlWrapper(env, "walker")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "e2e0"
                                         / "slot_feature_diagnosis.json"))
    args = ap.parse_args()

    from hippoact.adapters.slot_features import SlotFeatureExtractor

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext = SlotFeatureExtractor(args.ckpt, slot_init_seed=0).to(dev).eval()
    K, D = ext.num_slots, ext.slot_dim

    env = build_env(args.difficulty, args.seed)
    state = env.reset()

    slots, masks, proprio = [], [], []
    for _ in range(args.steps):
        frame = env.render(width=224, height=224)
        rgb = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)))
        with torch.no_grad():
            out = ext.encoder.encode_frame(
                ext.preprocess(rgb.unsqueeze(0).to(dev)), decode=False,
                slots_init=ext._fixed_init_buf)
            s = out.slots[0]                                   # (K, D)
            m = (out.router_logits[0].argmax(-1) == 1)         # (K,) bool
        slots.append(s.cpu().numpy())
        masks.append(m.cpu().numpy())
        proprio.append(np.asarray(state, dtype=np.float64))
        state, _, done, _ = env.step(env.action_space.sample())
        if done:
            state = env.reset()

    S = np.stack(slots)                     # (T, K, D)
    M = np.stack(masks)                     # (T, K) bool
    P = np.stack(proprio)                   # (T, 24)
    T = len(S)

    # --- 1. gate stability -------------------------------------------------
    flips = M[1:] != M[:-1]                                  # (T-1, K)
    n_fast = M.sum(1)
    gate = dict(
        num_slots=int(K),
        fast_slots_per_frame_mean=float(n_fast.mean()),
        fast_slots_per_frame_min=int(n_fast.min()),
        fast_slots_per_frame_max=int(n_fast.max()),
        zero_fraction_of_obs=float(1.0 - n_fast.mean() / K),
        slot_gate_flips_per_step_mean=float(flips.sum(1).mean()),
        frames_with_any_flip=float((flips.any(1)).mean()),
        dims_zeroed_or_revived_per_step=float(flips.sum(1).mean() * D),
    )

    # --- 2. how big is a flip compared with ordinary motion? ---------------
    # The observation TD-MPC2 sees is fast_slots.flatten().
    obs = (S * M[..., None]).reshape(T, K * D)
    d_obs = np.linalg.norm(np.diff(obs, axis=0), axis=1)
    flip_step = flips.any(1)
    # Same quantity with the gate removed: slots only, no masking.
    d_raw = np.linalg.norm(np.diff(S.reshape(T, K * D), axis=0), axis=1)
    d_proprio = np.linalg.norm(np.diff(P, axis=0), axis=1)
    jump = dict(
        obs_delta_mean=float(d_obs.mean()),
        obs_delta_on_flip_steps=float(d_obs[flip_step].mean()) if flip_step.any() else None,
        obs_delta_on_stable_steps=float(d_obs[~flip_step].mean()) if (~flip_step).any() else None,
        ungated_slot_delta_mean=float(d_raw.mean()),
        proprio_delta_mean=float(d_proprio.mean()),
        # Correlation between how much the physics moved and how much the
        # observation moved.  A usable observation tracks the state.
        corr_obs_delta_vs_proprio_delta=float(np.corrcoef(d_obs, d_proprio)[0, 1]),
        corr_ungated_delta_vs_proprio_delta=float(np.corrcoef(d_raw, d_proprio)[0, 1]),
    )

    # --- 3. slot identity stability ---------------------------------------
    # For each consecutive pair, is slot k at t+1 closest to slot k at t, or to
    # some other slot?  `same_slot_is_nearest` = 1.0 means identity is stable.
    nearest_is_same = []
    for t in range(T - 1):
        a, b = S[t], S[t + 1]                                  # (K, D)
        dist = np.linalg.norm(b[:, None, :] - a[None, :, :], axis=-1)   # (K,K)
        nearest_is_same.append((dist.argmin(1) == np.arange(K)).mean())
    identity = dict(
        same_slot_is_nearest_mean=float(np.mean(nearest_is_same)),
        # How distinguishable are slots at all?  If every slot collapsed to the
        # same vector, "nearest" is meaningless.
        mean_pairwise_slot_distance=float(np.mean([
            np.linalg.norm(S[t][:, None, :] - S[t][None, :, :], axis=-1)[
                np.triu_indices(K, 1)].mean() for t in range(T)])),
        mean_slot_norm=float(np.linalg.norm(S, axis=-1).mean()),
    )

    # --- 4. per-slot fast rate --------------------------------------------
    per_slot_fast_rate = M.mean(0).tolist()

    result = dict(ckpt=args.ckpt, difficulty=args.difficulty, steps=T,
                  gate=gate, jump=jump, identity=identity,
                  per_slot_fast_rate=[round(x, 3) for x in per_slot_fast_rate])
    Path(args.out).write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
