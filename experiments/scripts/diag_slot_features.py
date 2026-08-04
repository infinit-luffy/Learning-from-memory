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
    from envs.wrappers.timeout import Timeout
    from experiments.dcs.dcs_env import make_dm_env

    env = make_dm_env(difficulty, "walker", "walk", seed)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)
    env = DMControlWrapper(env, "walker")
    # DMControlWrapper never sets `done`; production relies on this wrapper for
    # the 500-step episode boundary.  Without it, stepping past the dm_control
    # episode end returns reward=None.
    return Timeout(env, max_episode_steps=500)


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
    slots_co, masks_co = [], []          # carryover_norm inference (R4.9 line A)
    sa = ext.encoder.slot_attn
    mu, sigma = sa.slots_mu, sa.slots_logsigma.exp()
    carried = ext._fixed_init_buf        # first frame starts from the fixed draw

    for _ in range(args.steps):
        frame = env.env.render(width=224, height=224)
        rgb = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)))
        x = ext.preprocess(rgb.unsqueeze(0).to(dev))
        with torch.no_grad():
            out = ext.encoder.encode_frame(x, decode=False,
                                           slots_init=ext._fixed_init_buf)
            s = out.slots[0]                                   # (K, D)
            m = (out.router_logits[0].argmax(-1) == 1)         # (K,) bool

            # `carryover_norm` (stage1.py:184, CP5g): the previous frame's
            # output slots, re-standardised onto the learned init manifold
            # N(mu, sigma), become this frame's init.  That is what preserves
            # slot identity across frames while keeping the near-symmetric
            # statistics that drive attention competition.  Measured here on
            # the *existing* `shared`-trained checkpoint -- if identity
            # recovers anyway, a Stage-1 retrain has evidence behind it; if
            # not, line A closes (R4.9 ruling 3).
            oc = ext.encoder.encode_frame(x, decode=False, slots_init=carried)
            sc = oc.slots
            z = (sc - sc.mean(-1, keepdim=True)) / (sc.std(-1, keepdim=True) + 1e-6)
            carried = (mu + sigma * z).detach()
            mc = (oc.router_logits[0].argmax(-1) == 1)

        slots.append(s.cpu().numpy())
        masks.append(m.cpu().numpy())
        slots_co.append(sc[0].cpu().numpy())
        masks_co.append(mc.cpu().numpy())
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

    # --- 5. readout comparison (R4.9 ruling 1 pre-flight) ------------------
    # `corr(d readout, d proprio)` is the number that decides whether TD-MPC2's
    # consistency objective is learnable at all.  Pre-registered target for
    # E2E-0p: **> 0.3**.  Measured here BEFORE any run is launched.
    #
    # Four readouts x two inference modes.  The pooled ones are permutation
    # invariant by construction, so slot re-ordering cannot scramble them.
    Sc, Mc = np.stack(slots_co), np.stack(masks_co)

    def readouts(S_, M_):
        n_fast = np.maximum(M_.sum(1, keepdims=True), 1)       # (T,1)
        gated = S_ * M_[..., None]                             # (T,K,D)
        return {
            "flatten_fast (E2E-0, current)": gated.reshape(T, K * D),
            "mean_over_fast (E2E-0p)": gated.sum(1) / n_fast,
            "sum_fast_over_K": gated.sum(1) / K,
            "mean_all_slots (no gate)": S_.mean(1),
        }

    def corr_with_proprio(X):
        d = np.linalg.norm(np.diff(X, axis=0), axis=1)
        if d.std() < 1e-12:
            return None
        return float(np.corrcoef(d, d_proprio)[0, 1])

    readout_corr = {}
    for mode, (S_, M_) in {"shared (as trained)": (S, M),
                           "carryover_norm": (Sc, Mc)}.items():
        readout_corr[mode] = {name: corr_with_proprio(X)
                              for name, X in readouts(S_, M_).items()}

    # Does carryover_norm inference restore slot identity? (ruling 3)
    def same_nearest(X):
        out = []
        for t in range(len(X) - 1):
            d = np.linalg.norm(X[t + 1][:, None, :] - X[t][None, :, :], axis=-1)
            out.append((d.argmin(1) == np.arange(K)).mean())
        return float(np.mean(out))

    carryover = dict(
        same_slot_is_nearest_shared=identity["same_slot_is_nearest_mean"],
        same_slot_is_nearest_carryover_norm=same_nearest(Sc),
        fast_slots_per_frame_carryover_norm=float(Mc.sum(1).mean()),
    )

    result = dict(ckpt=args.ckpt, difficulty=args.difficulty, steps=T,
                  gate=gate, jump=jump, identity=identity,
                  readout_corr_vs_proprio=readout_corr, carryover=carryover,
                  per_slot_fast_rate=[round(x, 3) for x in per_slot_fast_rate])
    Path(args.out).write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print()
    print("readout corr(delta readout, delta proprio) — E2E-0p needs > 0.3")
    print(f"{'readout':<34}" + "".join(f"{m:>22}" for m in readout_corr))
    for name in next(iter(readout_corr.values())):
        row = "".join(f"{(readout_corr[m][name] if readout_corr[m][name] is not None else float('nan')):>22.3f}"
                      for m in readout_corr)
        print(f"{name:<34}{row}")
    print()
    print(f"same_slot_is_nearest   shared {carryover['same_slot_is_nearest_shared']:.3f}"
          f"   carryover_norm {carryover['same_slot_is_nearest_carryover_norm']:.3f}")
    print(f"wrote {args.out}")



def probe_main():
    """Linear probe: is the walker's state *in* the representation at all?

    `corr(d readout, d proprio)` compares magnitudes of change, and walker's
    24-d proprio is 9-d velocities that a single frame barely shows -- so a low
    correlation is weak evidence.  A ridge probe answers the question directly:
    fit readout -> proprio on held-in frames, report R^2 held-out.

    The control that makes it interpretable is the DINOv2 patch mean, i.e. the
    same backbone *before* slot attention.  If DINOv2 probes well and the slots
    do not, slot attention is destroying the state; if neither probes, the
    problem is upstream of HippoAct entirely.

    Usage:  python experiments/scripts/diag_slot_features.py probe --steps 1500
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "results" / "e2e0"
                                         / "linear_probe.json"))
    args = ap.parse_args()

    from hippoact.adapters.slot_features import SlotFeatureExtractor

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext = SlotFeatureExtractor(args.ckpt, slot_init_seed=0).to(dev).eval()
    K, D = ext.num_slots, ext.slot_dim
    env = build_env(args.difficulty, args.seed)
    state = env.reset()

    S, M, P, Dino, episode = [], [], [], [], []
    ep_id = 0
    for _ in range(args.steps):
        frame = env.env.render(width=224, height=224)
        rgb = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)))
        x = ext.preprocess(rgb.unsqueeze(0).to(dev))
        with torch.no_grad():
            feats = ext.encoder.dino(x)                    # (1, N_patch, D_v)
            out = ext.encoder.encode_frame(x, decode=False,
                                           slots_init=ext._fixed_init_buf)
        Dino.append(feats[0].mean(0).cpu().numpy())
        S.append(out.slots[0].cpu().numpy())
        M.append((out.router_logits[0].argmax(-1) == 1).cpu().numpy())
        P.append(np.asarray(state, dtype=np.float64))
        episode.append(ep_id)
        state, _, done, _ = env.step(env.action_space.sample())
        if done:
            state = env.reset()
            ep_id += 1

    S, M, P, Dino = np.stack(S), np.stack(M), np.stack(P), np.stack(Dino)
    T = len(S)
    n_fast = np.maximum(M.sum(1, keepdims=True), 1)
    gated = S * M[..., None]

    flat = gated.reshape(T, K * D)
    pooled = gated.sum(1) / n_fast

    # The pixel baseline is `Pixels(env, cfg, num_frames=3)` -- a **3-frame
    # stack**, so velocity is observable to it.  `HippoActSlots` emits a single
    # frame's slots, so velocity is not observable to E2E-0 at all.  Stack the
    # same features and probe again: if velocity comes back, the E2E-0 deficit
    # is a wiring omission, not a property of the representation.
    ep = np.asarray(episode)

    def stack(X, n):
        """[x_{t-n+1} .. x_t], only where the whole window is one episode."""
        idx = np.arange(n - 1, T)
        idx = idx[np.all([ep[idx - j] == ep[idx] for j in range(n)], axis=0)]
        return np.concatenate([X[idx - j] for j in reversed(range(n))], axis=1), idx

    flat2, i2 = stack(flat, 2)
    flat3, i3 = stack(flat, 3)
    pool3, _ = stack(pooled, 3)

    feats = {
        "flatten_fast (E2E-0, 1 frame)": (flat, None),
        "mean_over_fast (E2E-0p, 1 frame)": (pooled, None),
        "mean_all_slots (1 frame)": (S.mean(1), None),
        "DINOv2 patch mean (control, 1 frame)": (Dino, None),
        "flatten_fast x2 frames": (flat2, i2),
        "flatten_fast x3 frames (= pixel stack)": (flat3, i3),
        "mean_over_fast x3 frames": (pool3, i3),
    }

    # walker proprio: dm_control orders it [orientations(14), height(1),
    # velocity(9)].  Positions are what a single frame can show; velocity needs
    # at least two.  Reported separately -- the 9 velocity dims dominate the
    # variance, so `proprio_all` alone hides which half is missing.
    targets = {"proprio_all(24)": P, "proprio_pos(15)": P[:, :15],
               "proprio_vel(9)": P[:, 15:]}

    # Two things a temporal split gets wrong here, both of which produced
    # R^2 < 0 for *every* feature including the DINOv2 control on the first
    # attempt:
    #   * episodes differ.  1500 steps is 3 episodes, and under `easy` each
    #     draws a different DAVIS background -- a time-ordered split tests on
    #     an unseen background, so it measures transfer, not encoding.
    #   * p >> n.  flatten_fast is 2048-d against 1050 training frames, so a
    #     single fixed lambda is nowhere near the right regulariser.
    # Random split + a lambda sweep fixes both.  The reported number is the
    # best test R^2 over the sweep, i.e. an **optimistic upper bound** -- which
    # is the right instrument for "is the state in there at all": if even the
    # best case fails, the information is absent.
    rng = np.random.default_rng(0)
    perm = rng.permutation(T)
    ntr = int(T * 0.7)
    tr, te = perm[:ntr], perm[ntr:]
    LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5]

    def ridge_r2(X, Y, idx):
        # `idx` maps rows of a stacked feature back to their frame; the target
        # is sliced the same way and the row split is drawn over X's own rows,
        # so stacked and unstacked variants stay comparable.
        if idx is not None:
            Y = Y[idx]
        n = len(X)
        pm = np.random.default_rng(0).permutation(n)
        tr_, te_ = pm[:int(n * 0.7)], pm[int(n * 0.7):]
        Xtr, Xte, Ytr, Yte = X[tr_], X[te_], Y[tr_], Y[te_]
        mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr, Xte = (Xtr - mx) / sx, (Xte - mx) / sx
        my = Ytr.mean(0)
        G, B = Xtr.T @ Xtr, Xtr.T @ (Ytr - my)
        eye = np.eye(Xtr.shape[1])
        ss_tot = ((Yte - Yte.mean(0)) ** 2).sum()
        best, best_lam = -np.inf, None
        for lam in LAMBDAS:
            W = np.linalg.solve(G + lam * eye, B)
            r2 = float(1 - ((Yte - (Xte @ W + my)) ** 2).sum() / ss_tot)
            if r2 > best:
                best, best_lam = r2, lam
        return round(best, 4), best_lam

    res, lams = {}, {}
    for name, (X, idx) in feats.items():
        res[name], lams[name] = {}, {}
        for t, Y in targets.items():
            res[name][t], lams[name][t] = ridge_r2(X, Y, idx)

    out = dict(ckpt=args.ckpt, difficulty=args.difficulty, steps=T,
               split="random", train_frames=ntr, test_frames=T - ntr,
               lambda_grid=LAMBDAS, best_lambda=lams,
               note="best test R^2 over the lambda sweep = optimistic upper bound",
               probe_r2=res)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"ridge probe, random split {ntr} train / {T-ntr} test, "
          f"best-of-sweep held-out R^2 (upper bound)")
    print(f"{'feature':<40}" + "".join(f"{t:>18}" for t in targets))
    for name, r in res.items():
        print(f"{name:<40}" + "".join(f"{r[t]:>18.3f}" for t in targets))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        probe_main()
    else:
        main()

