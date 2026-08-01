#!/usr/bin/env python3
"""W1.2 result table: TD-MPC2-pixel baseline matrix vs the official reference.

Reads every ``experiments/logs/<task>/<seed>/<exp>/eval.csv`` and prints
  - the per-cell mean +- std over seeds at the requested checkpoints,
  - the clean cells side by side with tdmpc2's own ``results/tdmpc2-pixels``
    curves, which is what the P2.1 / P2.2 criteria are defined against,
  - the criterion verdicts.

These are the numbers that go into Table V.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / "experiments" / "logs"
OFFICIAL = ROOT / "third_party" / "tdmpc2" / "results" / "tdmpc2-pixels"

# UNITS. Our `step` is cfg.steps = agent steps (episode = 500). The paper and
# the shipped results CSVs report ENVIRONMENT steps (episode = 1000), i.e.
# 2x ours, because DMControl action_repeat=2 (paper Table 6; cross-checked by
# README's `task=dog-run steps=7000000` == the 14M env-step budget).
# So official(step) must be read at 2*step. Getting this wrong makes our runs
# look ~1.8x better than the reference. See TODO_RESULT.md §W1.2.0 / §W1.2.7.
ENV_STEPS_PER_AGENT_STEP = 2

# Thresholds below are in AGENT steps, derived from the official curve at the
# corresponding env step (0.9x mean, which sits at/below the worst official seed).
CRITERIA = {
    # official @200K env = 836.1 (seeds 784/834/890)
    ("walker-walk", 100_000): ("P2.1 pipeline", 700.0),
    # official @1M env = 939.6 (seeds 929/942/949)
    ("walker-walk", 500_000): ("P2.2 alignment", 850.0),
    # official @1M env = 537.3 (seeds 453/570/590)
    ("cheetah-run", 500_000): ("P2.2b alignment", 480.0),
}


def read_eval(path):
    with open(path) as f:
        return [(int(float(r["step"])), float(r["episode_reward"])) for r in csv.DictReader(f)]


def official(task, agent_step):
    """Official pixel curve at the env step matching `agent_step` (2x, see above)."""
    p = OFFICIAL / f"{task}.csv"
    if not p.exists():
        return None
    env_step = agent_step * ENV_STEPS_PER_AGENT_STEP
    by = defaultdict(dict)
    with open(p) as f:
        for r in csv.DictReader(f):
            by[int(r["step"])][r["seed"]] = float(r["reward"])
    steps = [s for s in sorted(by) if s <= env_step]
    if not steps:
        return None
    v = list(by[steps[-1]].values())
    return steps[-1], statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0), len(v)


def at_step(rows, step, tol=1):
    """Reward at the eval closest to `step` without exceeding it (within tol)."""
    cand = [r for r in rows if r[0] <= step]
    if not cand:
        return None
    s, v = cand[-1]
    return v if abs(s - step) <= max(tol, step * 0.06) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="w12_pixel")
    ap.add_argument("--steps", default="100000,250000,500000")
    args = ap.parse_args()
    checkpoints = [int(s) for s in args.steps.split(",")]

    runs = defaultdict(dict)  # task -> seed -> rows
    for csv_path in sorted(LOG_ROOT.glob(f"*/*/{args.exp}/eval.csv")):
        seed = int(csv_path.parent.parent.name)
        task = csv_path.parent.parent.parent.name
        runs[task][seed] = read_eval(csv_path)

    if not runs:
        print("no eval.csv found yet under", LOG_ROOT)
        return

    order = ["walker-walk", "dcs-easy-walker-walk", "cheetah-run", "dcs-easy-cheetah-run"]
    tasks = [t for t in order if t in runs] + [t for t in runs if t not in order]

    hdr = f"{'cell':<24} {'seeds':>6}  " + "  ".join(f"{s//1000:>6}K" for s in checkpoints)
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for task in tasks:
        seeds = sorted(runs[task])
        cells = []
        for step in checkpoints:
            vals = [v for s in seeds if (v := at_step(runs[task][s], step)) is not None]
            if vals:
                m = statistics.mean(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                results[(task, step)] = (m, sd, len(vals))
                cells.append(f"{m:6.0f}" + (f"±{sd:<3.0f}" if len(vals) > 1 else "    "))
            else:
                cells.append(f"{'--':>6}    ")
        print(f"{task:<24} {len(seeds):>6}  " + " ".join(cells))

    print()
    print("clean cells vs official tdmpc2-pixels (results/tdmpc2-pixels/*.csv).")
    print("NOTE: our step = agent steps; official CSV = env steps = 2x ours (action_repeat=2).")
    print(f"{'cell':<24} {'agent':>8} {'env':>8} {'ours':>16} {'official':>16}  {'ratio':>6}")
    print("-" * 86)
    for task in [t for t in tasks if not t.startswith("dcs-")]:
        for step in checkpoints:
            if (task, step) not in results:
                continue
            m, sd, n = results[(task, step)]
            off = official(task, step)
            if off is None:
                continue
            ostep, om, osd, on = off
            print(f"{task:<24} {step:>8} {ostep:>8} {m:>8.1f}±{sd:<5.1f}(n{n}) "
                  f"{om:>8.1f}±{osd:<5.1f}(n{on})  {m/om:>6.2f}")

    # ---- Table V column: final checkpoint re-evaluated with more episodes ----
    # The single in-training 500K eval (10 episodes) landed below the surrounding
    # plateau in several runs; see final_eval.py for why this column exists.
    cache_path = LOG_ROOT / "final_eval.json"
    final_ckpt = defaultdict(list)
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        per_task = final_ckpt
        eps = set()
        for rec in cache.values():
            if rec.get("exp_name") == args.exp:
                per_task[rec["task"]].append((rec["seed"], rec["reward"]))
                eps.add(rec["episodes"])
        if per_task:
            n_eps = eps.pop() if len(eps) == 1 else "mixed"
            print()
            print(f"TABLE V — final checkpoint, {n_eps} eval episodes (see final_eval.py):")
            print(f"{'cell':<24} {'mean':>8} {'sd':>7} {'seeds':>6}   per-seed")
            print("-" * 76)
            for task in tasks:
                if task not in per_task:
                    continue
                vs = sorted(per_task[task])
                vals = [v for _, v in vs]
                m = statistics.mean(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                detail = "  ".join(f"s{s}:{v:.0f}" for s, v in vs)
                print(f"{task:<24} {m:>8.1f} {sd:>7.1f} {len(vals):>6}   {detail}")
            # Cost of TRAINING under distraction: each arm trained AND evaluated in
            # its own regime. This is NOT the paper's retention metric, which is
            # zero-shot (train on easy, eval on none vs hard) and is W3-4 work.
            print()
            print("    training-under-distraction cost (easy-trained / clean-trained,")
            print("    each evaluated in its own regime — NOT the zero-shot retention of §Q2):")
            for base in ["walker-walk", "cheetah-run"]:
                easy = f"dcs-easy-{base}"
                if base in per_task and easy in per_task:
                    mb = statistics.mean([v for _, v in per_task[base]])
                    me = statistics.mean([v for _, v in per_task[easy]])
                    print(f"      {base:<14} {me/mb:.3f}   ({me:.0f} / {mb:.0f})")

    print()
    print("criteria (recalibrated, TODO_RESULT.md §W1.2.0):")
    print("  P2.2* judged on the Table V number (final ckpt, 30 eps) when available,")
    print("  since that is the number the paper reports — not the single 500K eval.")
    any_crit = False
    for (task, step), (name, thresh) in sorted(CRITERIA.items(), key=lambda kv: kv[0][1]):
        # prefer the final-checkpoint reading for the end-of-training criteria
        source, m, n = None, None, None
        if step == 500_000 and task in final_ckpt:
            vals = [v for _, v in final_ckpt[task]]
            m, n, source = statistics.mean(vals), len(vals), "final ckpt"
        elif (task, step) in results:
            m, _sd, n = results[(task, step)]
            source = "eval.csv"
        if m is None:
            continue
        any_crit = True
        ok = m >= thresh
        line = (f"  {name:<18} {task} @{step:,}: {m:.1f} vs >= {thresh:.0f}  "
                f"-> {'PASS' if ok else 'FAIL'}  (n={n} seeds, {source})")
        if source == "final ckpt" and (task, step) in results:
            raw, _, rn = results[(task, step)]
            line += f"   [raw 500K eval: {raw:.1f}, n={rn}]"
        print(line)
    if not any_crit:
        print("  (no checkpoint reached yet)")


if __name__ == "__main__":
    main()
