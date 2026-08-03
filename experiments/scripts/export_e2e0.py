#!/usr/bin/env python3
"""Export the W2.1 E2E-0 family from run artifacts into `experiments/results/e2e0/`.

`experiments/logs/` is gitignored, so without this the numbers quoted in
TODO_RESULT §R4.8 have no committed source.  Safe to re-run while training is
in progress -- it exports whatever each run has reached so far and records the
step it reached.

Three arms, all `dcs-easy-walker-walk`, RL seed 1, only the observation differs:

    e2e0_s{1,3,5}     2048-d slots + 24-d proprio   (the confounded arm)
    e2e0vis_s{1,3,5}  2048-d slots, no proprio      (comparable to the pixel baseline)
    proprio_only      24-d proprio, no vision       (the ceiling proprio alone buys)

Also exports the official TD-MPC2 state-obs curve for walker-walk as an
external reference, converted from env steps to agent steps.

Usage:  python experiments/scripts/export_e2e0.py
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / "experiments" / "logs" / "dcs-easy-walker-walk" / "1"
PIXEL_LOG = ROOT / "experiments" / "logs" / "dcs-easy-walker-walk"
OFFICIAL_STATE = (ROOT / "third_party" / "tdmpc2" / "results" / "tdmpc2"
                  / "walker-walk.csv")
OUT = ROOT / "experiments" / "results" / "e2e0"

ARMS = {
    "e2e0": "slots+proprio",
    "e2e0vis": "slots only",
    "proprio_only": "proprio only",
}
# Pre-registered thresholds (TODO_RESULT §R4.7.3), set against the pixel baseline.
CRITERION = {50_000: 171, 100_000: 313, 200_000: 458, 250_000: 540}
STEPS = [0, 50_000, 100_000, 150_000, 200_000, 250_000, 300_000, 350_000,
         400_000, 450_000, 500_000]


def read_curve(path):
    with open(path) as f:
        return {int(float(r["step"])): float(r["episode_reward"])
                for r in csv.DictReader(f)}


def arm_of(exp_name):
    """`e2e0vis_s3` -> `e2e0vis`; `proprio_only` -> `proprio_only`."""
    return exp_name.rsplit("_s", 1)[0] if "_s" in exp_name else exp_name


def official_proprio():
    """Official 3-seed state-obs walker-walk, env step -> agent step."""
    by = defaultdict(list)
    with open(OFFICIAL_STATE) as f:
        for r in csv.DictReader(f):
            by[int(float(r["step"])) // 2].append(float(r["reward"]))
    return {k: statistics.mean(v) for k, v in sorted(by.items())}


def main():
    (OUT / "curves").mkdir(parents=True, exist_ok=True)

    curves, meta = {}, {}
    for d in sorted(LOG_ROOT.glob("*/eval.csv")):
        exp = d.parent.name
        if arm_of(exp) not in ARMS:
            continue
        c = read_curve(d)
        if not c:
            continue
        curves[exp] = c
        meta[exp] = dict(arm=arm_of(exp), reached_step=max(c),
                         last_reward=round(c[max(c)], 1))
        with open(OUT / "curves" / f"{exp}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "episode_reward"])
            w.writerows((int(s), c[s]) for s in sorted(c))

    # The pixel baseline this arm is supposed to beat (same task, 3 seeds).
    pixel = defaultdict(list)
    for d in sorted(PIXEL_LOG.glob("*/w12_pixel/eval.csv")):
        for s, v in read_curve(d).items():
            pixel[s].append(v)
    pixel = {k: statistics.mean(v) for k, v in pixel.items()}

    proprio_ref = official_proprio()
    with open(OUT / "proprio_reference.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent_step", "episode_reward"])
        w.writerows(proprio_ref.items())

    with open(OUT / "progress.json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    # Rendered comparison: arm means at the pre-registered read-out points.
    by_arm = defaultdict(list)
    for exp, c in curves.items():
        by_arm[arm_of(exp)].append(c)
    cols = [a for a in ARMS if a in by_arm]
    lines = ["| agent step | " + " | ".join(f"{ARMS[a]} (n={len(by_arm[a])})" for a in cols)
             + " | pixel baseline | 判据 | 官方纯 proprio |",
             "|---:" * (len(cols) + 4) + "|"]
    for s in STEPS:
        cells = []
        for a in cols:
            vs = [c[s] for c in by_arm[a] if s in c]
            cells.append(f"{statistics.mean(vs):.1f}" if vs else "—")
        cells.append(f"{pixel[s]:.1f}" if s in pixel else "—")
        cells.append(str(CRITERION.get(s, "—")))
        cells.append(f"{proprio_ref[s]:.1f}" if s in proprio_ref else "—")
        if any(c != "—" for c in cells[:len(cols)]):
            lines.append(f"| {s:,} | " + " | ".join(cells) + " |")
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n")

    print(f"exported {len(curves)} runs to {OUT.relative_to(ROOT)}")
    for exp in sorted(meta):
        m = meta[exp]
        print(f"  {exp:<16} {m['arm']:<14} {m['reached_step']:>7,} steps  "
              f"R={m['last_reward']:7.1f}")
    print()
    print((OUT / "comparison.md").read_text())


if __name__ == "__main__":
    main()
