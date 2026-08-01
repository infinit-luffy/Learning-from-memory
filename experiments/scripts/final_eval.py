#!/usr/bin/env python3
"""Table V numbers: re-evaluate each run's final checkpoint with more episodes.

Why this exists instead of just reading the last row of eval.csv
----------------------------------------------------------------
TD-MPC2's in-training evaluation uses 10 episodes, and the *last* one landed
well below the surrounding plateau in several W1.2 runs (walker-walk seed 1:
961.4 +- 5.0 over 300-475K, then 818.6 at 500K).  Re-evaluating the saved
`final.pt` in a fresh process reproduced it (846.9 over 30 episodes), so the
policy genuinely ends slightly worse -- it is not an evaluation artifact.

Taking the single 500K point would understate our own pixel baseline by ~9%,
which biases every later comparison *in HippoAct's favour*.  So Table V reports
the delivered policy evaluated with 30 episodes (3x the in-training sample),
and the drop itself is reported as an observation.

Results are cached in `experiments/logs/final_eval.json`; re-running only
evaluates runs that are missing or whose checkpoint is newer than the cache.

Usage (from project root):
    python experiments/scripts/final_eval.py                 # all finished runs
    python experiments/scripts/final_eval.py --episodes 30 --gpus 0,1
    python experiments/scripts/final_eval.py --only walker-walk --force
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_DIR = PROJECT_ROOT / "experiments"
LOG_ROOT = LAUNCH_DIR / "logs"
EVALUATE_PY = PROJECT_ROOT / "third_party" / "tdmpc2" / "tdmpc2" / "evaluate.py"
CACHE = LOG_ROOT / "final_eval.json"
PYTHON = os.environ.get(
    "HIPPOACT_PYTHON",
    "/usr1/home/s125mdg56_03/miniconda3/envs/hippoact/bin/python",
)

# EGL enumerates devices in its own order; see run_queue.py for the measurement.
CUDA_TO_EGL = {0: 2, 1: 3, 2: 0, 3: 1}

REWARD_RE = re.compile(r"R:\s*(-?[\d.]+)")


def find_runs(exp_name, only=None):
    """Runs that have a saved final checkpoint."""
    runs = []
    for ckpt in sorted(LOG_ROOT.glob(f"*/*/{exp_name}/models/final.pt")):
        exp_dir = ckpt.parent.parent
        seed = int(exp_dir.parent.name)
        task = exp_dir.parent.parent.name
        if only and only not in task:
            continue
        runs.append(dict(task=task, seed=seed, exp_name=exp_name, ckpt=str(ckpt),
                         mtime=ckpt.stat().st_mtime))
    return runs


def evaluate(run, episodes, gpu):
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        MUJOCO_GL="egl",
        MUJOCO_EGL_DEVICE_ID=str(CUDA_TO_EGL[gpu]),
        PYTHONUNBUFFERED="1",
    )
    hydra_dir = LOG_ROOT / "hydra_eval" / "{}_s{}".format(run["task"], run["seed"])
    cmd = [
        PYTHON, str(EVALUATE_PY),
        f"task={run['task']}", "obs=rgb", "model_size=5",
        f"seed={run['seed']}",
        f"eval_episodes={episodes}",
        "save_video=false",
        f"checkpoint={run['ckpt']}",
        f"hydra.run.dir={hydra_dir}",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(LAUNCH_DIR), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = proc.stdout
    matches = REWARD_RE.findall(out)
    if proc.returncode != 0 or not matches:
        tail = "\n".join(out.strip().splitlines()[-15:])
        raise RuntimeError(f"evaluate.py failed for {run['task']} seed {run['seed']}:\n{tail}")
    return float(matches[-1]), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="w12_pixel")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--only", default=None, help="substring filter on task name")
    ap.add_argument("--force", action="store_true", help="re-evaluate even if cached")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    runs = find_runs(args.exp, args.only)
    if not runs:
        print("no final.pt checkpoints found yet")
        return

    for i, run in enumerate(runs):
        key = f"{run['task']}|{run['seed']}|{run['exp_name']}"
        cached = cache.get(key)
        if (cached and not args.force
                and cached.get("episodes") == args.episodes
                and cached.get("ckpt_mtime") == run["mtime"]):
            print(f"  [cached] {run['task']:<24} seed={run['seed']}  "
                  f"R={cached['reward']:.1f}")
            continue
        gpu = gpus[i % len(gpus)]
        print(f"  [eval  ] {run['task']:<24} seed={run['seed']} gpu={gpu} "
              f"episodes={args.episodes} ...", flush=True)
        reward, secs = evaluate(run, args.episodes, gpu)
        cache[key] = dict(task=run["task"], seed=run["seed"], exp_name=run["exp_name"],
                          reward=reward, episodes=args.episodes,
                          ckpt_mtime=run["mtime"], eval_seconds=round(secs, 1))
        CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))
        print(f"             -> R={reward:.1f}  ({secs/60:.1f} min)", flush=True)

    print(f"\ncache: {CACHE}  ({len(cache)} runs)")


if __name__ == "__main__":
    sys.exit(main())
