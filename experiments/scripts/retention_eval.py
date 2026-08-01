#!/usr/bin/env python3
"""Q2 zero-shot retention: evaluate every trained checkpoint on {none, easy, hard}.

Paper protocol (§IV.C): train on DMC-Distracting Easy, then evaluate zero-shot
on {None, Easy, Hard} with no further training;
    retention rate = return_hard / return_none.

No training is involved, so this costs hours rather than days -- it puts the
Q2 baseline row in place immediately.  Clean-trained checkpoints are evaluated
on the same grid as a contrast (train-clean -> eval-hard is the harder shift).

The checkpoint is loaded into an env of a *different* difficulty than it was
trained on; obs/action shapes are identical across difficulties (3x64x64, same
action_dim), so the model loads unchanged.

Results cached in `experiments/logs/retention_eval.json`; re-running only
evaluates missing (checkpoint, difficulty) pairs.

Usage (from project root):
    python experiments/scripts/retention_eval.py --gpus 0,1 --episodes 30
    python experiments/scripts/retention_eval.py --report        # print only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_DIR = PROJECT_ROOT / "experiments"
LOG_ROOT = LAUNCH_DIR / "logs"
EVALUATE_PY = PROJECT_ROOT / "third_party" / "tdmpc2" / "tdmpc2" / "evaluate.py"
CACHE = LOG_ROOT / "retention_eval.json"
PYTHON = os.environ.get(
    "HIPPOACT_PYTHON",
    "/usr1/home/s125mdg56_03/miniconda3/envs/hippoact/bin/python",
)
CUDA_TO_EGL = {0: 2, 1: 3, 2: 0, 3: 1}          # measured; see egl_probe.py
REWARD_RE = re.compile(r"R:\s*(-?[\d.]+)")
DIFFICULTIES = ["none", "easy", "hard"]


def base_task(task):
    """`dcs-easy-walker-walk` -> `walker-walk`; `walker-walk` -> `walker-walk`."""
    return task[len("dcs-easy-"):] if task.startswith("dcs-easy-") else task


def eval_task_name(base, difficulty):
    # `none` goes through TD-MPC2's own dm_control path, bit-identical to
    # the official baseline (verified in smoke_env.py).
    return base if difficulty == "none" else f"dcs-{difficulty}-{base}"


def find_checkpoints(exp_name):
    out = []
    for ckpt in sorted(LOG_ROOT.glob(f"*/*/{exp_name}/models/final.pt")):
        seed = int(ckpt.parent.parent.parent.name)
        task = ckpt.parent.parent.parent.parent.name
        out.append(dict(train_task=task, seed=seed, exp_name=exp_name, ckpt=str(ckpt)))
    return out


def run_one(job, episodes, gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL="egl",
               MUJOCO_EGL_DEVICE_ID=str(CUDA_TO_EGL[gpu]), PYTHONUNBUFFERED="1")
    tag = f"{job['train_task']}_s{job['seed']}_on_{job['difficulty']}"
    cmd = [
        PYTHON, str(EVALUATE_PY),
        f"task={job['eval_task']}", "obs=rgb", "model_size=5",
        f"seed={job['seed']}", f"eval_episodes={episodes}", "save_video=false",
        f"checkpoint={job['ckpt']}",
        f"hydra.run.dir={LOG_ROOT / 'hydra_retention' / tag}",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(LAUNCH_DIR), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    m = REWARD_RE.findall(proc.stdout)
    if proc.returncode != 0 or not m:
        tail = "\n".join(proc.stdout.strip().splitlines()[-12:])
        raise RuntimeError(f"eval failed: {tag}\n{tail}")
    return float(m[-1]), time.time() - t0


def report(cache, exp_name):
    by = defaultdict(dict)          # train_task -> difficulty -> [rewards]
    for rec in cache.values():
        if rec.get("exp_name") != exp_name:
            continue
        by[rec["train_task"]].setdefault(rec["difficulty"], []).append(rec["reward"])

    if not by:
        print("nothing evaluated yet")
        return
    order = ["dcs-easy-walker-walk", "walker-walk", "dcs-easy-cheetah-run", "cheetah-run"]
    tasks = [t for t in order if t in by] + [t for t in by if t not in order]

    print("zero-shot evaluation of trained checkpoints (30 episodes each)")
    print(f"{'trained on':<24} " + "".join(f"{'eval:'+d:>16}" for d in DIFFICULTIES)
          + f"{'retention':>12}")
    print("-" * 88)
    for task in tasks:
        cells = []
        for d in DIFFICULTIES:
            vs = by[task].get(d)
            cells.append(f"{statistics.mean(vs):8.1f}±{statistics.stdev(vs):<5.1f}"
                         if vs and len(vs) > 1 else
                         (f"{vs[0]:8.1f}      " if vs else f"{'--':>8}      "))
        none_v, hard_v = by[task].get("none"), by[task].get("hard")
        ret = (f"{statistics.mean(hard_v)/statistics.mean(none_v):>12.3f}"
               if none_v and hard_v else f"{'--':>12}")
        print(f"{task:<24} " + "".join(cells) + ret)
    print()
    print("retention = mean(eval on hard) / mean(eval on none), per paper §IV.C.")
    print("Rows starting with `dcs-easy-` are the protocol rows (trained on Easy);")
    print("clean-trained rows are the contrast (a larger distribution shift).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="w12_pixel")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--difficulties", default=",".join(DIFFICULTIES))
    ap.add_argument("--report", action="store_true", help="print cached results and exit")
    args = ap.parse_args()

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if args.report:
        report(cache, args.exp)
        return 0

    gpus = [int(g) for g in args.gpus.split(",")]
    difficulties = args.difficulties.split(",")
    ckpts = find_checkpoints(args.exp)
    if not ckpts:
        print("no checkpoints found")
        return 1

    jobs = []
    for c in ckpts:
        for d in difficulties:
            key = f"{c['train_task']}|{c['seed']}|{d}|{args.exp}"
            if key in cache and cache[key].get("episodes") == args.episodes:
                continue
            jobs.append(dict(c, difficulty=d, key=key,
                             eval_task=eval_task_name(base_task(c["train_task"]), d)))

    print(f"{len(ckpts)} checkpoints x {len(difficulties)} difficulties; "
          f"{len(jobs)} to evaluate ({len(ckpts)*len(difficulties)-len(jobs)} cached)")
    if not jobs:
        report(cache, args.exp)
        return 0

    lock_msgs = []

    def worker(idx_job):
        idx, job = idx_job
        gpu = gpus[idx % len(gpus)]
        reward, secs = run_one(job, args.episodes, gpu)
        return job, reward, secs, gpu

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for job, reward, secs, gpu in pool.map(worker, enumerate(jobs)):
            cache[job["key"]] = dict(
                train_task=job["train_task"], seed=job["seed"],
                difficulty=job["difficulty"], eval_task=job["eval_task"],
                exp_name=job["exp_name"], reward=reward,
                episodes=args.episodes, eval_seconds=round(secs, 1))
            CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))
            print(f"  {job['train_task']:<24} s{job['seed']} on {job['difficulty']:<5}"
                  f" gpu={gpu}  R={reward:8.1f}  ({secs/60:.1f} min)", flush=True)

    print()
    report(cache, args.exp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
