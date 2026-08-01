#!/usr/bin/env python3
"""Run a queue of TD-MPC2 jobs across a fixed set of GPUs.

Keeps ``--slots-per-gpu`` processes alive on each GPU and pulls the next job
off the queue whenever one finishes.  Jobs that already have an ``eval.csv``
reaching the target step count are skipped, so the script is restartable.

Usage (from the project root)::

    python experiments/scripts/run_queue.py --jobs experiments/scripts/w12_jobs.txt \
        --gpus 0,1 --slots-per-gpu 3

Job file format — one job per line, ``#`` comments allowed::

    <task> <seed> <steps> <exp_name>
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Wrapper around tdmpc2's train.py that enables SIGUSR1 stack dumps and
# PR_SET_PTRACER; needed because the server blocks py-spy from attaching to a
# running job (ptrace_scope=1, no sudo). Identical argv, no algorithm change.
TRAIN_PY = PROJECT_ROOT / "experiments" / "scripts" / "train_dbg.py"
LAUNCH_DIR = PROJECT_ROOT / "experiments"          # -> logs land in experiments/logs/
CONSOLE_DIR = LAUNCH_DIR / "logs" / "console"
PYTHON = os.environ.get(
    "HIPPOACT_PYTHON",
    "/usr1/home/s125mdg56_03/miniconda3/envs/hippoact/bin/python",
)

# EGL enumerates devices in its OWN order, which on this box is rotated by two
# relative to CUDA / nvidia-smi.  Setting MUJOCO_EGL_DEVICE_ID=<cuda id> puts
# the MuJoCo render context on a *different* physical card -- silently, since
# rendering still works.  Measured with experiments/scripts/egl_probe.py:
#
#     MUJOCO_EGL_DEVICE_ID  0 1 2 3   ->   nvidia-smi GPU  2 3 0 1
#
# Re-measure with egl_probe.py if the host or driver changes.
CUDA_TO_EGL = {0: 2, 1: 3, 2: 0, 3: 1}


def parse_jobs(path):
    jobs = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        task, seed, steps, exp_name = line.split()
        jobs.append(dict(task=task, seed=int(seed), steps=int(steps), exp_name=exp_name))
    return jobs


def work_dir(job):
    return LAUNCH_DIR / "logs" / job["task"] / str(job["seed"]) / job["exp_name"]


def last_eval_step(job):
    """Highest step present in the run's eval.csv, or -1 if there is none."""
    csv_path = work_dir(job) / "eval.csv"
    if not csv_path.exists():
        return -1
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        return max(int(float(r["step"])) for r in rows) if rows else -1
    except Exception:
        return -1


def build_cmd(job, gpu):
    hydra_dir = LAUNCH_DIR / "logs" / "hydra" / f"{job['task']}_s{job['seed']}_{job['exp_name']}"
    return [
        PYTHON, str(TRAIN_PY),
        f"task={job['task']}",
        "obs=rgb",
        "model_size=5",
        f"steps={job['steps']}",
        f"seed={job['seed']}",
        f"exp_name={job['exp_name']}",
        "enable_wandb=false",
        "save_video=false",
        "save_agent=true",
        "eval_freq=25000",
        f"hydra.run.dir={hydra_dir}",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--slots-per-gpu", type=int, default=3)
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--stagger", type=float, default=25.0,
                    help="seconds between launches (torch.compile is CPU-heavy at startup)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = parse_jobs(args.jobs)
    CONSOLE_DIR.mkdir(parents=True, exist_ok=True)

    pending, skipped = [], []
    for job in jobs:
        if last_eval_step(job) >= job["steps"]:
            skipped.append(job)
        else:
            pending.append(job)

    print(f"[queue] {len(jobs)} jobs; {len(skipped)} already complete; {len(pending)} to run")
    print(f"[queue] gpus={gpus} slots_per_gpu={args.slots_per_gpu} "
          f"=> {len(gpus) * args.slots_per_gpu} concurrent")
    for job in pending:
        print(f"        - {job['task']:26s} seed={job['seed']} steps={job['steps']}")
    if args.dry_run:
        print("[queue] dry run, nothing launched")
        for job in pending[:1]:
            print("[queue] example cmd:", " ".join(build_cmd(job, gpus[0])))
        return

    running = []          # list of (proc, job, gpu, log_file_handle, t_start)
    slots = {g: 0 for g in gpus}
    queue = list(pending)
    interrupted = False

    def _sigint(signum, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[queue] interrupt received; terminating children")
        for proc, job, *_ in running:
            proc.terminate()

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    while (queue or running) and not interrupted:
        # Reap finished jobs
        for entry in list(running):
            proc, job, gpu, fh, t0 = entry
            if proc.poll() is not None:
                fh.close()
                slots[gpu] -= 1
                running.remove(entry)
                mins = (time.time() - t0) / 60
                status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
                print(f"[queue] {status} {job['task']} seed={job['seed']} "
                      f"gpu={gpu} {mins:.1f} min  last_eval_step={last_eval_step(job)}",
                      flush=True)

        # Fill free slots
        for gpu in gpus:
            while queue and slots[gpu] < args.slots_per_gpu:
                job = queue.pop(0)
                log_path = CONSOLE_DIR / f"{job['task']}_s{job['seed']}_{job['exp_name']}.log"
                fh = open(log_path, "a", buffering=1)
                fh.write(f"\n===== launch {datetime.now().isoformat()} gpu={gpu} =====\n")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["MUJOCO_GL"] = "egl"
                env["MUJOCO_EGL_DEVICE_ID"] = str(CUDA_TO_EGL[gpu])
                env["OMP_NUM_THREADS"] = "4"
                env["MKL_NUM_THREADS"] = "4"
                # without this the child's prints sit in a 8KB pipe buffer and
                # a multi-day run looks dead for hours
                env["PYTHONUNBUFFERED"] = "1"
                proc = subprocess.Popen(
                    build_cmd(job, gpu), cwd=str(LAUNCH_DIR), env=env,
                    stdout=fh, stderr=subprocess.STDOUT,
                )
                slots[gpu] += 1
                running.append((proc, job, gpu, fh, time.time()))
                print(f"[queue] START {job['task']} seed={job['seed']} gpu={gpu} "
                      f"pid={proc.pid} -> {log_path}", flush=True)
                time.sleep(args.stagger)

        time.sleep(args.poll)

    for proc, *_ in running:
        proc.wait()
    print("[queue] all done")


if __name__ == "__main__":
    sys.exit(main())
