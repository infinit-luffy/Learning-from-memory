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
import re
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
DRQV2_TRAIN_PY = PROJECT_ROOT / "third_party" / "drqv2" / "train.py"
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
    if job.get("runner") == "drqv2":
        # DrQ-v2 uses Path.cwd() as work_dir, which hydra sets to run.dir.
        return (LAUNCH_DIR / "logs" / "drqv2"
                / f"{job['task']}_{job['exp_name']}" / str(job["seed"]))
    return LAUNCH_DIR / "logs" / job["task"] / str(job["seed"]) / job["exp_name"]


def last_eval_step(job):
    """Highest step present in the run's eval.csv, or -1 if there is none.

    Both trainers write an `eval.csv`; `step` is agent steps in each. DrQ-v2
    additionally writes `frame` (= step * action_repeat) — we compare on
    `step` so job files stay in one unit across runners.
    """
    csv_path = work_dir(job) / "eval.csv"
    if not csv_path.exists():
        return -1
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        return max(int(float(r["step"])) for r in rows) if rows else -1
    except Exception:
        return -1


def claim(job):
    """Take an exclusive claim on a job's output dir; False if someone holds it.

    Two queues over overlapping job files once started the *same* run twice,
    both writing the same work_dir. The queue keeps its job list in memory, so
    editing the file cannot fix that after the fact — the claim has to live on
    disk, next to the output the job would corrupt.

    A claim whose pid is gone is stale (crash / kill) and gets taken over.
    """
    wd = work_dir(job)
    wd.mkdir(parents=True, exist_ok=True)
    lock = wd / ".claim"
    if lock.exists():
        try:
            pid = int(lock.read_text().split()[0])
            os.kill(pid, 0)          # raises unless the holder is alive
            return False
        except (ValueError, IndexError, ProcessLookupError):
            pass                     # stale claim, fall through and take it
        except PermissionError:
            return False             # alive, owned by another user
    lock.write_text(f"{os.getpid()} {datetime.now().isoformat()}\n")
    return True


def release(job):
    lock = work_dir(job) / ".claim"
    try:
        if lock.exists() and int(lock.read_text().split()[0]) == os.getpid():
            lock.unlink()
    except Exception:
        pass


STAGE1_CKPT = (PROJECT_ROOT / "hippoact" / "outputs"
               / "stage1_seed{seed}" / "ckpt_final.pt")


def build_cmd(job, gpu):
    if job.get("runner") in ("e2e0", "e2e1"):
        # W2.1 E2E-0 fast path: the frozen encoder lives in the env, so
        # TD-MPC2 runs with plain `obs=state` and no modification at all.
        # The Stage-1 seed comes from the exp_name suffix (e2e0_s3 -> seed 3)
        # so the job file stays one line per run.
        m = re.search(r"_s(\d+)$", job["exp_name"])
        assert m, f"e2e0 exp_name must end in _s<stage1 seed>: {job['exp_name']}"
        ckpt = str(STAGE1_CKPT).format(seed=m.group(1))
        assert Path(ckpt).exists(), f"missing Stage-1 checkpoint {ckpt}"
        hydra_dir = LAUNCH_DIR / "logs" / "hydra" / f"{job['task']}_s{job['seed']}_{job['exp_name']}"
        # E2E-1 (TODO §R4.10): same frozen extractor in the env, but it emits a
        # 3-frame stack plus the fast/slow mask, and TD-MPC2's state encoder is
        # swapped for the trainable permutation-equivariant binding transformer.
        # Vision-only is not optional here -- on locomotion proprio is the full
        # state (§R4.8), so Q2 would be vacuous with it.
        e2e1 = ["encoder_type=hippoact_binding", "hippoact_num_frames=3",
                "hippoact_emit_mask=true", "hippoact_include_proprio=false"] \
            if job["runner"] == "e2e1" else []
        return [
            PYTHON, str(TRAIN_PY),
            f"task={job['task']}",
            "obs=state",
            f"hippoact_precompute={ckpt}",
            # Explicit, so the saved hydra config records which slot init the
            # run used — it is part of the encoder's identity (§R4.6.4).
            "hippoact_slot_init_seed=0",
            "model_size=5",
            f"steps={job['steps']}",
            f"seed={job['seed']}",
            f"exp_name={job['exp_name']}",
            "enable_wandb=false",
            "save_video=false",
            "save_agent=true",
            "eval_freq=25000",
            f"hydra.run.dir={hydra_dir}",
        ] + e2e1 + job.get("extra", [])
    if job.get("runner") == "drqv2":
        # `steps` is agent steps for both runners; DrQ-v2 counts frames.
        task, distraction = job["task"].rsplit("__", 1)
        return [
            PYTHON, str(DRQV2_TRAIN_PY),
            # DrQ-v2 declares the task as `task@_global_`; hydra 1.3 rejects the
            # bare `task=` form their README uses (which assumed hydra 1.1).
            f"task@_global_={task}",
            f"distraction={distraction}",
            f"seed={job['seed']}",
            f"num_train_frames={job['steps'] * 2}",   # action_repeat=2
            "use_tb=false",
            "save_video=false",
            "save_train_video=false",
            "save_snapshot=true",
            "eval_every_frames=50000",                # = 25K agent steps, as tdmpc2
            f"hydra.run.dir={work_dir(job)}",
        ]
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
    ] + job.get("extra", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--runner", default="tdmpc2", choices=["tdmpc2", "drqv2", "e2e0", "e2e1"],
                    help="drqv2 job task field is '<domain>_<task>__<distraction>'")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--slots-per-gpu", type=int, default=3)
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--stagger", type=float, default=25.0,
                    help="seconds between launches (torch.compile is CPU-heavy at startup)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra", default="",
                    help="extra hydra overrides appended to every tdmpc2/e2e0 "
                         "command, space-separated (e.g. "
                         "'hippoact_include_proprio=false')")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = parse_jobs(args.jobs)
    for job in jobs:
        job["runner"] = args.runner
        job["extra"] = args.extra.split()
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
                release(job)
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
                if not claim(job):
                    print(f"[queue] SKIP {job['task']} seed={job['seed']} "
                          f"— claimed by another queue", flush=True)
                    continue
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
