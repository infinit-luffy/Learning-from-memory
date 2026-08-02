#!/usr/bin/env bash
# Top up the W1.3 DrQ-v2 queue onto gpu2/gpu3 as the cheetah re-runs finish.
#
# Safe to point at the same job file as the gpu1 queue: run_queue.py now takes
# an on-disk claim (`<work_dir>/.claim`, pid-checked) before launching a job, so
# two queues can share a list without ever running the same job twice. That
# accident happened once already, when two queues were started over one list.
set -u
cd /usr1/home/s125mdg56_03/Learning-from-memory
PY=/usr1/home/s125mdg56_03/miniconda3/envs/hippoact/bin/python

echo "[topup] waiting for the cheetah re-runs (gpu2/gpu3) to finish ..."
while pgrep -f 'w12_cheetah_rerun' > /dev/null; do sleep 120; done
echo "[topup] re-runs done at $(date)"

echo "[topup] launching a second W1.3 queue on gpu2,gpu3 (claims arbitrate)"
exec $PY experiments/scripts/run_queue.py \
    --jobs experiments/scripts/w13_jobs.txt \
    --runner drqv2 --gpus 2,3 --slots-per-gpu 3 --stagger 45
