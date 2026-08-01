#!/usr/bin/env bash
# Wait for the retention eval on gpu0/gpu1 to finish, then start the W1.3
# DrQ-v2 matrix on those two cards. Keeps the GPUs busy without having the
# two workloads contend.
#
# gpu2/gpu3 are separately running the cheetah clean s1/s2 re-runs.
set -u
cd /usr1/home/s125mdg56_03/Learning-from-memory
PY=/usr1/home/s125mdg56_03/miniconda3/envs/hippoact/bin/python

echo "[chain] waiting for retention_eval.py to finish ..."
while pgrep -f 'scripts/retention_eval\.py' > /dev/null; do sleep 60; done
echo "[chain] retention done at $(date)"

echo "[chain] launching W1.3 DrQ-v2 on gpu0,gpu1"
exec $PY experiments/scripts/run_queue.py \
    --jobs experiments/scripts/w13_jobs.txt \
    --runner drqv2 --gpus 0,1 --slots-per-gpu 3 --stagger 45
