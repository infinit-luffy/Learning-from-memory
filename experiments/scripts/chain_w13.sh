#!/usr/bin/env bash
# Wait for the retention eval to finish, then start the W1.3 DrQ-v2 matrix on
# **gpu1 only** — gpu0 is reserved for Stage-1 retraining and then W2.1 E2E-0,
# which is the TODO's original per-GPU assignment.
#
# gpu2/gpu3 are running the cheetah clean s1/s2 re-runs; when those finish,
# top up DrQ-v2 there with a second queue against the same job file (the queue
# skips runs that already reached their step target, so it is safe to re-run):
#
#   nohup $PY experiments/scripts/run_queue.py --jobs experiments/scripts/w13_jobs.txt \
#       --runner drqv2 --gpus 2,3 --slots-per-gpu 3 >> experiments/logs/queue_w13b.log 2>&1 &
set -u
cd /usr1/home/s125mdg56_03/Learning-from-memory
PY=/usr1/home/s125mdg56_03/miniconda3/envs/hippoact/bin/python

echo "[chain] waiting for retention_eval.py to finish ..."
while pgrep -f 'scripts/retention_eval\.py' > /dev/null; do sleep 60; done
echo "[chain] retention done at $(date)"

echo "[chain] launching W1.3 DrQ-v2 on gpu1 (gpu0 reserved for Stage-1 -> E2E-0)"
exec $PY experiments/scripts/run_queue.py \
    --jobs experiments/scripts/w13_jobs.txt \
    --runner drqv2 --gpus 1 --slots-per-gpu 3 --stagger 45
