#!/usr/bin/env bash
# When the three E2E-0 runs finish, immediately run the zero-cost Q2 evaluation
# on their checkpoints (TODO ruling 1): each 500K ckpt evaluated on
# {none, easy, hard} with no further training — the first test of the
# flat-invariance prediction from the R1 dual-ratio metrics.
#
# Pre-registered (TODO ruling 1):
#   HippoAct  background-presence invariance  R_none/R_easy  should be
#             clearly above pixel's 0.65 (walker) / 0.30 (cheetah)
#   HippoAct  escalation retention            R_hard/R_easy  should be
#             above pixel's 0.74 (walker) / 0.49 (cheetah)
set -u
cd /usr1/home/s125mdg56_03/Learning-from-memory
PY=/usr1/home/s125mdg56_03/miniconda3/envs/hippoact/bin/python

echo "[chain] waiting for the E2E-0 queue to finish ..."
while pgrep -f 'run_queue\.py --jobs experiments/scripts/e2e0_jobs' > /dev/null; do
    sleep 120
done
echo "[chain] E2E-0 done at $(date)"

for n in 1 3 5; do
    d=experiments/logs/dcs-easy-walker-walk/1/e2e0_s$n
    if [ ! -f "$d/models/final.pt" ]; then
        echo "[chain] WARN: no final.pt for e2e0_s$n — skipping Q2 for it"
    fi
done

echo "[chain] Q2 zero-shot evaluation of the E2E-0 checkpoints"
exec env HIPPOACT_STRICT_DINO=1 $PY experiments/scripts/retention_eval.py \
    --exp e2e0 --episodes 30 --gpus 0,2,3
