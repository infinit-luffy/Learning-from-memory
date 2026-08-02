#!/usr/bin/env bash
# Retrain Stage-1 on DCS (the W1.1 checkpoint lives on the retired 5080 and was
# never transferred). Recipe is CP6's locked configuration, unchanged:
#
#   slow_signal = alpha_connectivity   (CP6: Cohen's d -1.25 -> +1.89)
#   slot_init_mode = shared, slot_iters = 3, slot_dim = 128, lambda_slow = 0.5
#
# Trained on clean + easy jointly, exactly as W1.1.
#
# This run is only usable if it REPRODUCES W1.1's criterion. Run
# `stage1_check.sh` afterwards; the numbers to match are
#   clean  fast-slot walker enrichment 4.15x, Cohen's d +2.262, AUC 0.943
#   easy   fast-slot walker enrichment 4.26x, Cohen's d +1.844, AUC 0.833
# with the controls oracle 9.54 and uniform exactly 1.00.
set -euo pipefail

cd /usr1/home/s125mdg56_03/Learning-from-memory/hippoact
PY=/usr1/home/s125mdg56_03/miniconda3/envs/hippoact/bin/python

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=${STAGE1_GPU:-0}
export HIPPOACT_FORCE_MOCK=0        # never silently fall back to MockDinoV2

RUN=${RUN_NAME:-stage1_dcs_a5000}

$PY scripts/pretrain_stage1.py \
    --config configs/stage1_dcs.yaml \
    --data-dir data/dcs_walker/train_both \
    --run-name "$RUN" \
    --num-workers 8
