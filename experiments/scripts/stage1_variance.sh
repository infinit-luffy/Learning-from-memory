#!/usr/bin/env bash
# Stage-1 seed-variance sweep (R4 ruling 3b, brought forward to use an idle GPU).
#
# Question: is the retrained encoder's 3.51 / 3.26 enrichment vs W1.1's
# 4.15 / 4.26 a systematic gap, or ordinary run-to-run spread?
#
# Stage-1 was UNSEEDED until b7a2f94 (`train.seed` existed in every config but
# was never read), so the W1.1 run and the retrain were already two independent
# draws — there was never a "same seed should match" baseline. This sweep
# measures the spread directly.
#
# Usage:  stage1_variance.sh <seed> <gpu>
set -euo pipefail

SEED=${1:?usage: stage1_variance.sh <seed> <gpu>}
GPU=${2:?usage: stage1_variance.sh <seed> <gpu>}

cd /usr1/home/s125mdg56_03/Learning-from-memory/hippoact
PY=/usr1/home/s125mdg56_03/miniconda3/envs/hippoact/bin/python
DATA=data/dcs_walker

declare -A EGL=( [0]=2 [1]=3 [2]=0 [3]=1 )     # measured; see egl_probe.py
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=$GPU
export MUJOCO_EGL_DEVICE_ID=${EGL[$GPU]}
export PYTHONPATH=/usr1/home/s125mdg56_03/Learning-from-memory/hippoact
export HIPPOACT_FORCE_MOCK=0

OUT=outputs/stage1_seed$SEED

$PY - <<'PYEOF' || exit 1
from hippoact.encoders.dinov2 import DinoV2Encoder, MockDinoV2Encoder
enc = DinoV2Encoder("dinov2_vits14", 224, 14)
inner = getattr(enc, "_backbone", None)
assert inner is not None and not isinstance(inner, MockDinoV2Encoder), \
    "DINOv2 fell back to MockDinoV2Encoder -- refusing to train"
print(f"[var] DINOv2 backbone OK: {type(inner).__name__}")
PYEOF

echo "[var] seed=$SEED gpu=$GPU -> $OUT   $(date)"
$PY scripts/pretrain_stage1.py \
    --config configs/stage1_dcs.yaml \
    --data-dir "$DATA/train_both" \
    --seed "$SEED" \
    --num-workers 8 2>&1 | sed "s/^/[s$SEED] /"

# pretrain writes to <log.out_dir>/stage1; move it aside so seeds do not clash
mv outputs/stage1 "$OUT"
echo "[var] training done, checkpoints in $OUT   $(date)"

CKPT="$OUT/ckpt_final.pt"
[ -f "$CKPT" ] || { echo "[var] FATAL: no $CKPT"; exit 1; }

export PYTHONPATH=$PYTHONPATH:$PWD/tools/diagnostics:$PWD/tools/dcs
for split in clean easy; do
    echo "--- seed=$SEED eval_$split ---"
    $PY tools/dcs/walker_enrichment.py \
        --config configs/stage1_dcs.yaml --ckpt "$CKPT" \
        --data-dir "$DATA/eval_$split" --label "s${SEED}_${split}"
done
