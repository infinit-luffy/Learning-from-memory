#!/usr/bin/env bash
# Full Stage-1 retrain pipeline on gpu0: wait for collection -> merge -> train
# -> reproduction check against W1.1.
#
# The check is the point. A retrained Stage-1 is only usable for W2.1 E2E-0 if
# it reproduces W1.1's criterion; otherwise every downstream E2E number is
# built on a different encoder than the one the paper describes.
set -u

cd /usr1/home/s125mdg56_03/Learning-from-memory/hippoact
PY=/usr1/home/s125mdg56_03/miniconda3/envs/hippoact/bin/python
DATA=data/dcs_walker

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=2       # -> physical gpu0
# `python scripts/pretrain_stage1.py` puts scripts/ on sys.path, not the
# package root, so `import hippoact` fails. It only appeared to work in
# interactive checks because `python -c` / `python -m` add the cwd.
export PYTHONPATH=/usr1/home/s125mdg56_03/Learning-from-memory/hippoact${PYTHONPATH:+:$PYTHONPATH}
export HIPPOACT_FORCE_MOCK=0        # never silently fall back to MockDinoV2

echo "[chain] waiting for collect_stage1_data.sh ..."
while pgrep -f 'collect_stage1_data\.sh' > /dev/null; do sleep 30; done
echo "[chain] collection finished at $(date)"

# Fail loudly if the DINOv2 download ever falls back to MockDinoV2Encoder:
# training would otherwise run to completion on a random frozen CNN and
# produce numbers that look plausible but mean nothing.
$PY - <<'PYEOF' || exit 1
from hippoact.encoders.dinov2 import DinoV2Encoder, MockDinoV2Encoder
enc = DinoV2Encoder("dinov2_vits14", 224, 14)
inner = getattr(enc, "_backbone", None)
assert inner is not None and not isinstance(inner, MockDinoV2Encoder), \
    "DINOv2 fell back to MockDinoV2Encoder -- refusing to train"
print(f"[chain] DINOv2 backbone OK: {type(inner).__name__}")
PYEOF

# Expected clip counts, so a half-finished collection cannot silently become
# a shorter training set. (The first attempt at this chain waited on the wrong
# process name and fell through mid-collection; this guard caught it.)
check_clips () {   # dir expected
    local n; n=$(ls -d "$DATA/$1"/clip_* 2>/dev/null | wc -l)
    echo "[chain]   $1: $n clips (expected $2)"
    if [ "$n" -ne "$2" ]; then echo "[chain] FATAL: $1 has $n clips, expected $2"; exit 1; fi
}
check_clips train_clean 1000
check_clips train_easy  1000
check_clips eval_clean   100
check_clips eval_easy    100

# --- merge clean + easy into one training root -----------------------------
# W1.1 trained on both jointly (48000 pairs = 2 x 1000 clips x 24 pairs).
# Symlinks with a difficulty prefix, since clip names collide between sets.
echo "[chain] building $DATA/train_both"
rm -rf "$DATA/train_both"
mkdir -p "$DATA/train_both"
for diff in clean easy; do
    for c in "$DATA/train_$diff"/clip_*; do
        ln -s "$(realpath "$c")" "$DATA/train_both/${diff}_$(basename "$c")"
    done
done
echo "[chain]   train_both: $(ls -d "$DATA"/train_both/* | wc -l) clips"

# --- train -----------------------------------------------------------------
echo "[chain] training Stage-1 (CP6 recipe, alpha_connectivity) at $(date)"
$PY scripts/pretrain_stage1.py \
    --config configs/stage1_dcs.yaml \
    --data-dir "$DATA/train_both" \
    --run-name stage1_dcs_a5000 \
    --num-workers 8
rc=$?
echo "[chain] training exited rc=$rc at $(date)"
[ $rc -ne 0 ] && exit $rc

CKPT=$(ls -t outputs/stage1_dcs_a5000/*.pt 2>/dev/null | head -1)
echo "[chain] checkpoint: $CKPT"
[ -z "$CKPT" ] && { echo "[chain] FATAL: no checkpoint written"; exit 1; }

# --- reproduction check ----------------------------------------------------
# W1.1 reference:  clean 4.15x (d +2.262, AUC 0.943) / easy 4.26x (d +1.844,
# AUC 0.833);  controls: oracle 9.54, uniform exactly 1.00.
echo
echo "===================== W1.1 复现判据 ====================="
for split in clean easy; do
    echo "--- eval_$split ---"
    $PY tools/dcs/walker_enrichment.py \
        --config configs/stage1_dcs.yaml \
        --ckpt "$CKPT" \
        --data-dir "$DATA/eval_$split" \
        --label "$split"
done
echo "========================================================="
echo "判据: fast slots 富集 >= 3.0 (W1.1: clean 4.15 / easy 4.26)"
echo "对照: uniform 必须精确等于 1.00 —— 不等于则是尺子坏了, 不是模型坏了"
