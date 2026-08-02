#!/usr/bin/env bash
# Re-collect the Stage-1 DCS dataset on the A5000 box (W1.1 was run on the 5080
# and its checkpoint was never transferred; retraining is cheaper than waiting).
#
# Reproduces the W1.1 collection exactly (TODO_RESULT §W1.1):
#   train: walker-walk clean + easy, 25K frames each, random policy,
#          clip structure + qpos/qvel, 224x224, camera 0
#   eval : clean + easy, 2500 frames each, WITH exact MuJoCo segmentation
#          masks; both use the same seed so the physics is identical and the
#          only difference is the background (controlled contrast).
#
# Eval uses seed 1 while training uses seed 0, so the criterion is not measured
# on the training frames.
set -euo pipefail

cd /usr1/home/s125mdg56_03/Learning-from-memory/hippoact
PY=/usr1/home/s125mdg56_03/miniconda3/envs/hippoact/bin/python
DAVIS=/usr1/home/s125mdg56_03/datasets/davis/DAVIS/JPEGImages/480p
OUT=data/dcs_walker

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=2      # -> physical gpu0 (see experiments/scripts/egl_probe.py)

collect () {   # difficulty out_subdir n_frames seed extra...
    local diff=$1 sub=$2 n=$3 seed=$4; shift 4
    echo "=== collecting $sub: difficulty=$diff n=$n seed=$seed $* ==="
    $PY tools/dcs/collect_frames.py \
        --domain walker --task walk \
        --difficulty "$diff" \
        --davis-path "$DAVIS" \
        --out "$OUT/$sub" \
        --n-frames "$n" --clip-len 25 --size 224 --camera-id 0 \
        --rebuild-every 8 --reset-every 2 \
        --seed "$seed" "$@"
}

collect clean train_clean 25000 0
collect easy  train_easy  25000 0
collect clean eval_clean   2500 1 --save-segmentation
collect easy  eval_easy    2500 1 --save-segmentation

echo
echo "=== sizes ==="
du -sh "$OUT"/*
for d in "$OUT"/*/; do
    echo "  $(basename "$d"): $(ls -d "$d"clip_* 2>/dev/null | wc -l) clips"
done
