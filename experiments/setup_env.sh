#!/usr/bin/env bash
# W1.2 环境安装 — A5000 服务器 (4x RTX A5000, sm_86, driver 535 / CUDA 12.2)
#
# 版本选择依据：
#   - 主体沿用 tdmpc2 官方 docker/environment.yaml 的 pin（公平对比的根基）
#   - dm-control 1.0.16 + mujoco 3.1.2：mujoco 在 3.2 才把 tex_rgb 改名 tex_data，
#     停在 3.1.2 则 distracting_control 无需打补丁（PHASE2_PLAN 坑 3 不触发）。
#     若实测不通再按 PHASE2_PLAN §2 升级 dm_control 并打补丁。
#   - torch 官方 cu126 wheel（driver 535 支持 CUDA 12.x 前向兼容）
set -euo pipefail

ENV_NAME=hippoact
CONDA=/usr1/home/s125mdg56_03/miniconda3/bin/conda

$CONDA create -y -n $ENV_NAME python=3.11
PY=/usr1/home/s125mdg56_03/miniconda3/envs/$ENV_NAME/bin/python

$PY -m pip install --upgrade pip
$PY -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
$PY -m pip install \
    "numpy==1.24.4" \
    dm-control==1.0.16 \
    mujoco==3.1.2 \
    glfw==2.7.0 \
    gymnasium==0.29.1 \
    imageio==2.34.1 imageio-ffmpeg==0.4.9 \
    h5py==3.11.0 \
    hydra-core==1.3.2 hydra-submitit-launcher==1.2.0 submitit==1.5.1 omegaconf==2.3.0 \
    moviepy==1.0.3 \
    tensordict==0.8.3 torchrl==0.8.1 \
    kornia==0.7.2 \
    termcolor==2.4.0 tqdm==4.66.4 pandas==2.0.3 \
    wandb==0.17.4

# DCS：distracting_control 依赖 cv2 但未声明（PHASE2_PLAN 坑 2）
$PY -m pip install distracting-control opencv-python-headless

echo "=== versions ==="
$PY - <<'EOF'
import importlib.metadata as md
for p in ["torch","numpy","dm_control","mujoco","distracting_control",
          "gymnasium","tensordict","torchrl","hydra-core"]:
    try:
        print(f"  {p:22s} {md.version(p)}")
    except Exception:
        print(f"  {p:22s} MISSING")
import torch
print("  cuda available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
EOF
