# SDAM: Static-Dynamic Associative Memory

This repository implements a research framework for Static-Dynamic Associative Memory (SDAM) visual decision representation learning.

## What Is Implemented

### Core SDAM modules

- `StaticEncoder`: encodes slow-changing visual context from a short observation window.
- `DynamicEncoder`: encodes adjacent-frame visual differences.
- `AssociativeMemory`: fuses dynamic latents with static context using a GRU-based associative memory.
- `SDAMEncoder`: combines static, dynamic, and associative latents into a compact decision memory.
- `PositionVelocityHead`: predicts target position and velocity from SDAM memory.

### Synthetic experiment

- Synthetic moving-object video dataset with static clutter and motion-defined labels.
- Target and distractors share appearance, so the task cannot be solved by color alone.
- Training and evaluation scripts:
  - `scripts/train_synthetic.py`
  - `scripts/eval_synthetic.py`
- Config:
  - `configs/synthetic/sdam.yaml`

### Atari + Stable-Baselines3 integration

- Optional Atari dependencies via the `atari` extra.
- Atari config loader and default PPO config:
  - `configs/atari/sdam_ppo.yaml`
- SB3-compatible SDAM feature extractor:
  - `sdam.policies.SDAMAtariFeaturesExtractor`
- Atari observation conversion:
  - `[B, T, H, W] -> [B, T, 1, H, W]`
  - integer Atari frames are scaled by `255.0`
- Atari PPO experiment builder:
  - lazy SB3 imports
  - `make_atari_env`
  - `VecFrameStack`
  - `PPO("CnnPolicy", ..., policy_kwargs=SDAM)`
- Training script:
  - `scripts/train_atari_sdam.py`

Default tests do not require Stable-Baselines3, ALE, or Atari ROMs.

## Local Test Commands

Create and activate an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the full default suite:

```bash
python -m pytest -v
```

Run compile checks:

```bash
python -m compileall src scripts tests
```

Run synthetic smoke tests:

```bash
python scripts/eval_synthetic.py --config configs/synthetic/sdam.yaml
python scripts/train_synthetic.py --config configs/synthetic/sdam.yaml --steps 1
```

Run Atari tests that do not need SB3/ROMs:

```bash
python -m pytest tests/test_atari_config.py tests/test_sb3_atari_policy.py tests/test_atari_experiment.py -v
```

## Server Atari Setup

Install Atari dependencies on the server:

```bash
pip install -e ".[dev,atari]"
```

If your Gymnasium/ALE installation needs ROMs, install/import them according to your server policy. The project does not download ROMs automatically.

Confirm the training CLI works:

```bash
python scripts/train_atari_sdam.py --help
```

Start a short Atari smoke run:

```bash
python scripts/train_atari_sdam.py \
  --config configs/atari/sdam_ppo.yaml \
  --timesteps 1000 \
  --save-path runs/atari/sdam_ppo_smoke
```

Run a simple PPO NatureCNN vs SDAM feature-extractor comparison:

```bash
python scripts/compare_atari.py \
  --config configs/atari/sdam_ppo.yaml \
  --timesteps 100000 \
  --eval-episodes 10 \
  --output-dir runs/atari/compare_smoke
```

The comparison script writes:

- `runs/atari/compare_smoke/comparison.csv`
- `runs/atari/compare_smoke/comparison.md`
- `runs/atari/compare_smoke/naturecnn.zip`
- `runs/atari/compare_smoke/sdam.zip`

Run the full main-branch-style Alien pipeline. This trains the VAE, trains
`ENV_MODEL_V2`, then trains the vector-state RL agent:

```bash
python scripts/run_main_atari_pipeline.py \
  --config configs/atari/alien_sdam_ppo.yaml \
  --output-dir runs/alien/full_pipeline \
  --vae-episodes 2000 \
  --vae-train-steps 300 \
  --env-episodes 2000 \
  --env-train-steps 300 \
  --rl-algo dqn \
  --timesteps 5000000 \
  --eval-episodes 20 \
  --device cuda \
  --collect-log-interval 10 \
  --train-log-interval 100
```

This matches the stronger route from `main`: the pretrained VAE/background
model and transition model are frozen, each Atari frame is converted to a
160-dimensional vector observation, and Stable-Baselines3 DQN trains an
`MlpPolicy` on top of that vector state.

The output directory contains:

- `runs/alien/full_pipeline/vae_frames.pt`
- `runs/alien/full_pipeline/vae_Alien.pth`
- `runs/alien/full_pipeline/env_model_dataset.pt`
- `runs/alien/full_pipeline/env_Alien.pth`
- `runs/alien/full_pipeline/rl/comparison.csv`
- `runs/alien/full_pipeline/rl/main_vector_dqn.zip`

Run the matched NatureCNN DQN baseline with the same Atari preprocessing and
RL budget:

```bash
python scripts/train_atari_dqn_baseline.py \
  --config configs/atari/alien_sdam_ppo.yaml \
  --timesteps 5000000 \
  --eval-episodes 20 \
  --device cuda \
  --output-dir runs/alien/naturecnn_dqn_5m
```

For debugging individual stages, you can still run the environment model
pretraining or vector DQN training separately:

```bash
python scripts/pretrain_main_env_model.py \
  --config configs/atari/alien_sdam_ppo.yaml \
  --vae-path runs/alien/full_pipeline/vae_Alien.pth \
  --save-path runs/alien/full_pipeline/env_Alien.pth \
  --episodes 2000 \
  --train-steps 300 \
  --device cuda

python scripts/train_main_vector_dqn.py \
  --config configs/atari/alien_sdam_ppo.yaml \
  --vae-path runs/alien/full_pipeline/vae_Alien.pth \
  --env-model-path runs/alien/full_pipeline/env_Alien.pth \
  --timesteps 5000000 \
  --eval-episodes 20 \
  --device cuda \
  --output-dir runs/alien/main_vector_dqn_5m
```

Run the default configured Atari job:

```bash
python scripts/train_atari_sdam.py --config configs/atari/sdam_ppo.yaml
```

## Atari Config

Default config:

```yaml
env:
  env_id: PongNoFrameskip-v4
  n_envs: 1
  n_stack: 4
  seed: 0
  terminal_on_life_loss: false
model:
  static_dim: 64
  dynamic_dim: 64
  assoc_dim: 128
  hidden_channels: 32
  features_dim: 256
ppo:
  learning_rate: 0.00025
  n_steps: 128
  batch_size: 64
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.1
training:
  total_timesteps: 10000
  save_path: runs/atari/sdam_ppo
```

For longer server experiments, copy `configs/atari/sdam_ppo.yaml`, increase `training.total_timesteps`, and optionally increase `env.n_envs`.

## Notes

- The repository keeps SB3 and Atari packages optional so synthetic experiments stay lightweight.
- The default test suite is designed to pass without `stable_baselines3` installed.
- If PyTorch prints a NumPy warning in a minimal environment, install NumPy on the server:

```bash
pip install numpy
```
