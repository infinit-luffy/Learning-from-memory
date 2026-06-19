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

Run a simple NatureCNN vs SDAM comparison:

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
