# SDAM Atari Stable-Baselines3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Stable-Baselines3 Atari integration where `CnnPolicy` uses SDAM as its image feature extractor.

**Architecture:** Keep SB3 responsible for PPO, Atari vector envs, rollout collection, losses, logging, and saving. Add SDAM-specific Atari config, a feature extractor that maps stacked Atari observations into `SDAMEncoder`, a small experiment builder, and a thin training script.

**Tech Stack:** Python 3.11+, PyTorch, PyYAML, optional Stable-Baselines3/Gymnasium/ALE, pytest.

---

## File Structure

- Modify `pyproject.toml`: add the optional `atari` extra.
- Modify `src/sdam/config.py`: add Atari-specific config dataclasses and `load_atari_config`.
- Create `configs/atari/sdam_ppo.yaml`: default Atari PPO config.
- Create `src/sdam/policies/sb3_atari.py`: SB3-compatible `SDAMAtariFeaturesExtractor` and tensor conversion helpers.
- Modify `src/sdam/policies/__init__.py`: export Atari policy helpers without importing SB3 at package import time.
- Create `src/sdam/experiments/atari.py`: lazy SB3 imports, env/model builders, and train runner.
- Modify `src/sdam/experiments/__init__.py`: export Atari runner names only if safe, or leave explicit imports.
- Create `scripts/train_atari_sdam.py`: CLI entry point.
- Create `tests/test_atari_config.py`: Atari config validation tests.
- Create `tests/test_sb3_atari_policy.py`: tensor conversion and extractor tests using a fake SB3 module.
- Create `tests/test_atari_experiment.py`: policy kwargs, lazy dependency errors, and script runner tests.

## Task 1: Optional Dependency Extra

**Files:**
- Modify: `pyproject.toml`
- Test: none standalone; verified by import tests and config tests later

- [ ] **Step 1: Update optional dependencies**

Add an `atari` extra while preserving the existing `dev` extra:

```toml
[project.optional-dependencies]
dev = [
  "pytest>=8.0",
]
atari = [
  "stable-baselines3[extra]>=2.3",
  "gymnasium[atari,accept-rom-license]>=0.29",
  "ale-py>=0.8",
]
```

- [ ] **Step 2: Run metadata-sensitive tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_imports.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add optional atari dependencies"
```

## Task 2: Atari Config Loader

**Files:**
- Modify: `src/sdam/config.py`
- Create: `configs/atari/sdam_ppo.yaml`
- Test: `tests/test_atari_config.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/test_atari_config.py`:

```python
from pathlib import Path

import pytest

from sdam.config import load_atari_config


def test_default_atari_config_loads():
    config = load_atari_config(Path("configs/atari/sdam_ppo.yaml"))

    assert config.env.env_id == "PongNoFrameskip-v4"
    assert config.env.n_stack == 4
    assert config.model.features_dim == 256
    assert config.ppo.learning_rate == pytest.approx(0.00025)
    assert config.training.total_timesteps == 10000


def test_atari_config_rejects_unknown_section(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
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
extra:
  value: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown section: extra"):
        load_atari_config(path)


def test_atari_config_rejects_invalid_frame_stack(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
env:
  env_id: PongNoFrameskip-v4
  n_envs: 1
  n_stack: 1
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
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="env.n_stack must be at least 2"):
        load_atari_config(path)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_atari_config.py -v
```

Expected: FAIL with import or missing file errors for `load_atari_config` / config.

- [ ] **Step 3: Add config dataclasses and loader**

Modify `src/sdam/config.py` by adding new dataclasses after `SDAMConfig`:

```python
@dataclass(frozen=True)
class AtariEnvConfig:
    env_id: str
    n_envs: int
    n_stack: int
    seed: int
    terminal_on_life_loss: bool


@dataclass(frozen=True)
class AtariModelConfig:
    static_dim: int
    dynamic_dim: int
    assoc_dim: int
    hidden_channels: int
    features_dim: int


@dataclass(frozen=True)
class AtariPPOConfig:
    learning_rate: float
    n_steps: int
    batch_size: int
    gamma: float
    gae_lambda: float
    clip_range: float


@dataclass(frozen=True)
class AtariTrainingConfig:
    total_timesteps: int
    save_path: str


@dataclass(frozen=True)
class AtariSDAMConfig:
    env: AtariEnvConfig
    model: AtariModelConfig
    ppo: AtariPPOConfig
    training: AtariTrainingConfig
```

Add string/bool support to `_load_section`:

```python
        if expected_type is str and type(value) is not str:
            raise ValueError(f"{field_path} must be a string")
        if expected_type is bool and type(value) is not bool:
            raise ValueError(f"{field_path} must be a bool")
```

Add:

```python
def load_atari_config(path: str | Path) -> AtariSDAMConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    section_names = ("env", "model", "ppo", "training")
    for section in section_names:
        if section not in raw:
            raise ValueError(f"missing required section: {section}")
    for section in raw:
        if section not in section_names:
            raise ValueError(f"unknown section: {section}")

    config = AtariSDAMConfig(
        env=_load_section("env", raw["env"], AtariEnvConfig),
        model=_load_section("model", raw["model"], AtariModelConfig),
        ppo=_load_section("ppo", raw["ppo"], AtariPPOConfig),
        training=_load_section("training", raw["training"], AtariTrainingConfig),
    )
    _validate_atari_config(config)
    return config


def _validate_atari_config(config: AtariSDAMConfig) -> None:
    if not config.env.env_id:
        raise ValueError("env.env_id must be non-empty")
    if config.env.n_envs <= 0:
        raise ValueError("env.n_envs must be positive")
    if config.env.n_stack < 2:
        raise ValueError("env.n_stack must be at least 2")
    if config.env.seed < 0:
        raise ValueError("env.seed must be non-negative")
    if config.model.static_dim <= 0:
        raise ValueError("model.static_dim must be positive")
    if config.model.dynamic_dim <= 0:
        raise ValueError("model.dynamic_dim must be positive")
    if config.model.assoc_dim <= 0:
        raise ValueError("model.assoc_dim must be positive")
    if config.model.hidden_channels <= 0:
        raise ValueError("model.hidden_channels must be positive")
    if config.model.features_dim <= 0:
        raise ValueError("model.features_dim must be positive")
    if config.ppo.learning_rate <= 0:
        raise ValueError("ppo.learning_rate must be positive")
    if config.ppo.n_steps <= 0:
        raise ValueError("ppo.n_steps must be positive")
    if config.ppo.batch_size <= 0:
        raise ValueError("ppo.batch_size must be positive")
    if config.ppo.gamma <= 0 or config.ppo.gamma > 1:
        raise ValueError("ppo.gamma must satisfy 0 < gamma <= 1")
    if config.ppo.gae_lambda <= 0 or config.ppo.gae_lambda > 1:
        raise ValueError("ppo.gae_lambda must satisfy 0 < gae_lambda <= 1")
    if config.ppo.clip_range <= 0:
        raise ValueError("ppo.clip_range must be positive")
    if config.training.total_timesteps <= 0:
        raise ValueError("training.total_timesteps must be positive")
    if not config.training.save_path:
        raise ValueError("training.save_path must be non-empty")
```

- [ ] **Step 4: Add default Atari config**

Create `configs/atari/sdam_ppo.yaml`:

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

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```bash
./.venv/bin/python -m pytest tests/test_atari_config.py tests/test_experiment_config.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sdam/config.py configs/atari/sdam_ppo.yaml tests/test_atari_config.py
git commit -m "feat: add atari experiment config"
```

## Task 3: Atari Observation Conversion

**Files:**
- Create: `src/sdam/policies/sb3_atari.py`
- Modify: `src/sdam/policies/__init__.py`
- Test: `tests/test_sb3_atari_policy.py`

- [ ] **Step 1: Write failing tensor conversion tests**

Create `tests/test_sb3_atari_policy.py`:

```python
import pytest
import torch

from sdam.policies.sb3_atari import atari_observations_to_sdam


def test_atari_observations_to_sdam_adds_channel_and_normalizes_uint_like_values():
    observations = torch.full((2, 4, 84, 84), 255.0)

    converted = atari_observations_to_sdam(observations, sequence_length=4)

    assert converted.shape == (2, 4, 1, 84, 84)
    assert converted.dtype == torch.float32
    assert torch.allclose(converted, torch.ones_like(converted))


def test_atari_observations_to_sdam_preserves_already_normalized_values():
    observations = torch.full((2, 4, 84, 84), 0.5)

    converted = atari_observations_to_sdam(observations, sequence_length=4)

    assert torch.allclose(converted, torch.full((2, 4, 1, 84, 84), 0.5))


def test_atari_observations_to_sdam_rejects_invalid_rank():
    with pytest.raises(ValueError, match="observations must have shape"):
        atari_observations_to_sdam(torch.rand(2, 4, 1, 84, 84), sequence_length=4)


def test_atari_observations_to_sdam_rejects_wrong_frame_stack():
    with pytest.raises(ValueError, match="observations frame stack must be 4"):
        atari_observations_to_sdam(torch.rand(2, 3, 84, 84), sequence_length=4)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sb3_atari_policy.py -v
```

Expected: FAIL because `sdam.policies.sb3_atari` does not exist.

- [ ] **Step 3: Implement conversion helper**

Create `src/sdam/policies/sb3_atari.py`:

```python
from __future__ import annotations

import torch


_SB3_EXTRA_MESSAGE = 'Stable-Baselines3 is required for Atari experiments. Install with: pip install -e ".[atari]"'


def atari_observations_to_sdam(observations: torch.Tensor, sequence_length: int) -> torch.Tensor:
    if observations.ndim != 4:
        raise ValueError("observations must have shape [B, T, H, W]")
    if observations.shape[1] != sequence_length:
        raise ValueError(f"observations frame stack must be {sequence_length}")

    converted = observations.float()
    if converted.numel() > 0 and converted.max() > 1.0:
        converted = converted / 255.0
    return converted.unsqueeze(2)
```

Modify `src/sdam/policies/__init__.py`:

```python
from sdam.policies.encoder_adapter import PolicyEncoderAdapter
from sdam.policies.sb3_atari import atari_observations_to_sdam

__all__ = ["PolicyEncoderAdapter", "atari_observations_to_sdam"]
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sb3_atari_policy.py tests/test_imports.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sdam/policies/sb3_atari.py src/sdam/policies/__init__.py tests/test_sb3_atari_policy.py
git commit -m "feat: add atari observation conversion"
```

## Task 4: SB3-Compatible SDAM Feature Extractor

**Files:**
- Modify: `src/sdam/policies/sb3_atari.py`
- Test: `tests/test_sb3_atari_policy.py`

- [ ] **Step 1: Add fake SB3 extractor tests**

Append to `tests/test_sb3_atari_policy.py`:

```python
import sys
import types


class FakeBox:
    def __init__(self, shape):
        self.shape = shape


def install_fake_sb3(monkeypatch):
    import torch.nn as nn

    stable_baselines3 = types.ModuleType("stable_baselines3")
    common = types.ModuleType("stable_baselines3.common")
    torch_layers = types.ModuleType("stable_baselines3.common.torch_layers")

    class BaseFeaturesExtractor(nn.Module):
        def __init__(self, observation_space, features_dim):
            super().__init__()
            self.observation_space = observation_space
            self._features_dim = features_dim

        @property
        def features_dim(self):
            return self._features_dim

    torch_layers.BaseFeaturesExtractor = BaseFeaturesExtractor
    common.torch_layers = torch_layers
    stable_baselines3.common = common
    monkeypatch.setitem(sys.modules, "stable_baselines3", stable_baselines3)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", common)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.torch_layers", torch_layers)


def test_sdam_atari_features_extractor_returns_features(monkeypatch):
    install_fake_sb3(monkeypatch)
    from sdam.policies.sb3_atari import SDAMAtariFeaturesExtractor

    extractor = SDAMAtariFeaturesExtractor(
        FakeBox((4, 84, 84)),
        static_dim=8,
        dynamic_dim=8,
        assoc_dim=16,
        hidden_channels=4,
        features_dim=32,
        sequence_length=4,
    )

    features = extractor(torch.rand(2, 4, 84, 84))

    assert features.shape == (2, 32)
    assert extractor.features_dim == 32


def test_sdam_atari_features_extractor_requires_matching_observation_space(monkeypatch):
    install_fake_sb3(monkeypatch)
    from sdam.policies.sb3_atari import SDAMAtariFeaturesExtractor

    with pytest.raises(ValueError, match="observation_space shape must be"):
        SDAMAtariFeaturesExtractor(
            FakeBox((3, 84, 84)),
            static_dim=8,
            dynamic_dim=8,
            assoc_dim=16,
            hidden_channels=4,
            features_dim=32,
            sequence_length=4,
        )
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sb3_atari_policy.py::test_sdam_atari_features_extractor_returns_features tests/test_sb3_atari_policy.py::test_sdam_atari_features_extractor_requires_matching_observation_space -v
```

Expected: FAIL because `SDAMAtariFeaturesExtractor` is not defined.

- [ ] **Step 3: Implement lazy SB3 base class and extractor**

Modify `src/sdam/policies/sb3_atari.py`:

```python
from torch import nn

from sdam.models import SDAMEncoder


def _load_base_features_extractor():
    try:
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    except ImportError as exc:
        raise ImportError(_SB3_EXTRA_MESSAGE) from exc
    return BaseFeaturesExtractor


class SDAMAtariFeaturesExtractor(_load_base_features_extractor()):
    def __init__(
        self,
        observation_space,
        static_dim: int,
        dynamic_dim: int,
        assoc_dim: int,
        hidden_channels: int,
        features_dim: int,
        sequence_length: int,
    ) -> None:
        if tuple(observation_space.shape) != (sequence_length, 84, 84):
            raise ValueError(f"observation_space shape must be ({sequence_length}, 84, 84)")
        super().__init__(observation_space, features_dim)
        self.sequence_length = sequence_length
        self.encoder = SDAMEncoder(
            in_channels=1,
            sequence_length=sequence_length,
            hidden_channels=hidden_channels,
            static_dim=static_dim,
            dynamic_dim=dynamic_dim,
            assoc_dim=assoc_dim,
        )
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.memory_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        sdam_obs = atari_observations_to_sdam(observations, self.sequence_length)
        memory = self.encoder(sdam_obs)["memory"]
        return self.projection(memory)
```

If importing this module without SB3 now fails, revise by defining a fallback base class:

```python
try:
    _BaseFeaturesExtractor = _load_base_features_extractor()
except ImportError:
    class _BaseFeaturesExtractor(nn.Module):
        def __init__(self, *args, **kwargs):
            raise ImportError(_SB3_EXTRA_MESSAGE)
```

Then inherit from `_BaseFeaturesExtractor`. Default tests must still be able to import `atari_observations_to_sdam` without SB3 installed.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
./.venv/bin/python -m pytest tests/test_sb3_atari_policy.py tests/test_imports.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sdam/policies/sb3_atari.py tests/test_sb3_atari_policy.py
git commit -m "feat: add sb3 sdam atari extractor"
```

## Task 5: Atari Experiment Builder

**Files:**
- Create: `src/sdam/experiments/atari.py`
- Test: `tests/test_atari_experiment.py`

- [ ] **Step 1: Write failing experiment tests**

Create `tests/test_atari_experiment.py`:

```python
from dataclasses import replace

import pytest

from sdam.config import load_atari_config
from sdam.experiments.atari import (
    build_atari_env,
    build_sdam_atari_model,
    build_sdam_atari_policy_kwargs,
)
from sdam.policies.sb3_atari import SDAMAtariFeaturesExtractor


def test_build_sdam_atari_policy_kwargs_uses_sdam_extractor():
    config = load_atari_config("configs/atari/sdam_ppo.yaml")

    kwargs = build_sdam_atari_policy_kwargs(config)

    assert kwargs["features_extractor_class"] is SDAMAtariFeaturesExtractor
    assert kwargs["features_extractor_kwargs"]["sequence_length"] == 4
    assert kwargs["features_extractor_kwargs"]["features_dim"] == 256


def test_build_atari_env_raises_clear_error_without_sb3():
    config = load_atari_config("configs/atari/sdam_ppo.yaml")

    with pytest.raises(ImportError, match="Stable-Baselines3 is required"):
        build_atari_env(config)


def test_build_sdam_atari_model_uses_ppo_constructor(monkeypatch):
    config = load_atari_config("configs/atari/sdam_ppo.yaml")
    calls = {}

    class FakePPO:
        def __init__(self, policy, env, policy_kwargs, learning_rate, n_steps, batch_size, gamma, gae_lambda, clip_range, verbose):
            calls["policy"] = policy
            calls["env"] = env
            calls["policy_kwargs"] = policy_kwargs
            calls["learning_rate"] = learning_rate
            calls["n_steps"] = n_steps
            calls["batch_size"] = batch_size
            calls["gamma"] = gamma
            calls["gae_lambda"] = gae_lambda
            calls["clip_range"] = clip_range
            calls["verbose"] = verbose

    monkeypatch.setattr("sdam.experiments.atari._load_ppo", lambda: FakePPO)

    model = build_sdam_atari_model(config, env="fake-env", verbose=2)

    assert isinstance(model, FakePPO)
    assert calls["policy"] == "CnnPolicy"
    assert calls["env"] == "fake-env"
    assert calls["policy_kwargs"]["features_extractor_class"] is SDAMAtariFeaturesExtractor
    assert calls["learning_rate"] == pytest.approx(config.ppo.learning_rate)
    assert calls["verbose"] == 2
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_atari_experiment.py -v
```

Expected: FAIL because `sdam.experiments.atari` does not exist.

- [ ] **Step 3: Implement experiment builder**

Create `src/sdam/experiments/atari.py`:

```python
from __future__ import annotations

from sdam.config import AtariSDAMConfig
from sdam.policies.sb3_atari import SDAMAtariFeaturesExtractor


_SB3_EXTRA_MESSAGE = 'Stable-Baselines3 is required for Atari experiments. Install with: pip install -e ".[atari]"'


def _load_ppo():
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(_SB3_EXTRA_MESSAGE) from exc
    return PPO


def _load_atari_env_tools():
    try:
        from stable_baselines3.common.env_util import make_atari_env
        from stable_baselines3.common.vec_env import VecFrameStack
    except ImportError as exc:
        raise ImportError(_SB3_EXTRA_MESSAGE) from exc
    return make_atari_env, VecFrameStack


def build_sdam_atari_policy_kwargs(config: AtariSDAMConfig) -> dict:
    return {
        "features_extractor_class": SDAMAtariFeaturesExtractor,
        "features_extractor_kwargs": {
            "static_dim": config.model.static_dim,
            "dynamic_dim": config.model.dynamic_dim,
            "assoc_dim": config.model.assoc_dim,
            "hidden_channels": config.model.hidden_channels,
            "features_dim": config.model.features_dim,
            "sequence_length": config.env.n_stack,
        },
    }


def build_atari_env(config: AtariSDAMConfig):
    make_atari_env, VecFrameStack = _load_atari_env_tools()
    env = make_atari_env(
        config.env.env_id,
        n_envs=config.env.n_envs,
        seed=config.env.seed,
        wrapper_kwargs={"terminal_on_life_loss": config.env.terminal_on_life_loss},
    )
    return VecFrameStack(env, n_stack=config.env.n_stack)


def build_sdam_atari_model(config: AtariSDAMConfig, env, verbose: int = 1):
    PPO = _load_ppo()
    return PPO(
        "CnnPolicy",
        env,
        policy_kwargs=build_sdam_atari_policy_kwargs(config),
        learning_rate=config.ppo.learning_rate,
        n_steps=config.ppo.n_steps,
        batch_size=config.ppo.batch_size,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_range=config.ppo.clip_range,
        verbose=verbose,
    )
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
./.venv/bin/python -m pytest tests/test_atari_experiment.py tests/test_imports.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sdam/experiments/atari.py tests/test_atari_experiment.py
git commit -m "feat: add atari sb3 experiment builder"
```

## Task 6: Atari Training Script

**Files:**
- Create: `scripts/train_atari_sdam.py`
- Modify: `src/sdam/experiments/atari.py`
- Test: `tests/test_atari_experiment.py`

- [ ] **Step 1: Write failing runner and script tests**

Append to `tests/test_atari_experiment.py`:

```python
import subprocess
import sys


def test_train_sdam_atari_closes_env_and_saves_model(monkeypatch, tmp_path):
    config = load_atari_config("configs/atari/sdam_ppo.yaml")
    calls = {}

    class FakeEnv:
        def close(self):
            calls["closed"] = True

    class FakeModel:
        def learn(self, total_timesteps):
            calls["timesteps"] = total_timesteps
            return self

        def save(self, save_path):
            calls["save_path"] = str(save_path)

    monkeypatch.setattr("sdam.experiments.atari.build_atari_env", lambda cfg: FakeEnv())
    monkeypatch.setattr("sdam.experiments.atari.build_sdam_atari_model", lambda cfg, env, verbose=1: FakeModel())

    from sdam.experiments.atari import train_sdam_atari

    train_sdam_atari(config, total_timesteps=12, save_path=tmp_path / "model")

    assert calls["timesteps"] == 12
    assert calls["save_path"] == str(tmp_path / "model")
    assert calls["closed"] is True


def test_train_atari_script_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/train_atari_sdam.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--config" in result.stdout
    assert "--timesteps" in result.stdout
    assert "--save-path" in result.stdout
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_atari_experiment.py::test_train_sdam_atari_closes_env_and_saves_model tests/test_atari_experiment.py::test_train_atari_script_help_runs -v
```

Expected: FAIL because `train_sdam_atari` and script do not exist.

- [ ] **Step 3: Implement train runner**

Append to `src/sdam/experiments/atari.py`:

```python
from pathlib import Path


def train_sdam_atari(
    config: AtariSDAMConfig,
    total_timesteps: int | None = None,
    save_path: str | Path | None = None,
    verbose: int = 1,
):
    env = build_atari_env(config)
    try:
        model = build_sdam_atari_model(config, env, verbose=verbose)
        steps = total_timesteps if total_timesteps is not None else config.training.total_timesteps
        output_path = Path(save_path if save_path is not None else config.training.save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.learn(total_timesteps=steps)
        model.save(output_path)
        return model
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()
```

- [ ] **Step 4: Implement training script**

Create `scripts/train_atari_sdam.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from sdam.config import load_atari_config
from sdam.experiments.atari import train_sdam_atari


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SDAM Atari PPO with Stable-Baselines3.")
    parser.add_argument("--config", default="configs/atari/sdam_ppo.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_atari_config(Path(args.config))
    train_sdam_atari(
        config,
        total_timesteps=args.timesteps,
        save_path=args.save_path,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify GREEN**

Run:

```bash
./.venv/bin/python -m pytest tests/test_atari_experiment.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sdam/experiments/atari.py scripts/train_atari_sdam.py tests/test_atari_experiment.py
git commit -m "feat: add atari training script"
```

## Task 7: Final Verification And Review

**Files:**
- All files changed in Tasks 1-6

- [ ] **Step 1: Run full tests**

Run:

```bash
./.venv/bin/python -m pytest -v
```

Expected: PASS. A PyTorch warning about missing NumPy may appear if NumPy is not installed; do not treat that warning as a failure unless tests fail.

- [ ] **Step 2: Run compile check**

Run:

```bash
./.venv/bin/python -m compileall src scripts tests
```

Expected: exit code 0.

- [ ] **Step 3: Run help smoke test**

Run:

```bash
./.venv/bin/python scripts/train_atari_sdam.py --help
```

Expected: exit code 0 and help text includes `--config`, `--timesteps`, and `--save-path`.

- [ ] **Step 4: Confirm default tests do not require SB3**

Run:

```bash
./.venv/bin/python -c "import importlib.util; print(importlib.util.find_spec('stable_baselines3'))"
./.venv/bin/python -m pytest tests/test_atari_config.py tests/test_sb3_atari_policy.py tests/test_atari_experiment.py -v
```

Expected: if SB3 is not installed, the first command prints `None`, and the targeted tests still pass.

- [ ] **Step 5: Request code review**

Use superpowers:requesting-code-review with:

```bash
BASE_SHA=599d613
HEAD_SHA=$(git rev-parse HEAD)
```

Review request fields:

```text
WHAT_WAS_IMPLEMENTED: Stable-Baselines3 Atari integration for SDAM.
PLAN_OR_REQUIREMENTS: docs/superpowers/specs/2026-06-19-sdam-atari-sb3-design.md and docs/superpowers/plans/2026-06-19-sdam-atari-sb3.md.
BASE_SHA: use the BASE_SHA command output above.
HEAD_SHA: use the HEAD_SHA command output above.
DESCRIPTION: Added optional Atari dependencies, Atari config, SDAM SB3 feature extractor, Atari PPO builders, and train_atari_sdam.py.
```

Fix any Critical or Important issues before proceeding.

- [ ] **Step 6: Final commit if review fixes were needed**

If review required fixes:

```bash
git status --short
git add pyproject.toml src/sdam/config.py src/sdam/policies/__init__.py src/sdam/policies/sb3_atari.py src/sdam/experiments/atari.py scripts/train_atari_sdam.py configs/atari/sdam_ppo.yaml tests/test_atari_config.py tests/test_sb3_atari_policy.py tests/test_atari_experiment.py
git commit -m "fix: address atari sb3 review feedback"
```

- [ ] **Step 7: Finish branch**

Use superpowers:finishing-a-development-branch. Present merge/PR/keep/discard options after fresh verification passes.
