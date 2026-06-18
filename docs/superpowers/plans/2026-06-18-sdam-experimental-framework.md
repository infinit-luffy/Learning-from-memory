# SDAM Experimental Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first PyTorch implementation of Static-Dynamic Associative Memory with synthetic moving-object data, model tests, config loading, and lightweight train/eval entry points.

**Architecture:** The project will use a `src/sdam` package with focused modules for config, synthetic data, static/dynamic encoding, associative memory, SDAM composition, losses, policy adapters, and synthetic experiments. Tests drive each behavior from the outside: first data/config contracts, then tensor-shape model contracts, then an end-to-end synthetic prediction smoke path.

**Tech Stack:** Python 3.11+, PyTorch, PyYAML, pytest, setuptools editable install.

---

## File Structure

- Create `pyproject.toml`: project metadata, package discovery, runtime/test dependencies.
- Create `configs/synthetic/sdam.yaml`: default synthetic experiment configuration.
- Create `src/sdam/__init__.py`: package exports.
- Create `src/sdam/config.py`: dataclass configs plus YAML loading and validation.
- Create `src/sdam/data/synthetic_video.py`: synthetic moving-object dataset.
- Create `src/sdam/models/static_encoder.py`: static background encoder.
- Create `src/sdam/models/dynamic_encoder.py`: frame-difference dynamic encoder.
- Create `src/sdam/models/associative_memory.py`: GRU associative memory.
- Create `src/sdam/models/sdam_encoder.py`: combined SDAM wrapper and shape validation.
- Create `src/sdam/models/prediction_heads.py`: position/velocity prediction head.
- Create `src/sdam/losses/predictive.py`: supervised prediction loss.
- Create `src/sdam/losses/flow_matching.py`: future-extension interface that raises `NotImplementedError`.
- Create `src/sdam/policies/encoder_adapter.py`: policy-facing encoder adapter.
- Create `src/sdam/experiments/synthetic.py`: config-to-model/data/loss wiring, train/eval helpers.
- Create `scripts/train_synthetic.py`: thin training CLI.
- Create `scripts/eval_synthetic.py`: thin evaluation CLI.
- Create `tests/test_imports.py`: package smoke tests.
- Create `tests/test_experiment_config.py`: config validation tests.
- Create `tests/test_synthetic_data.py`: data generation tests.
- Create `tests/test_models.py`: model interface and shape validation tests.
- Create `tests/test_experiment_smoke.py`: one-batch synthetic experiment smoke tests.

## Environment Notes

The current global Python environment does not have `torch`, `yaml`, or `pytest` installed. Implementation should create and use a project-local virtual environment:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e ".[dev]"
```

If dependency installation fails due to network restrictions, request escalation for:

```bash
./.venv/bin/python -m pip install -e ".[dev]"
```

---

### Task 1: Project Packaging And Import Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/sdam/__init__.py`
- Create: `src/sdam/data/__init__.py`
- Create: `src/sdam/models/__init__.py`
- Create: `src/sdam/losses/__init__.py`
- Create: `src/sdam/policies/__init__.py`
- Create: `src/sdam/experiments/__init__.py`
- Create: `tests/test_imports.py`

- [ ] **Step 1: Create package import test**

Create `tests/test_imports.py`:

```python
def test_sdam_package_imports():
    import sdam

    assert sdam.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
./.venv/bin/python -m pytest tests/test_imports.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'sdam'` if the package skeleton has not been created yet. If `.venv` does not exist, create it and install dependencies first using the commands in Environment Notes.

- [ ] **Step 3: Add package metadata and empty package skeleton**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sdam"
version = "0.1.0"
description = "Static-Dynamic Associative Memory for visual decision representation learning."
readme = "docs/superpowers/specs/2026-06-18-sdam-experimental-framework-design.md"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.2",
  "PyYAML>=6.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

Create `src/sdam/__init__.py`:

```python
"""Static-Dynamic Associative Memory research framework."""

__version__ = "0.1.0"
```

Create each package marker file with this exact content:

```python
"""Package namespace."""
```

- [ ] **Step 4: Install editable package and verify test passes**

Run:

```bash
./.venv/bin/python -m pip install -e ".[dev]"
./.venv/bin/python -m pytest tests/test_imports.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/sdam tests/test_imports.py
git commit -m "chore: add python package skeleton"
```

---

### Task 2: Config Loading And Validation

**Files:**
- Create: `configs/synthetic/sdam.yaml`
- Create: `src/sdam/config.py`
- Modify: `src/sdam/__init__.py`
- Create: `tests/test_experiment_config.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/test_experiment_config.py`:

```python
from pathlib import Path

import pytest

from sdam.config import load_config


def test_default_synthetic_config_loads():
    config = load_config(Path("configs/synthetic/sdam.yaml"))

    assert config.data.image_size == 32
    assert config.data.sequence_length == 5
    assert config.model.static_dim == 16
    assert config.model.dynamic_dim == 12
    assert config.model.assoc_dim == 20
    assert config.training.batch_size == 4


def test_missing_required_config_section_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("data:\\n  image_size: 32\\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required section: model"):
        load_config(path)


def test_sequence_length_must_allow_frame_differences(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
data:
  image_size: 32
  channels: 3
  sequence_length: 1
  dataset_size: 8
  object_size: 4
  clutter_count: 1
  min_speed: 1.0
  max_speed: 2.0
model:
  static_dim: 16
  dynamic_dim: 12
  assoc_dim: 20
  hidden_channels: 8
  q_dim: 0
  action_dim: 0
training:
  batch_size: 4
  learning_rate: 0.001
  train_steps: 2
  velocity_loss_weight: 0.25
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="data.sequence_length must be at least 2"):
        load_config(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_experiment_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'sdam.config'` or missing config file.

- [ ] **Step 3: Add default config and loader**

Create `configs/synthetic/sdam.yaml`:

```yaml
data:
  image_size: 32
  channels: 3
  sequence_length: 5
  dataset_size: 64
  object_size: 4
  clutter_count: 2
  min_speed: 1.0
  max_speed: 3.0
model:
  static_dim: 16
  dynamic_dim: 12
  assoc_dim: 20
  hidden_channels: 8
  q_dim: 0
  action_dim: 0
training:
  batch_size: 4
  learning_rate: 0.001
  train_steps: 3
  velocity_loss_weight: 0.25
```

Create `src/sdam/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    image_size: int
    channels: int
    sequence_length: int
    dataset_size: int
    object_size: int
    clutter_count: int
    min_speed: float
    max_speed: float


@dataclass(frozen=True)
class ModelConfig:
    static_dim: int
    dynamic_dim: int
    assoc_dim: int
    hidden_channels: int
    q_dim: int
    action_dim: int


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    train_steps: int
    velocity_loss_weight: float


@dataclass(frozen=True)
class SDAMConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str | Path) -> SDAMConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    for section in ("data", "model", "training"):
        if section not in raw:
            raise ValueError(f"missing required section: {section}")

    data = DataConfig(**raw["data"])
    model = ModelConfig(**raw["model"])
    training = TrainingConfig(**raw["training"])
    config = SDAMConfig(data=data, model=model, training=training)
    _validate_config(config)
    return config


def _validate_config(config: SDAMConfig) -> None:
    if config.data.sequence_length < 2:
        raise ValueError("data.sequence_length must be at least 2")
    if config.data.image_size <= 0:
        raise ValueError("data.image_size must be positive")
    if config.data.channels <= 0:
        raise ValueError("data.channels must be positive")
    if config.data.object_size <= 0:
        raise ValueError("data.object_size must be positive")
    if config.data.object_size >= config.data.image_size:
        raise ValueError("data.object_size must be smaller than data.image_size")
    if config.data.min_speed <= 0 or config.data.max_speed < config.data.min_speed:
        raise ValueError("data speed range must satisfy 0 < min_speed <= max_speed")
    if config.model.static_dim <= 0:
        raise ValueError("model.static_dim must be positive")
    if config.model.dynamic_dim <= 0:
        raise ValueError("model.dynamic_dim must be positive")
    if config.model.assoc_dim <= 0:
        raise ValueError("model.assoc_dim must be positive")
    if config.model.hidden_channels <= 0:
        raise ValueError("model.hidden_channels must be positive")
    if config.model.q_dim < 0 or config.model.action_dim < 0:
        raise ValueError("model q_dim and action_dim must be non-negative")
    if config.training.batch_size <= 0:
        raise ValueError("training.batch_size must be positive")
    if config.training.learning_rate <= 0:
        raise ValueError("training.learning_rate must be positive")
    if config.training.train_steps <= 0:
        raise ValueError("training.train_steps must be positive")
    if config.training.velocity_loss_weight < 0:
        raise ValueError("training.velocity_loss_weight must be non-negative")
```

Update `src/sdam/__init__.py`:

```python
"""Static-Dynamic Associative Memory research framework."""

from sdam.config import SDAMConfig, load_config

__version__ = "0.1.0"

__all__ = ["SDAMConfig", "load_config", "__version__"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
./.venv/bin/python -m pytest tests/test_experiment_config.py tests/test_imports.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/synthetic/sdam.yaml src/sdam/config.py src/sdam/__init__.py tests/test_experiment_config.py
git commit -m "feat: add synthetic experiment config loading"
```

---

### Task 3: Synthetic Moving-Object Dataset

**Files:**
- Create: `src/sdam/data/synthetic_video.py`
- Modify: `src/sdam/data/__init__.py`
- Create: `tests/test_synthetic_data.py`

- [ ] **Step 1: Write failing dataset tests**

Create `tests/test_synthetic_data.py`:

```python
import torch

from sdam.data.synthetic_video import SyntheticVideoConfig, SyntheticVideoDataset


def make_dataset(seed: int = 7) -> SyntheticVideoDataset:
    config = SyntheticVideoConfig(
        image_size=32,
        channels=3,
        sequence_length=5,
        dataset_size=8,
        object_size=4,
        clutter_count=2,
        min_speed=1.0,
        max_speed=2.0,
    )
    return SyntheticVideoDataset(config=config, seed=seed)


def test_synthetic_video_sample_shapes_and_dtype():
    sample = make_dataset()[0]

    assert sample["obs"].shape == (5, 3, 32, 32)
    assert sample["obs"].dtype == torch.float32
    assert sample["dynamic_mask"].shape == (5, 1, 32, 32)
    assert sample["background"].shape == (3, 32, 32)
    assert sample["target_position"].shape == (2,)
    assert sample["target_velocity"].shape == (2,)


def test_target_moves_and_mask_is_non_empty():
    sample = make_dataset()[0]

    centers = sample["target_positions"]
    assert centers.shape == (5, 2)
    assert not torch.allclose(centers[0], centers[-1])
    assert sample["dynamic_mask"].sum() > 0


def test_dataset_is_deterministic_by_index_and_seed():
    first = make_dataset(seed=11)[3]
    second = make_dataset(seed=11)[3]

    assert torch.allclose(first["obs"], second["obs"])
    assert torch.allclose(first["target_positions"], second["target_positions"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_synthetic_data.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'sdam.data.synthetic_video'`.

- [ ] **Step 3: Implement synthetic dataset**

Create `src/sdam/data/synthetic_video.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticVideoConfig:
    image_size: int
    channels: int
    sequence_length: int
    dataset_size: int
    object_size: int
    clutter_count: int
    min_speed: float
    max_speed: float


class SyntheticVideoDataset(Dataset):
    def __init__(self, config: SyntheticVideoConfig, seed: int = 0) -> None:
        if config.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.config = config
        self.seed = seed

    def __len__(self) -> int:
        return self.config.dataset_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        cfg = self.config
        background = self._make_background(generator)
        obs = background.unsqueeze(0).repeat(cfg.sequence_length, 1, 1, 1)
        mask = torch.zeros(cfg.sequence_length, 1, cfg.image_size, cfg.image_size)

        positions = self._target_positions(generator)
        color = torch.full((cfg.channels, 1, 1), 0.95)
        self._draw_static_clutter(obs, generator)

        for t, pos in enumerate(positions):
            x = int(pos[0].item())
            y = int(pos[1].item())
            obs[t, :, y : y + cfg.object_size, x : x + cfg.object_size] = color
            mask[t, :, y : y + cfg.object_size, x : x + cfg.object_size] = 1.0

        velocity = positions[-1] - positions[-2]
        return {
            "obs": obs.clamp(0.0, 1.0).float(),
            "target_positions": positions.float(),
            "target_position": positions[-1].float(),
            "target_velocity": velocity.float(),
            "dynamic_mask": mask.float(),
            "background": background.float(),
        }

    def _make_background(self, generator: torch.Generator) -> torch.Tensor:
        cfg = self.config
        low_res = torch.rand(cfg.channels, 4, 4, generator=generator) * 0.35
        return torch.nn.functional.interpolate(
            low_res.unsqueeze(0),
            size=(cfg.image_size, cfg.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _target_positions(self, generator: torch.Generator) -> torch.Tensor:
        cfg = self.config
        limit = cfg.image_size - cfg.object_size - 1
        pos = torch.rand(2, generator=generator) * limit
        direction = torch.randn(2, generator=generator)
        direction = direction / direction.norm().clamp_min(1e-6)
        speed = cfg.min_speed + torch.rand(1, generator=generator).item() * (cfg.max_speed - cfg.min_speed)
        velocity = direction * speed
        positions = []
        for _ in range(cfg.sequence_length):
            positions.append(pos.round().clamp(0, limit))
            next_pos = pos + velocity
            for axis in range(2):
                if next_pos[axis] < 0 or next_pos[axis] > limit:
                    velocity[axis] = -velocity[axis]
            pos = (pos + velocity).clamp(0, limit)
        return torch.stack(positions)

    def _draw_static_clutter(self, obs: torch.Tensor, generator: torch.Generator) -> None:
        cfg = self.config
        limit = cfg.image_size - cfg.object_size - 1
        for _ in range(cfg.clutter_count):
            x = int(torch.randint(0, limit + 1, (1,), generator=generator).item())
            y = int(torch.randint(0, limit + 1, (1,), generator=generator).item())
            color = torch.full((cfg.channels, 1, 1), 0.8)
            obs[:, :, y : y + cfg.object_size, x : x + cfg.object_size] = color
```

Update `src/sdam/data/__init__.py`:

```python
"""Synthetic data utilities."""

from sdam.data.synthetic_video import SyntheticVideoConfig, SyntheticVideoDataset

__all__ = ["SyntheticVideoConfig", "SyntheticVideoDataset"]
```

- [ ] **Step 4: Run dataset tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_synthetic_data.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sdam/data tests/test_synthetic_data.py
git commit -m "feat: add synthetic moving object dataset"
```

---

### Task 4: Static And Dynamic Encoders

**Files:**
- Create: `src/sdam/models/static_encoder.py`
- Create: `src/sdam/models/dynamic_encoder.py`
- Modify: `src/sdam/models/__init__.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing encoder tests**

Create `tests/test_models.py`:

```python
import pytest
import torch

from sdam.models import DynamicEncoder, StaticEncoder


def make_obs(batch: int = 2, time: int = 5) -> torch.Tensor:
    return torch.rand(batch, time, 3, 32, 32)


def test_static_encoder_returns_static_latent_shape():
    encoder = StaticEncoder(in_channels=3, hidden_channels=8, static_dim=16)

    result = encoder(make_obs())

    assert result.shape == (2, 16)


def test_dynamic_encoder_returns_sequence_of_dynamic_latents():
    encoder = DynamicEncoder(in_channels=3, hidden_channels=8, dynamic_dim=12)

    result = encoder(make_obs())

    assert result.shape == (2, 4, 12)


def test_visual_encoders_reject_invalid_obs_rank():
    static_encoder = StaticEncoder(in_channels=3, hidden_channels=8, static_dim=16)
    dynamic_encoder = DynamicEncoder(in_channels=3, hidden_channels=8, dynamic_dim=12)

    with pytest.raises(ValueError, match="obs must have shape"):
        static_encoder(torch.rand(2, 3, 32, 32))

    with pytest.raises(ValueError, match="obs must have shape"):
        dynamic_encoder(torch.rand(2, 3, 32, 32))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_models.py -v
```

Expected: FAIL with missing `StaticEncoder` or `DynamicEncoder`.

- [ ] **Step 3: Implement encoders**

Create `src/sdam/models/static_encoder.py`:

```python
from __future__ import annotations

import torch
from torch import nn


class StaticEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, static_dim: int) -> None:
        super().__init__()
        self.static_dim = static_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels, static_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 5:
            raise ValueError("obs must have shape [B, T, C, H, W]")
        if obs.shape[1] < 2:
            raise ValueError("obs time dimension must be at least 2")
        background_estimate = obs.mean(dim=1)
        return self.net(background_estimate)
```

Create `src/sdam/models/dynamic_encoder.py`:

```python
from __future__ import annotations

import torch
from torch import nn


class DynamicEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dynamic_dim: int) -> None:
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels, dynamic_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 5:
            raise ValueError("obs must have shape [B, T, C, H, W]")
        if obs.shape[1] < 2:
            raise ValueError("obs time dimension must be at least 2")
        diffs = (obs[:, 1:] - obs[:, :-1]).abs()
        batch, steps, channels, height, width = diffs.shape
        encoded = self.net(diffs.reshape(batch * steps, channels, height, width))
        return encoded.reshape(batch, steps, self.dynamic_dim)
```

Update `src/sdam/models/__init__.py`:

```python
"""Model components for Static-Dynamic Associative Memory."""

from sdam.models.dynamic_encoder import DynamicEncoder
from sdam.models.static_encoder import StaticEncoder

__all__ = ["DynamicEncoder", "StaticEncoder"]
```

- [ ] **Step 4: Run encoder tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_models.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sdam/models tests/test_models.py
git commit -m "feat: add static and dynamic encoders"
```

---

### Task 5: Associative Memory

**Files:**
- Create: `src/sdam/models/associative_memory.py`
- Modify: `src/sdam/models/__init__.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Add failing associative memory tests**

Append to `tests/test_models.py`:

```python
from sdam.models import AssociativeMemory


def test_associative_memory_returns_context_latent_without_optional_inputs():
    memory = AssociativeMemory(dynamic_dim=12, static_dim=16, assoc_dim=20)

    c = memory(z_seq=torch.rand(2, 4, 12), b=torch.rand(2, 16))

    assert c.shape == (2, 20)


def test_associative_memory_accepts_q_and_actions():
    memory = AssociativeMemory(dynamic_dim=12, static_dim=16, assoc_dim=20, q_dim=3, action_dim=2)

    c = memory(
        z_seq=torch.rand(2, 4, 12),
        b=torch.rand(2, 16),
        q=torch.rand(2, 3),
        actions=torch.rand(2, 4, 2),
    )

    assert c.shape == (2, 20)


def test_associative_memory_rejects_wrong_dynamic_dim():
    memory = AssociativeMemory(dynamic_dim=12, static_dim=16, assoc_dim=20)

    with pytest.raises(ValueError, match="z_seq last dimension must be 12"):
        memory(z_seq=torch.rand(2, 4, 11), b=torch.rand(2, 16))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_models.py -v
```

Expected: FAIL with missing `AssociativeMemory`.

- [ ] **Step 3: Implement associative memory**

Create `src/sdam/models/associative_memory.py`:

```python
from __future__ import annotations

import torch
from torch import nn


class AssociativeMemory(nn.Module):
    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        assoc_dim: int,
        q_dim: int = 0,
        action_dim: int = 0,
    ) -> None:
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.assoc_dim = assoc_dim
        self.q_dim = q_dim
        self.action_dim = action_dim
        self.gru = nn.GRU(input_size=dynamic_dim, hidden_size=assoc_dim, batch_first=True)
        fusion_dim = assoc_dim + static_dim + q_dim + action_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, assoc_dim),
            nn.ReLU(),
            nn.Linear(assoc_dim, assoc_dim),
        )

    def forward(
        self,
        z_seq: torch.Tensor,
        b: torch.Tensor,
        q: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_seq.ndim != 3:
            raise ValueError("z_seq must have shape [B, K, dynamic_dim]")
        if z_seq.shape[-1] != self.dynamic_dim:
            raise ValueError(f"z_seq last dimension must be {self.dynamic_dim}")
        if b.ndim != 2 or b.shape[-1] != self.static_dim:
            raise ValueError(f"b must have shape [B, {self.static_dim}]")
        if b.shape[0] != z_seq.shape[0]:
            raise ValueError("b batch dimension must match z_seq")

        _, hidden = self.gru(z_seq)
        parts = [hidden[-1], b]

        if self.q_dim:
            if q is None or q.shape != (z_seq.shape[0], self.q_dim):
                raise ValueError(f"q must have shape [B, {self.q_dim}]")
            parts.append(q)
        elif q is not None:
            raise ValueError("q was provided but q_dim is 0")

        if self.action_dim:
            if actions is None or actions.shape != (z_seq.shape[0], z_seq.shape[1], self.action_dim):
                raise ValueError(f"actions must have shape [B, K, {self.action_dim}]")
            parts.append(actions.mean(dim=1))
        elif actions is not None:
            raise ValueError("actions were provided but action_dim is 0")

        return self.fusion(torch.cat(parts, dim=-1))
```

Update `src/sdam/models/__init__.py`:

```python
"""Model components for Static-Dynamic Associative Memory."""

from sdam.models.associative_memory import AssociativeMemory
from sdam.models.dynamic_encoder import DynamicEncoder
from sdam.models.static_encoder import StaticEncoder

__all__ = ["AssociativeMemory", "DynamicEncoder", "StaticEncoder"]
```

- [ ] **Step 4: Run model tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_models.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sdam/models tests/test_models.py
git commit -m "feat: add associative temporal memory"
```

---

### Task 6: SDAM Encoder And Prediction Head

**Files:**
- Create: `src/sdam/models/sdam_encoder.py`
- Create: `src/sdam/models/prediction_heads.py`
- Modify: `src/sdam/models/__init__.py`
- Modify: `tests/test_models.py`

- [ ] **Step 1: Add failing SDAM wrapper tests**

Append to `tests/test_models.py`:

```python
from sdam.models import PositionVelocityHead, SDAMEncoder


def test_sdam_encoder_returns_structured_memory_dict():
    encoder = SDAMEncoder(
        in_channels=3,
        sequence_length=5,
        hidden_channels=8,
        static_dim=16,
        dynamic_dim=12,
        assoc_dim=20,
    )

    outputs = encoder(make_obs())

    assert set(outputs.keys()) == {"b", "z_seq", "c", "memory", "aux"}
    assert outputs["b"].shape == (2, 16)
    assert outputs["z_seq"].shape == (2, 4, 12)
    assert outputs["c"].shape == (2, 20)
    assert outputs["memory"].shape == (2, encoder.memory_dim)


def test_sdam_encoder_accepts_time_aligned_q_and_actions():
    encoder = SDAMEncoder(
        in_channels=3,
        sequence_length=5,
        hidden_channels=8,
        static_dim=16,
        dynamic_dim=12,
        assoc_dim=20,
        q_dim=3,
        action_dim=2,
    )

    outputs = encoder(make_obs(), q=torch.rand(2, 5, 3), actions=torch.rand(2, 4, 2))

    assert outputs["memory"].shape == (2, encoder.memory_dim)


def test_position_velocity_head_predicts_four_values():
    encoder = SDAMEncoder(in_channels=3, sequence_length=5, hidden_channels=8, static_dim=16, dynamic_dim=12, assoc_dim=20)
    head = PositionVelocityHead(memory_dim=encoder.memory_dim, hidden_dim=32)

    prediction = head(encoder(make_obs())["memory"])

    assert prediction["position"].shape == (2, 2)
    assert prediction["velocity"].shape == (2, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_models.py -v
```

Expected: FAIL with missing `SDAMEncoder` or `PositionVelocityHead`.

- [ ] **Step 3: Implement SDAM wrapper and head**

Create `src/sdam/models/sdam_encoder.py`:

```python
from __future__ import annotations

import torch
from torch import nn

from sdam.models.associative_memory import AssociativeMemory
from sdam.models.dynamic_encoder import DynamicEncoder
from sdam.models.static_encoder import StaticEncoder


class SDAMEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        sequence_length: int,
        hidden_channels: int,
        static_dim: int,
        dynamic_dim: int,
        assoc_dim: int,
        q_dim: int = 0,
        action_dim: int = 0,
    ) -> None:
        super().__init__()
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.static_encoder = StaticEncoder(in_channels, hidden_channels, static_dim)
        self.dynamic_encoder = DynamicEncoder(in_channels, hidden_channels, dynamic_dim)
        self.associative_memory = AssociativeMemory(dynamic_dim, static_dim, assoc_dim, q_dim, action_dim)
        self.sequence_length = sequence_length
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.assoc_dim = assoc_dim
        self.q_dim = q_dim
        self.action_dim = action_dim
        self.memory_dim = static_dim + ((sequence_length - 1) * dynamic_dim) + assoc_dim + q_dim

    def forward(
        self,
        obs: torch.Tensor,
        q: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if obs.ndim != 5:
            raise ValueError("obs must have shape [B, T, C, H, W]")
        if obs.shape[1] < 2:
            raise ValueError("obs time dimension must be at least 2")
        if obs.shape[1] != self.sequence_length:
            raise ValueError(f"obs time dimension must be {self.sequence_length}")
        batch, time = obs.shape[:2]

        q_t = None
        if self.q_dim:
            if q is None or q.shape != (batch, time, self.q_dim):
                raise ValueError(f"q must have shape [B, T, {self.q_dim}]")
            q_t = q[:, -1]
        elif q is not None:
            raise ValueError("q was provided but q_dim is 0")

        if self.action_dim:
            if actions is None or actions.shape != (batch, time - 1, self.action_dim):
                raise ValueError(f"actions must have shape [B, T - 1, {self.action_dim}]")
        elif actions is not None:
            raise ValueError("actions were provided but action_dim is 0")

        b = self.static_encoder(obs)
        z_seq = self.dynamic_encoder(obs)
        c = self.associative_memory(z_seq=z_seq, b=b, q=q_t, actions=actions)
        memory_parts = [b, z_seq.flatten(start_dim=1), c]
        if q_t is not None:
            memory_parts.append(q_t)
        memory = torch.cat(memory_parts, dim=-1)
        return {"b": b, "z_seq": z_seq, "c": c, "memory": memory, "aux": {}}
```

Create `src/sdam/models/prediction_heads.py`:

```python
from __future__ import annotations

import torch
from torch import nn


class PositionVelocityHead(nn.Module):
    def __init__(self, memory_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, memory: torch.Tensor) -> dict[str, torch.Tensor]:
        if memory.ndim != 2:
            raise ValueError("memory must have shape [B, memory_dim]")
        prediction = self.net(memory)
        return {"position": prediction[:, :2], "velocity": prediction[:, 2:]}
```

Update `src/sdam/models/__init__.py`:

```python
"""Model components for Static-Dynamic Associative Memory."""

from sdam.models.associative_memory import AssociativeMemory
from sdam.models.dynamic_encoder import DynamicEncoder
from sdam.models.prediction_heads import PositionVelocityHead
from sdam.models.sdam_encoder import SDAMEncoder
from sdam.models.static_encoder import StaticEncoder

__all__ = [
    "AssociativeMemory",
    "DynamicEncoder",
    "PositionVelocityHead",
    "SDAMEncoder",
    "StaticEncoder",
]
```

- [ ] **Step 4: Run model tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_models.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sdam/models tests/test_models.py
git commit -m "feat: compose sdam encoder memory state"
```

---

### Task 7: Losses, Policy Adapter, And Synthetic Experiment Wiring

**Files:**
- Create: `src/sdam/losses/predictive.py`
- Create: `src/sdam/losses/flow_matching.py`
- Modify: `src/sdam/losses/__init__.py`
- Create: `src/sdam/policies/encoder_adapter.py`
- Modify: `src/sdam/policies/__init__.py`
- Create: `src/sdam/experiments/synthetic.py`
- Modify: `src/sdam/experiments/__init__.py`
- Create: `tests/test_experiment_smoke.py`

- [ ] **Step 1: Write failing experiment smoke tests**

Create `tests/test_experiment_smoke.py`:

```python
from pathlib import Path

import pytest

from sdam.config import load_config
from sdam.experiments.synthetic import build_synthetic_components, evaluate_one_batch, train_one_step
from sdam.losses.flow_matching import latent_flow_matching_loss


def test_build_synthetic_components_and_train_one_step():
    config = load_config(Path("configs/synthetic/sdam.yaml"))
    components = build_synthetic_components(config)

    metrics = train_one_step(components)

    assert metrics["loss"] >= 0.0
    assert metrics["position_loss"] >= 0.0
    assert metrics["velocity_loss"] >= 0.0


def test_evaluate_one_batch_reports_memory_dim_and_losses():
    config = load_config(Path("configs/synthetic/sdam.yaml"))
    components = build_synthetic_components(config)

    metrics = evaluate_one_batch(components)

    assert metrics["memory_dim"] > 0
    assert metrics["position_mse"] >= 0.0
    assert metrics["velocity_mse"] >= 0.0


def test_latent_flow_matching_loss_is_future_extension_interface():
    with pytest.raises(NotImplementedError, match="Latent flow association is not implemented"):
        latent_flow_matching_loss()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_experiment_smoke.py -v
```

Expected: FAIL with missing experiment or loss modules.

- [ ] **Step 3: Implement predictive loss and flow extension interface**

Create `src/sdam/losses/predictive.py`:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def position_velocity_loss(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    velocity_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    position_loss = F.mse_loss(predictions["position"], batch["target_position"])
    velocity_loss = F.mse_loss(predictions["velocity"], batch["target_velocity"])
    loss = position_loss + velocity_weight * velocity_loss
    metrics = {
        "loss": float(loss.detach().item()),
        "position_loss": float(position_loss.detach().item()),
        "velocity_loss": float(velocity_loss.detach().item()),
    }
    return loss, metrics
```

Create `src/sdam/losses/flow_matching.py`:

```python
from __future__ import annotations


def latent_flow_matching_loss(*args: object, **kwargs: object) -> None:
    raise NotImplementedError("Latent flow association is not implemented in the first milestone")
```

Update `src/sdam/losses/__init__.py`:

```python
"""Loss functions for SDAM experiments."""

from sdam.losses.flow_matching import latent_flow_matching_loss
from sdam.losses.predictive import position_velocity_loss

__all__ = ["latent_flow_matching_loss", "position_velocity_loss"]
```

- [ ] **Step 4: Implement policy adapter and synthetic experiment wiring**

Create `src/sdam/policies/encoder_adapter.py`:

```python
from __future__ import annotations

import torch
from torch import nn


class PolicyEncoderAdapter(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    @property
    def memory_dim(self) -> int:
        return int(self.encoder.memory_dim)

    def forward(self, obs: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(obs, **kwargs)
        return outputs["memory"]
```

Update `src/sdam/policies/__init__.py`:

```python
"""Policy-facing adapters for SDAM encoders."""

from sdam.policies.encoder_adapter import PolicyEncoderAdapter

__all__ = ["PolicyEncoderAdapter"]
```

Create `src/sdam/experiments/synthetic.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from sdam.config import SDAMConfig
from sdam.data import SyntheticVideoConfig, SyntheticVideoDataset
from sdam.losses import position_velocity_loss
from sdam.models import PositionVelocityHead, SDAMEncoder


@dataclass
class SyntheticComponents:
    config: SDAMConfig
    loader: DataLoader
    encoder: SDAMEncoder
    head: PositionVelocityHead
    optimizer: torch.optim.Optimizer


def build_synthetic_components(config: SDAMConfig, seed: int = 0) -> SyntheticComponents:
    dataset_config = SyntheticVideoConfig(
        image_size=config.data.image_size,
        channels=config.data.channels,
        sequence_length=config.data.sequence_length,
        dataset_size=config.data.dataset_size,
        object_size=config.data.object_size,
        clutter_count=config.data.clutter_count,
        min_speed=config.data.min_speed,
        max_speed=config.data.max_speed,
    )
    dataset = SyntheticVideoDataset(dataset_config, seed=seed)
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False)
    encoder = SDAMEncoder(
        in_channels=config.data.channels,
        sequence_length=config.data.sequence_length,
        hidden_channels=config.model.hidden_channels,
        static_dim=config.model.static_dim,
        dynamic_dim=config.model.dynamic_dim,
        assoc_dim=config.model.assoc_dim,
        q_dim=config.model.q_dim,
        action_dim=config.model.action_dim,
    )
    head = PositionVelocityHead(memory_dim=encoder.memory_dim)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=config.training.learning_rate,
    )
    return SyntheticComponents(config=config, loader=loader, encoder=encoder, head=head, optimizer=optimizer)


def train_one_step(components: SyntheticComponents) -> dict[str, float]:
    components.encoder.train()
    components.head.train()
    batch = next(iter(components.loader))
    components.optimizer.zero_grad()
    outputs = components.encoder(batch["obs"])
    predictions = components.head(outputs["memory"])
    loss, metrics = position_velocity_loss(
        predictions,
        batch,
        velocity_weight=components.config.training.velocity_loss_weight,
    )
    loss.backward()
    components.optimizer.step()
    return metrics


@torch.no_grad()
def evaluate_one_batch(components: SyntheticComponents) -> dict[str, float]:
    components.encoder.eval()
    components.head.eval()
    batch = next(iter(components.loader))
    outputs = components.encoder(batch["obs"])
    predictions = components.head(outputs["memory"])
    _, metrics = position_velocity_loss(
        predictions,
        batch,
        velocity_weight=components.config.training.velocity_loss_weight,
    )
    return {
        "position_mse": metrics["position_loss"],
        "velocity_mse": metrics["velocity_loss"],
        "memory_dim": float(components.encoder.memory_dim),
    }
```

Update `src/sdam/experiments/__init__.py`:

```python
"""Experiment builders for SDAM."""

from sdam.experiments.synthetic import (
    SyntheticComponents,
    build_synthetic_components,
    evaluate_one_batch,
    train_one_step,
)

__all__ = ["SyntheticComponents", "build_synthetic_components", "evaluate_one_batch", "train_one_step"]
```

- [ ] **Step 5: Run experiment smoke tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_experiment_smoke.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sdam/losses src/sdam/policies src/sdam/experiments tests/test_experiment_smoke.py
git commit -m "feat: wire synthetic prediction experiment"
```

---

### Task 8: Training And Evaluation Scripts

**Files:**
- Create: `scripts/train_synthetic.py`
- Create: `scripts/eval_synthetic.py`
- Modify: `tests/test_experiment_smoke.py`

- [ ] **Step 1: Add failing script-entry tests**

Append to `tests/test_experiment_smoke.py`:

```python
import subprocess
import sys


def test_eval_script_runs_on_default_config():
    result = subprocess.run(
        [sys.executable, "scripts/eval_synthetic.py", "--config", "configs/synthetic/sdam.yaml"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "position_mse" in result.stdout
    assert "memory_dim" in result.stdout


def test_train_script_runs_for_one_step():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_synthetic.py",
            "--config",
            "configs/synthetic/sdam.yaml",
            "--steps",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "step=1" in result.stdout
    assert "loss=" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
./.venv/bin/python -m pytest tests/test_experiment_smoke.py::test_eval_script_runs_on_default_config tests/test_experiment_smoke.py::test_train_script_runs_for_one_step -v
```

Expected: FAIL because `scripts/eval_synthetic.py` and `scripts/train_synthetic.py` do not exist.

- [ ] **Step 3: Implement evaluation script**

Create `scripts/eval_synthetic.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from sdam.config import load_config
from sdam.experiments import build_synthetic_components, evaluate_one_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic/sdam.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    components = build_synthetic_components(config)
    metrics = evaluate_one_batch(components)
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement training script**

Create `scripts/train_synthetic.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from sdam.config import load_config
from sdam.experiments import build_synthetic_components, train_one_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic/sdam.yaml"))
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    components = build_synthetic_components(config)
    steps = args.steps if args.steps is not None else config.training.train_steps
    for step in range(1, steps + 1):
        metrics = train_one_step(components)
        print(
            f"step={step} loss={metrics['loss']:.6f} "
            f"position_loss={metrics['position_loss']:.6f} "
            f"velocity_loss={metrics['velocity_loss']:.6f}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run script tests**

Run:

```bash
./.venv/bin/python -m pytest tests/test_experiment_smoke.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts tests/test_experiment_smoke.py
git commit -m "feat: add synthetic train and eval scripts"
```

---

### Task 9: Full Verification And Documentation Check

**Files:**
- Modify only files needed to fix verification failures found in this task.

- [ ] **Step 1: Run full test suite**

Run:

```bash
./.venv/bin/python -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 2: Run compile check**

Run:

```bash
python3 -m compileall src scripts tests
```

Expected: command completes without syntax errors.

- [ ] **Step 3: Run one manual smoke command**

Run:

```bash
./.venv/bin/python scripts/eval_synthetic.py --config configs/synthetic/sdam.yaml
```

Expected: stdout contains `position_mse=`, `velocity_mse=`, and `memory_dim=`.

- [ ] **Step 4: Confirm scope against spec**

Check:

```bash
rg -n "Atari|robot|grasp|world-model|rollout" src scripts tests configs
```

Expected: no implementation of Atari, robotics, grasping, or long-horizon rollout exists in `src`, `scripts`, `tests`, or `configs`.

- [ ] **Step 5: Commit final verification fixes if any**

If Step 1 through Step 4 required code fixes, run:

```bash
git add src scripts tests configs pyproject.toml
git commit -m "test: verify sdam synthetic framework"
```

If no files changed, do not create an empty commit.

---

## Plan Self-Review

- Spec coverage: This plan covers project packaging, config loading, synthetic data, static encoder, dynamic encoder, associative memory, SDAM wrapper, prediction head, predictive loss, flow future-extension interface, policy adapter, train/eval scripts, and CPU tests.
- Scope control: This plan does not implement Atari, robot grasping, image reconstruction, online RL, or long-horizon world-model rollout.
- Type consistency: `obs` is `[B, T, C, H, W]`, `z_seq` is `[B, T - 1, dynamic_dim]`, `b` is `[B, static_dim]`, `c` is `[B, assoc_dim]`, and `memory` is `[B, static_dim + ((T - 1) * dynamic_dim) + assoc_dim + q_dim]` throughout the tasks.
- Execution risk: Dependency installation needs network access because the current environment lacks `torch`, `PyYAML`, and `pytest`.
