from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
