from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import get_type_hints

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

    data = _load_section("data", raw["data"], DataConfig)
    model = _load_section("model", raw["model"], ModelConfig)
    training = _load_section("training", raw["training"], TrainingConfig)
    config = SDAMConfig(data=data, model=model, training=training)
    _validate_config(config)
    return config


def _load_section(section: str, payload: object, config_type):
    if not isinstance(payload, dict):
        raise ValueError(f"section {section} must be a mapping")

    config_fields = fields(config_type)
    field_names = {field.name for field in config_fields}
    for key in payload:
        if key not in field_names:
            raise ValueError(f"unknown key: {section}.{key}")

    for field in config_fields:
        if field.name not in payload:
            raise ValueError(f"missing required key: {section}.{field.name}")

    type_hints = get_type_hints(config_type)
    for field in config_fields:
        value = payload[field.name]
        expected_type = type_hints[field.name]
        field_path = f"{section}.{field.name}"
        if expected_type is int and (type(value) is not int):
            raise ValueError(f"{field_path} must be an int")
        if expected_type is float and (
            type(value) not in (int, float)
        ):
            raise ValueError(f"{field_path} must be a number")

    return config_type(**payload)


def _validate_config(config: SDAMConfig) -> None:
    if config.data.sequence_length < 2:
        raise ValueError("data.sequence_length must be at least 2")
    if config.data.image_size <= 0:
        raise ValueError("data.image_size must be positive")
    if config.data.channels <= 0:
        raise ValueError("data.channels must be positive")
    if config.data.dataset_size <= 0:
        raise ValueError("data.dataset_size must be positive")
    if config.data.object_size <= 0:
        raise ValueError("data.object_size must be positive")
    if config.data.clutter_count < 0:
        raise ValueError("data.clutter_count must be non-negative")
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
