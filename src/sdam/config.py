from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
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
class AtariPretrainingConfig:
    collect_steps: int = 50000
    train_steps: int = 1000
    batch_size: int = 32
    learning_rate: float = 0.0003
    reconstruction_weight: float = 1.0
    prediction_weight: float = 1.0
    dataset_path: str = "runs/atari/random_sequences.pt"


@dataclass(frozen=True)
class AtariAlternatingConfig:
    interval: int = 1
    updates: int = 1
    batch_size: int = 32
    learning_rate: float = 0.0001
    pretrained_path: str = ""


@dataclass(frozen=True)
class AtariSDAMConfig:
    env: AtariEnvConfig
    model: AtariModelConfig
    ppo: AtariPPOConfig
    training: AtariTrainingConfig
    pretraining: AtariPretrainingConfig = field(default_factory=AtariPretrainingConfig)
    alternating: AtariAlternatingConfig = field(default_factory=AtariAlternatingConfig)


def load_config(path: str | Path) -> SDAMConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    section_names = ("data", "model", "training")
    for section in section_names:
        if section not in raw:
            raise ValueError(f"missing required section: {section}")

    for section in raw:
        if section not in section_names:
            raise ValueError(f"unknown section: {section}")

    data = _load_section("data", raw["data"], DataConfig)
    model = _load_section("model", raw["model"], ModelConfig)
    training = _load_section("training", raw["training"], TrainingConfig)
    config = SDAMConfig(data=data, model=model, training=training)
    _validate_config(config)
    return config


def load_atari_config(path: str | Path) -> AtariSDAMConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")

    required_sections = ("env", "model", "ppo", "training")
    optional_sections = ("pretraining", "alternating")
    section_names = required_sections + optional_sections
    for section in required_sections:
        if section not in raw:
            raise ValueError(f"missing required section: {section}")

    for section in raw:
        if section not in section_names:
            raise ValueError(f"unknown section: {section}")

    env = _load_section("env", raw["env"], AtariEnvConfig)
    model = _load_section("model", raw["model"], AtariModelConfig)
    ppo = _load_section("ppo", raw["ppo"], AtariPPOConfig)
    training = _load_section("training", raw["training"], AtariTrainingConfig)
    pretraining = (
        _load_section("pretraining", raw["pretraining"], AtariPretrainingConfig)
        if "pretraining" in raw
        else AtariPretrainingConfig()
    )
    alternating = (
        _load_section("alternating", raw["alternating"], AtariAlternatingConfig)
        if "alternating" in raw
        else AtariAlternatingConfig()
    )
    config = AtariSDAMConfig(
        env=env,
        model=model,
        ppo=ppo,
        training=training,
        pretraining=pretraining,
        alternating=alternating,
    )
    _validate_atari_config(config)
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
        if expected_type is str and (type(value) is not str):
            raise ValueError(f"{field_path} must be a string")
        if expected_type is bool and (type(value) is not bool):
            raise ValueError(f"{field_path} must be a bool")
        if expected_type is int and (type(value) is not int):
            raise ValueError(f"{field_path} must be an int")
        if expected_type is float and (
            type(value) not in (int, float)
        ):
            raise ValueError(f"{field_path} must be a number")
        if expected_type in (int, float) and not math.isfinite(value):
            raise ValueError(f"{field_path} must be finite")

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


def _validate_atari_config(config: AtariSDAMConfig) -> None:
    if not config.env.env_id.strip():
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
    if not (0 < config.ppo.gamma <= 1):
        raise ValueError("ppo.gamma must satisfy 0 < value <= 1")
    if not (0 < config.ppo.gae_lambda <= 1):
        raise ValueError("ppo.gae_lambda must satisfy 0 < value <= 1")
    if config.ppo.clip_range <= 0:
        raise ValueError("ppo.clip_range must be positive")
    if config.training.total_timesteps <= 0:
        raise ValueError("training.total_timesteps must be positive")
    if not config.training.save_path.strip():
        raise ValueError("training.save_path must be non-empty")
    if config.pretraining.collect_steps <= 0:
        raise ValueError("pretraining.collect_steps must be positive")
    if config.pretraining.train_steps <= 0:
        raise ValueError("pretraining.train_steps must be positive")
    if config.pretraining.batch_size <= 0:
        raise ValueError("pretraining.batch_size must be positive")
    if config.pretraining.learning_rate <= 0:
        raise ValueError("pretraining.learning_rate must be positive")
    if config.pretraining.reconstruction_weight < 0:
        raise ValueError("pretraining.reconstruction_weight must be non-negative")
    if config.pretraining.prediction_weight < 0:
        raise ValueError("pretraining.prediction_weight must be non-negative")
    if not config.pretraining.dataset_path.strip():
        raise ValueError("pretraining.dataset_path must be non-empty")
    if config.alternating.interval <= 0:
        raise ValueError("alternating.interval must be positive")
    if config.alternating.updates <= 0:
        raise ValueError("alternating.updates must be positive")
    if config.alternating.batch_size <= 0:
        raise ValueError("alternating.batch_size must be positive")
    if config.alternating.learning_rate <= 0:
        raise ValueError("alternating.learning_rate must be positive")
