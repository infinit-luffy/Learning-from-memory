from pathlib import Path

import pytest

from sdam.config import DataConfig, ModelConfig, SDAMConfig, TrainingConfig, load_config


def test_default_synthetic_config_loads():
    config = load_config(Path("configs/synthetic/sdam.yaml"))

    assert config == SDAMConfig(
        data=DataConfig(
            image_size=32,
            channels=3,
            sequence_length=5,
            dataset_size=64,
            object_size=4,
            clutter_count=2,
            min_speed=1.0,
            max_speed=3.0,
        ),
        model=ModelConfig(
            static_dim=16,
            dynamic_dim=12,
            assoc_dim=20,
            hidden_channels=8,
            q_dim=0,
            action_dim=0,
        ),
        training=TrainingConfig(
            batch_size=4,
            learning_rate=0.001,
            train_steps=3,
            velocity_loss_weight=0.25,
        ),
    )


def test_missing_required_config_section_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("data:\n  image_size: 32\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required section: model"):
        load_config(path)


def test_section_must_be_mapping(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
data: []
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

    with pytest.raises(ValueError, match="section data must be a mapping"):
        load_config(path)


def test_missing_required_config_key_raises_with_context(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
data:
  image_size: 32
  sequence_length: 5
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

    with pytest.raises(ValueError, match="missing required key: data.channels"):
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


def test_dataset_size_must_be_positive(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
data:
  image_size: 32
  channels: 3
  sequence_length: 5
  dataset_size: 0
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

    with pytest.raises(ValueError, match="data.dataset_size must be positive"):
        load_config(path)


def test_clutter_count_must_be_non_negative(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
data:
  image_size: 32
  channels: 3
  sequence_length: 5
  dataset_size: 8
  object_size: 4
  clutter_count: -1
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

    with pytest.raises(ValueError, match="data.clutter_count must be non-negative"):
        load_config(path)
