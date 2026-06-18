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
    path.write_text("data:\n  image_size: 32\n", encoding="utf-8")

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
