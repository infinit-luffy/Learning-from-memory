from pathlib import Path
import subprocess
import sys

import torch

from sdam.config import load_minerl_prediction_config
from sdam.data.minerl_sequence import MineRLSequenceDataset
from sdam.experiments.minerl_prediction import build_minerl_prediction_components, train_one_step
from sdam.models import SDAM3DScenePredictor, sdam_3d_prediction_loss


def _write_shard(path: Path, samples: int = 3, sequence_length: int = 4, action_dim: int = 6) -> None:
    torch.save(
        {
            "obs": torch.rand(samples, sequence_length, 3, 64, 64),
            "actions": torch.rand(samples, sequence_length - 1, action_dim),
        },
        path,
    )


def test_minerl_prediction_config_loads_default_file():
    config = load_minerl_prediction_config("configs/minerl/navigate_sdam_prediction.yaml")

    assert config.data.sequence_length == 4
    assert config.data.channels == 3
    assert config.data.image_size == 64
    assert config.data.action_dim == 6
    assert config.model.static_dim == 32
    assert config.model.dynamic_dim == 32
    assert config.model.assoc_dim == 48
    assert config.training.change_loss_weight > 0


def test_minerl_sequence_dataset_loads_preprocessed_pt_shards(tmp_path):
    _write_shard(tmp_path / "shard_000.pt", samples=2)
    _write_shard(tmp_path / "shard_001.pt", samples=1)

    dataset = MineRLSequenceDataset(tmp_path, sequence_length=4, image_size=64, action_dim=6)

    assert len(dataset) == 3
    sample = dataset[0]
    assert set(sample.keys()) == {"obs", "actions", "next_obs", "change_mask"}
    assert sample["obs"].shape == (3, 3, 64, 64)
    assert sample["actions"].shape == (3, 6)
    assert sample["next_obs"].shape == (3, 64, 64)
    assert sample["change_mask"].shape == (1, 64, 64)
    assert sample["change_mask"].dtype == torch.float32


def test_sdam_3d_predictor_forward_shapes():
    model = SDAM3DScenePredictor(
        channels=3,
        image_size=64,
        action_dim=6,
        static_dim=16,
        dynamic_dim=12,
        assoc_dim=20,
        hidden_channels=8,
    )

    outputs = model(
        obs=torch.rand(2, 3, 3, 64, 64),
        actions=torch.rand(2, 3, 6),
    )

    assert outputs["static"].shape == (2, 16)
    assert outputs["dynamic_seq"].shape == (2, 3, 12)
    assert outputs["z_assoc"].shape == (2, 20)
    assert outputs["pred_next_latent"].shape == (2, 12)
    assert outputs["target_next_latent"].shape == (2, 12)
    assert outputs["pred_next_frame"].shape == (2, 3, 64, 64)
    assert outputs["pred_change_mask"].shape == (2, 1, 64, 64)


def test_sdam_3d_prediction_loss_backpropagates():
    model = SDAM3DScenePredictor(
        channels=3,
        image_size=64,
        action_dim=6,
        static_dim=16,
        dynamic_dim=12,
        assoc_dim=20,
        hidden_channels=8,
    )
    batch = {
        "obs": torch.rand(2, 3, 3, 64, 64),
        "actions": torch.rand(2, 3, 6),
        "next_obs": torch.rand(2, 3, 64, 64),
        "change_mask": torch.rand(2, 1, 64, 64).round(),
    }

    outputs = model(batch["obs"], batch["actions"])
    loss, metrics = sdam_3d_prediction_loss(outputs, batch)
    loss.backward()

    assert loss.ndim == 0
    assert metrics["total_loss"] >= 0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_minerl_prediction_train_one_step_uses_synthetic_shard(tmp_path):
    _write_shard(tmp_path / "shard_000.pt", samples=4)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  dataset_path: "{tmp_path}"
  sequence_length: 4
  image_size: 64
  channels: 3
  action_dim: 6
  change_threshold: 0.05
model:
  static_dim: 16
  dynamic_dim: 12
  assoc_dim: 20
  hidden_channels: 8
training:
  batch_size: 2
  learning_rate: 0.001
  train_steps: 1
  frame_loss_weight: 1.0
  latent_loss_weight: 1.0
  change_loss_weight: 0.25
  recon_loss_weight: 0.1
  device: "cpu"
""",
        encoding="utf-8",
    )
    components = build_minerl_prediction_components(load_minerl_prediction_config(config_path))

    metrics = train_one_step(components)

    assert metrics["total_loss"] >= 0
    assert metrics["frame_loss"] >= 0
    assert metrics["latent_loss"] >= 0
    assert metrics["change_loss"] >= 0


def test_train_minerl_sdam_prediction_script_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/train_minerl_sdam_prediction.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--config" in result.stdout
    assert "--train-steps" in result.stdout
    assert "MineRL SDAM-3D scene prediction" in result.stdout
