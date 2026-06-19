from pathlib import Path

import pytest

import sdam.config as config_module


def _write_atari_config(
    tmp_path,
    *,
    env_id="PongNoFrameskip-v4",
    terminal_on_life_loss="false",
    ppo_learning_rate="0.00025",
):
    path = tmp_path / "bad.yaml"
    path.write_text(
        f"""
env:
  env_id: {env_id}
  n_envs: 1
  n_stack: 4
  seed: 0
  terminal_on_life_loss: {terminal_on_life_loss}
model:
  static_dim: 64
  dynamic_dim: 64
  assoc_dim: 128
  hidden_channels: 32
  features_dim: 256
ppo:
  learning_rate: {ppo_learning_rate}
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
    return path


def test_default_atari_config_loads():
    config = config_module.load_atari_config(Path("configs/atari/sdam_ppo.yaml"))

    assert config.env.env_id == "PongNoFrameskip-v4"
    assert config.env.n_envs == 1
    assert config.env.n_stack == 4
    assert config.env.seed == 0
    assert config.env.terminal_on_life_loss is False
    assert config.model.static_dim == 64
    assert config.model.dynamic_dim == 64
    assert config.model.assoc_dim == 128
    assert config.model.hidden_channels == 32
    assert config.model.features_dim == 256
    assert config.ppo.learning_rate == 0.00025
    assert config.ppo.n_steps == 128
    assert config.ppo.batch_size == 64
    assert config.ppo.gamma == 0.99
    assert config.ppo.gae_lambda == 0.95
    assert config.ppo.clip_range == 0.1
    assert config.training.total_timesteps == 10000
    assert config.training.save_path == "runs/atari/sdam_ppo"


def test_unknown_root_section_raises(tmp_path):
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
extra: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown section: extra"):
        config_module.load_atari_config(path)


def test_n_stack_must_be_at_least_two(tmp_path):
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
        config_module.load_atari_config(path)


def test_env_id_must_be_string(tmp_path):
    path = _write_atari_config(tmp_path, env_id="123")

    with pytest.raises(ValueError, match=r"env\.env_id must be a string"):
        config_module.load_atari_config(path)


def test_terminal_on_life_loss_must_be_bool(tmp_path):
    path = _write_atari_config(tmp_path, terminal_on_life_loss="1")

    with pytest.raises(ValueError, match=r"env\.terminal_on_life_loss must be a bool"):
        config_module.load_atari_config(path)


def test_ppo_learning_rate_must_be_finite(tmp_path):
    path = _write_atari_config(tmp_path, ppo_learning_rate=".nan")

    with pytest.raises(ValueError, match=r"ppo\.learning_rate must be finite"):
        config_module.load_atari_config(path)
