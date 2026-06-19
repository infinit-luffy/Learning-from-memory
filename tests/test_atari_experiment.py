import builtins
import importlib
import subprocess
import sys
import types
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings(
    "ignore",
    message="Failed to initialize NumPy: No module named 'numpy'.*",
    category=UserWarning,
)

from sdam.config import load_atari_config
from sdam.experiments.atari import (
    build_atari_env,
    build_sdam_atari_model,
    build_sdam_atari_policy_kwargs,
    train_sdam_atari,
)
from sdam.policies.sb3_atari import SDAMAtariFeaturesExtractor


CONFIG_PATH = Path("configs/atari/sdam_ppo.yaml")


def install_fake_atari_env_tools(monkeypatch, make_atari_env, VecFrameStack):
    stable_baselines3 = types.ModuleType("stable_baselines3")
    stable_baselines3.__path__ = []
    common = types.ModuleType("stable_baselines3.common")
    common.__path__ = []
    env_util = types.ModuleType("stable_baselines3.common.env_util")
    vec_env = types.ModuleType("stable_baselines3.common.vec_env")

    env_util.make_atari_env = make_atari_env
    vec_env.VecFrameStack = VecFrameStack
    common.env_util = env_util
    common.vec_env = vec_env
    stable_baselines3.common = common

    monkeypatch.setitem(sys.modules, "stable_baselines3", stable_baselines3)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", common)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.env_util", env_util)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.vec_env", vec_env)


def test_build_sdam_atari_policy_kwargs_uses_sdam_extractor():
    config = load_atari_config(CONFIG_PATH)

    policy_kwargs = build_sdam_atari_policy_kwargs(config)

    assert policy_kwargs["features_extractor_class"] is SDAMAtariFeaturesExtractor
    extractor_kwargs = policy_kwargs["features_extractor_kwargs"]
    assert extractor_kwargs["sequence_length"] == config.env.n_stack
    assert extractor_kwargs["features_dim"] == config.model.features_dim


def test_build_atari_env_wires_sb3_atari_helpers(monkeypatch):
    config = load_atari_config(CONFIG_PATH)
    fake_env = object()
    wrapped_env = object()
    calls = {}

    def fake_make_atari_env(env_id, *, n_envs, seed, wrapper_kwargs):
        calls["make_atari_env"] = {
            "env_id": env_id,
            "n_envs": n_envs,
            "seed": seed,
            "wrapper_kwargs": wrapper_kwargs,
        }
        return fake_env

    def fake_VecFrameStack(env, *, n_stack):
        calls["VecFrameStack"] = {
            "env": env,
            "n_stack": n_stack,
        }
        return wrapped_env

    install_fake_atari_env_tools(monkeypatch, fake_make_atari_env, fake_VecFrameStack)

    result = build_atari_env(config)

    assert calls["make_atari_env"] == {
        "env_id": config.env.env_id,
        "n_envs": config.env.n_envs,
        "seed": config.env.seed,
        "wrapper_kwargs": {
            "terminal_on_life_loss": config.env.terminal_on_life_loss,
        },
    }
    assert calls["VecFrameStack"] == {
        "env": fake_env,
        "n_stack": config.env.n_stack,
    }
    assert result is wrapped_env


def test_build_atari_env_raises_clear_error_without_sb3(monkeypatch):
    config = load_atari_config(CONFIG_PATH)
    real_import = builtins.__import__

    def import_without_sb3(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "stable_baselines3" or name.startswith("stable_baselines3."):
            raise ModuleNotFoundError("No module named 'stable_baselines3'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "stable_baselines3", raising=False)
    monkeypatch.delitem(sys.modules, "stable_baselines3.common", raising=False)
    monkeypatch.delitem(
        sys.modules,
        "stable_baselines3.common.env_util",
        raising=False,
    )
    monkeypatch.delitem(
        sys.modules,
        "stable_baselines3.common.vec_env",
        raising=False,
    )
    monkeypatch.setattr(builtins, "__import__", import_without_sb3)

    with pytest.raises(ImportError, match="Stable-Baselines3 is required"):
        build_atari_env(config)


def test_atari_module_imports_before_sb3_is_installed(monkeypatch):
    config = load_atari_config(CONFIG_PATH)
    real_import = builtins.__import__

    def import_without_sb3(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "stable_baselines3" or name.startswith("stable_baselines3."):
            raise ModuleNotFoundError("No module named 'stable_baselines3'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "stable_baselines3", raising=False)
    monkeypatch.delitem(sys.modules, "stable_baselines3.common", raising=False)
    monkeypatch.delitem(
        sys.modules,
        "stable_baselines3.common.env_util",
        raising=False,
    )
    monkeypatch.delitem(
        sys.modules,
        "stable_baselines3.common.vec_env",
        raising=False,
    )
    monkeypatch.setattr(builtins, "__import__", import_without_sb3)

    import sdam.experiments.atari as atari

    atari = importlib.reload(atari)

    with pytest.raises(ImportError, match="Stable-Baselines3 is required"):
        atari.build_atari_env(config)


def test_build_sdam_atari_model_uses_ppo_constructor(monkeypatch):
    config = load_atari_config(CONFIG_PATH)
    received = {}

    class FakePPO:
        def __init__(self, *args, **kwargs):
            received["args"] = args
            received["kwargs"] = kwargs

    import sdam.experiments.atari as atari

    monkeypatch.setattr(atari, "_load_ppo", lambda: FakePPO)

    model = build_sdam_atari_model(config, env="fake-env", verbose=2)

    assert isinstance(model, FakePPO)
    assert received["args"] == ("CnnPolicy", "fake-env")
    assert (
        received["kwargs"]["policy_kwargs"]["features_extractor_class"]
        is SDAMAtariFeaturesExtractor
    )
    assert received["kwargs"]["learning_rate"] == config.ppo.learning_rate
    assert received["kwargs"]["n_steps"] == config.ppo.n_steps
    assert received["kwargs"]["batch_size"] == config.ppo.batch_size
    assert received["kwargs"]["gamma"] == config.ppo.gamma
    assert received["kwargs"]["gae_lambda"] == config.ppo.gae_lambda
    assert received["kwargs"]["clip_range"] == config.ppo.clip_range
    assert received["kwargs"]["verbose"] == 2


def test_train_sdam_atari_closes_env_and_saves_model(monkeypatch, tmp_path):
    config = load_atari_config(CONFIG_PATH)
    calls = {}

    class FakeEnv:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeModel:
        def __init__(self):
            self.learn_timesteps = None
            self.save_path = None

        def learn(self, *, total_timesteps):
            self.learn_timesteps = total_timesteps

        def save(self, path):
            self.save_path = path

    fake_env = FakeEnv()
    fake_model = FakeModel()

    import sdam.experiments.atari as atari

    def fake_build_atari_env(received_config):
        calls["build_env_config"] = received_config
        return fake_env

    def fake_build_sdam_atari_model(received_config, env, *, verbose):
        calls["build_model_config"] = received_config
        calls["build_model_env"] = env
        calls["build_model_verbose"] = verbose
        return fake_model

    monkeypatch.setattr(atari, "build_atari_env", fake_build_atari_env)
    monkeypatch.setattr(atari, "build_sdam_atari_model", fake_build_sdam_atari_model)

    save_path = tmp_path / "model"
    model = train_sdam_atari(
        config,
        total_timesteps=12,
        save_path=save_path,
        verbose=3,
    )

    assert model is fake_model
    assert calls["build_env_config"] is config
    assert calls["build_model_config"] is config
    assert calls["build_model_env"] is fake_env
    assert calls["build_model_verbose"] == 3
    assert fake_model.learn_timesteps == 12
    assert fake_model.save_path == save_path
    assert fake_env.closed


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
