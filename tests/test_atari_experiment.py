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
    compare_atari_methods,
    build_atari_env,
    build_naturecnn_atari_model,
    build_sdam_atari_model,
    build_sdam_atari_policy_kwargs,
    evaluate_atari_model,
    format_comparison_markdown,
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


def test_build_naturecnn_atari_model_uses_default_cnn_policy(monkeypatch):
    config = load_atari_config(CONFIG_PATH)
    received = {}

    class FakePPO:
        def __init__(self, *args, **kwargs):
            received["args"] = args
            received["kwargs"] = kwargs

    import sdam.experiments.atari as atari

    monkeypatch.setattr(atari, "_load_ppo", lambda: FakePPO)

    model = build_naturecnn_atari_model(config, env="fake-env", verbose=2)

    assert isinstance(model, FakePPO)
    assert received["args"] == ("CnnPolicy", "fake-env")
    assert "policy_kwargs" not in received["kwargs"]
    assert received["kwargs"]["learning_rate"] == config.ppo.learning_rate
    assert received["kwargs"]["verbose"] == 2


def test_evaluate_atari_model_uses_sb3_evaluate_policy(monkeypatch):
    calls = {}

    def fake_evaluate_policy(model, env, *, n_eval_episodes, deterministic, return_episode_rewards):
        calls["model"] = model
        calls["env"] = env
        calls["n_eval_episodes"] = n_eval_episodes
        calls["deterministic"] = deterministic
        calls["return_episode_rewards"] = return_episode_rewards
        return [1.0, 3.0], [10, 20]

    import sdam.experiments.atari as atari

    monkeypatch.setattr(atari, "_load_evaluate_policy", lambda: fake_evaluate_policy)

    metrics = evaluate_atari_model("model", "env", n_eval_episodes=2)

    assert calls == {
        "model": "model",
        "env": "env",
        "n_eval_episodes": 2,
        "deterministic": True,
        "return_episode_rewards": True,
    }
    assert metrics == {
        "mean_reward": 2.0,
        "std_reward": 1.0,
        "mean_ep_length": 15.0,
        "episodes": 2,
    }


def test_compare_atari_methods_trains_and_evaluates_sdam_and_baseline(monkeypatch, tmp_path):
    config = load_atari_config(CONFIG_PATH)
    calls = {"closed": []}

    class FakeEnv:
        def __init__(self, name):
            self.name = name

        def close(self):
            calls["closed"].append(self.name)

    class FakeModel:
        def __init__(self, name):
            self.name = name

        def learn(self, *, total_timesteps):
            calls[f"{self.name}_timesteps"] = total_timesteps

        def save(self, path):
            calls[f"{self.name}_save_path"] = str(path)

    import sdam.experiments.atari as atari

    env_counter = {"value": 0}

    def fake_build_atari_env(received_config):
        env_counter["value"] += 1
        return FakeEnv(f"env-{env_counter['value']}")

    def fake_build_sdam_atari_model(received_config, env, *, verbose):
        calls["sdam_env"] = env.name
        return FakeModel("sdam")

    def fake_build_naturecnn_atari_model(received_config, env, *, verbose):
        calls["naturecnn_env"] = env.name
        return FakeModel("naturecnn")

    def fake_evaluate_atari_model(model, env, *, n_eval_episodes):
        return {
            "mean_reward": 10.0 if model.name == "sdam" else 5.0,
            "std_reward": 1.0,
            "mean_ep_length": 20.0,
            "episodes": n_eval_episodes,
        }

    monkeypatch.setattr(atari, "build_atari_env", fake_build_atari_env)
    monkeypatch.setattr(atari, "build_sdam_atari_model", fake_build_sdam_atari_model)
    monkeypatch.setattr(atari, "build_naturecnn_atari_model", fake_build_naturecnn_atari_model)
    monkeypatch.setattr(atari, "evaluate_atari_model", fake_evaluate_atari_model)

    rows = compare_atari_methods(
        config,
        total_timesteps=12,
        eval_episodes=3,
        output_dir=tmp_path,
        methods=("naturecnn", "sdam"),
        verbose=0,
    )

    assert [row["method"] for row in rows] == ["naturecnn", "sdam"]
    assert rows[0]["mean_reward"] == 5.0
    assert rows[1]["mean_reward"] == 10.0
    assert calls["naturecnn_timesteps"] == 12
    assert calls["sdam_timesteps"] == 12
    assert calls["closed"] == ["env-1", "env-2"]
    assert calls["naturecnn_save_path"].endswith("naturecnn.zip")
    assert calls["sdam_save_path"].endswith("sdam.zip")


def test_format_comparison_markdown_includes_methods_and_rewards():
    markdown = format_comparison_markdown(
        [
            {
                "method": "naturecnn",
                "mean_reward": 5.0,
                "std_reward": 1.0,
                "mean_ep_length": 20.0,
                "episodes": 3,
                "model_path": "runs/naturecnn.zip",
            },
            {
                "method": "sdam",
                "mean_reward": 10.0,
                "std_reward": 2.0,
                "mean_ep_length": 30.0,
                "episodes": 3,
                "model_path": "runs/sdam.zip",
            },
        ]
    )

    assert "| Method | Mean Reward | Std Reward | Mean Episode Length | Episodes | Model Path |" in markdown
    assert "| naturecnn | 5.000 | 1.000 | 20.000 | 3 | runs/naturecnn.zip |" in markdown
    assert "| sdam | 10.000 | 2.000 | 30.000 | 3 | runs/sdam.zip |" in markdown


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


def test_train_sdam_atari_rejects_non_positive_timesteps_before_building_env(
    monkeypatch,
    tmp_path,
):
    config = load_atari_config(CONFIG_PATH)
    env_built = False

    import sdam.experiments.atari as atari

    def fake_build_atari_env(received_config):
        nonlocal env_built
        env_built = True
        return object()

    monkeypatch.setattr(atari, "build_atari_env", fake_build_atari_env)

    with pytest.raises(ValueError, match="total_timesteps must be positive"):
        train_sdam_atari(
            config,
            total_timesteps=0,
            save_path=tmp_path / "model",
        )

    assert not env_built


def test_train_sdam_atari_closes_env_when_model_construction_raises(monkeypatch):
    config = load_atari_config(CONFIG_PATH)

    class FakeEnv:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    fake_env = FakeEnv()

    import sdam.experiments.atari as atari

    monkeypatch.setattr(atari, "build_atari_env", lambda received_config: fake_env)

    def fake_build_sdam_atari_model(received_config, env, *, verbose):
        raise RuntimeError("model construction failed")

    monkeypatch.setattr(atari, "build_sdam_atari_model", fake_build_sdam_atari_model)

    with pytest.raises(RuntimeError, match="model construction failed"):
        train_sdam_atari(config)

    assert fake_env.closed


def test_train_sdam_atari_closes_env_when_learn_raises(monkeypatch, tmp_path):
    config = load_atari_config(CONFIG_PATH)

    class FakeEnv:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeModel:
        def learn(self, *, total_timesteps):
            raise RuntimeError("learn failed")

        def save(self, path):
            raise AssertionError("save should not run after learn failure")

    fake_env = FakeEnv()

    import sdam.experiments.atari as atari

    monkeypatch.setattr(atari, "build_atari_env", lambda received_config: fake_env)
    monkeypatch.setattr(
        atari,
        "build_sdam_atari_model",
        lambda received_config, env, *, verbose: FakeModel(),
    )

    with pytest.raises(RuntimeError, match="learn failed"):
        train_sdam_atari(
            config,
            total_timesteps=12,
            save_path=tmp_path / "model",
        )

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


def test_compare_atari_script_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/compare_atari.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--config" in result.stdout
    assert "--timesteps" in result.stdout
    assert "--eval-episodes" in result.stdout
    assert "--output-dir" in result.stdout
