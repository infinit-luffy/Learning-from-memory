import builtins
import importlib
import subprocess
import sys
import types
import warnings
from pathlib import Path

import pytest
import torch

warnings.filterwarnings(
    "ignore",
    message="Failed to initialize NumPy: No module named 'numpy'.*",
    category=UserWarning,
)

from sdam.config import load_atari_config
from sdam.experiments.atari import (
    SDAMAtariPretrainer,
    collect_random_atari_sequences,
    compare_atari_methods,
    build_atari_env,
    build_naturecnn_atari_model,
    build_sdam_atari_model,
    build_sdam_atari_policy_kwargs,
    evaluate_atari_model,
    format_comparison_markdown,
    _resolve_device,
    _safe_torch_load,
    train_sdam_atari,
)
from sdam.experiments.main_vector_dqn import (
    MainEnvModelV2,
    MainVectorObservationWrapper,
    _ale_fallback_env_id,
    _make_gym_atari_env,
    collect_main_env_model_dataset,
    train_main_vae_from_frames,
    train_main_env_model_from_dataset,
)
from sdam.policies.sb3_atari import SDAMAtariAutoEncoder, SDAMAtariFeaturesExtractor


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


def install_fake_gymnasium(monkeypatch):
    gymnasium = types.ModuleType("gymnasium")
    spaces = types.ModuleType("gymnasium.spaces")

    class Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class ObservationWrapper:
        def __init__(self, env):
            self.env = env
            self.observation_space = getattr(env, "observation_space", None)

        def reset(self, **kwargs):
            result = self.env.reset(**kwargs)
            observation, info = result if isinstance(result, tuple) else (result, {})
            return self.observation(observation), info

        def step(self, action):
            observation, reward, terminated, truncated, info = self.env.step(action)
            return self.observation(observation), reward, terminated, truncated, info

    spaces.Box = Box
    gymnasium.ObservationWrapper = ObservationWrapper
    gymnasium.spaces = spaces
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    monkeypatch.setitem(sys.modules, "gymnasium.spaces", spaces)


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


def test_main_vector_observation_wrapper_outputs_origin_main_vector(monkeypatch):
    install_fake_gymnasium(monkeypatch)

    class FakeEnv:
        observation_space = None

        def reset(self, **kwargs):
            return torch.ones(84, 84, 1).numpy(), {}

        def step(self, action):
            return torch.ones(84, 84, 1).numpy(), 0.0, False, False, {}

    class FakeVAE:
        def eval(self):
            return self

        def generate(self, frame):
            return torch.zeros(1, 1, 84, 84)

        def encode(self, frame):
            return torch.arange(32, dtype=torch.float32).unsqueeze(0), torch.zeros(1, 32)

        def reparameterize(self, mu, log_var):
            return mu

    class FakeEnvModel:
        def eval(self):
            return self

        def get_feature_encode(self, frame):
            return torch.full((1, 32), 2.0)

        def get_z_encode(self, state, background):
            return torch.full((1, 32), 3.0)

    wrapper = MainVectorObservationWrapper(
        FakeEnv(),
        FakeVAE(),
        FakeEnvModel(),
        torch.device("cpu"),
    ).unwrap()

    vector = wrapper.observation(torch.ones(84, 84, 1).numpy())

    assert vector.shape == (192,)
    assert vector.dtype.name == "float32"
    assert vector[:96].tolist() == [0.0] * 96
    assert vector[96:128].tolist() == [2.0] * 32
    assert vector[128:160].tolist() == [3.0] * 32
    assert vector[160:].tolist() == [float(index) for index in range(32)]


def test_main_vector_env_model_keeps_origin_main_checkpoint_keys():
    state_dict = MainEnvModelV2(input_size=32, hidden_size=128).state_dict()

    assert "connection_recog.atten.attn.weight" in state_dict
    assert "connection_recog.atten.attn.bias" in state_dict
    assert "feature_recog.fc_mu.weight" in state_dict


def test_main_vector_maps_legacy_atari_env_id_to_gymnasium_ale():
    assert _ale_fallback_env_id("AlienNoFrameskip-v4") == "ALE/Alien-v5"
    assert _ale_fallback_env_id("PongNoFrameskip-v4") == "ALE/Pong-v5"
    assert _ale_fallback_env_id("ALE/Alien-v5") is None


def test_make_gym_atari_env_tries_ale_fallback():
    calls = []

    class FakeGym:
        def make(self, env_id, **kwargs):
            calls.append((env_id, kwargs))
            if env_id == "ALE/Alien-v5" and kwargs.get("frameskip") == 1:
                return "fake-env"
            raise RuntimeError(f"missing {env_id}")

    assert _make_gym_atari_env(FakeGym(), "AlienNoFrameskip-v4") == "fake-env"
    assert calls == [
        ("AlienNoFrameskip-v4", {}),
        ("ALE/Alien-v5", {"frameskip": 1, "repeat_action_probability": 0.0}),
    ]


def test_make_gym_atari_env_reports_all_attempts():
    class FakeGym:
        def make(self, env_id, **kwargs):
            raise RuntimeError(f"missing {env_id}")

    with pytest.raises(RuntimeError) as exc_info:
        _make_gym_atari_env(FakeGym(), "AlienNoFrameskip-v4")

    message = str(exc_info.value)
    assert "AlienNoFrameskip-v4" in message
    assert "ALE/Alien-v5" in message
    assert "pip install" in message


def test_train_main_vae_from_frames_saves_checkpoint(tmp_path):
    frames = torch.rand(2, 84, 84)
    save_path = tmp_path / "vae_Alien.pth"

    result = train_main_vae_from_frames(
        frames,
        save_path=save_path,
        train_steps=1,
        batch_size=1,
        learning_rate=1e-4,
        device="cpu",
    )

    assert save_path.exists()
    assert result["checkpoint_path"] == str(save_path)
    assert result["train_steps"] == 1
    assert result["loss"] >= 0.0


def test_collect_main_env_model_dataset_matches_origin_main_shapes():
    class FakeActionSpace:
        def sample(self):
            return 0

    class FakeEnv:
        action_space = FakeActionSpace()

        def __init__(self):
            self.step_count = 0

        def reset(self):
            self.step_count = 0
            return torch.zeros(1, 84, 84, 1)

        def step(self, action):
            self.step_count += 1
            observation = torch.full((1, 84, 84, 1), float(self.step_count))
            done = self.step_count >= 6
            return observation, 0.0, done, {}

    class FakeVAE:
        def eval(self):
            return self

        def generate(self, frame):
            return torch.zeros_like(frame)

    dataset = collect_main_env_model_dataset(
        FakeEnv(),
        FakeVAE(),
        episodes=2,
        device="cpu",
    )

    assert dataset["org"].shape == (4, 1, 84, 84)
    assert dataset["feature"].shape == (4, 5, 1, 84, 84)
    assert dataset["background"].shape == (4, 1, 1, 60, 45)
    assert dataset["feature"].dtype == torch.float32


def test_train_main_env_model_from_dataset_saves_checkpoint(tmp_path):
    dataset = {
        "org": torch.rand(2, 1, 84, 84),
        "feature": torch.rand(2, 5, 1, 84, 84),
        "background": torch.rand(2, 1, 1, 60, 45),
    }
    save_path = tmp_path / "env_Alien.pth"

    result = train_main_env_model_from_dataset(
        dataset,
        save_path=save_path,
        train_steps=1,
        batch_size=1,
        learning_rate=1e-4,
        device="cpu",
    )

    assert save_path.exists()
    assert result["checkpoint_path"] == str(save_path)
    assert result["train_steps"] == 1
    assert result["loss"] >= 0.0
    assert result["next_feature_loss"] >= 0.0


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

    model = build_sdam_atari_model(config, env="fake-env", verbose=2, device="cuda")

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
    assert received["kwargs"]["device"] == "cuda"


def test_build_naturecnn_atari_model_uses_default_cnn_policy(monkeypatch):
    config = load_atari_config(CONFIG_PATH)
    received = {}

    class FakePPO:
        def __init__(self, *args, **kwargs):
            received["args"] = args
            received["kwargs"] = kwargs

    import sdam.experiments.atari as atari

    monkeypatch.setattr(atari, "_load_ppo", lambda: FakePPO)

    model = build_naturecnn_atari_model(config, env="fake-env", verbose=2, device="cuda")

    assert isinstance(model, FakePPO)
    assert received["args"] == ("CnnPolicy", "fake-env")
    assert "policy_kwargs" not in received["kwargs"]
    assert received["kwargs"]["learning_rate"] == config.ppo.learning_rate
    assert received["kwargs"]["verbose"] == 2
    assert received["kwargs"]["device"] == "cuda"


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


def test_collect_random_atari_sequences_samples_expected_shape(tmp_path):
    class FakeActionSpace:
        def sample(self):
            return 0

    class FakeEnv:
        action_space = FakeActionSpace()

        def __init__(self):
            self.step_count = 0

        def reset(self):
            return torch.zeros(4, 84, 84, dtype=torch.uint8)

        def step(self, action):
            self.step_count += 1
            observation = torch.full(
                (4, 84, 84),
                self.step_count,
                dtype=torch.uint8,
            )
            return observation, 0.0, False, {}

    output_path = tmp_path / "random_sequences.pt"
    logs = []

    dataset_path = collect_random_atari_sequences(
        FakeEnv(),
        steps=5,
        sequence_length=4,
        output_path=output_path,
        log_interval=2,
        logger=logs.append,
    )

    assert dataset_path == output_path
    payload = torch.load(output_path)
    assert payload["observations"].shape == (5, 4, 84, 84)
    assert payload["observations"].dtype == torch.uint8
    assert any("step=2/5" in message for message in logs)
    assert any("saved samples=5" in message for message in logs)


def test_collect_random_atari_sequences_accepts_channel_last_vec_stack(tmp_path):
    class FakeActionSpace:
        def sample(self):
            return 0

    class FakeEnv:
        action_space = FakeActionSpace()
        num_envs = 1

        def reset(self):
            return torch.zeros(1, 84, 84, 4, dtype=torch.uint8)

        def step(self, action):
            observation = torch.zeros(1, 84, 84, 4, dtype=torch.uint8)
            observation[0, :, :, 2] = 7
            return observation, torch.tensor([0.0]), torch.tensor([False]), [{}]

    output_path = tmp_path / "channel_last_sequences.pt"

    collect_random_atari_sequences(
        FakeEnv(),
        steps=1,
        sequence_length=4,
        output_path=output_path,
    )

    payload = torch.load(output_path)
    assert payload["observations"].shape == (1, 4, 84, 84)
    assert torch.all(payload["observations"][0, 2] == 7)


def test_sdam_atari_pretrainer_saves_checkpoint(tmp_path):
    config = load_atari_config(Path("configs/atari/alien_sdam_pretrain.yaml"))
    observations = torch.randint(
        0,
        256,
        (4, config.env.n_stack, 84, 84),
        dtype=torch.uint8,
    )
    dataset_path = tmp_path / "dataset.pt"
    torch.save({"observations": observations}, dataset_path)

    pretrainer = SDAMAtariPretrainer(config, device="cpu")
    logs = []
    result = pretrainer.train(
        dataset_path=dataset_path,
        save_path=tmp_path / "pretrain",
        train_steps=1,
        log_interval=1,
        logger=logs.append,
    )

    assert result["checkpoint_path"].endswith("sdam_autoencoder.pt")
    assert Path(result["checkpoint_path"]).exists()
    assert result["loss"] >= 0.0
    assert result["device"] == "cpu"
    assert any("device=cpu" in message for message in logs)
    assert any("step=1/1" in message for message in logs)
    assert any("saved checkpoint=" in message for message in logs)


def test_resolve_device_auto_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    device = _resolve_device("auto")

    assert device.type == "cuda"


def test_resolve_device_auto_falls_back_to_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    device = _resolve_device("auto")

    assert device.type == "cpu"


def test_resolve_device_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA was requested"):
        _resolve_device("cuda")


def test_safe_torch_load_requests_weights_only(monkeypatch, tmp_path):
    calls = {}

    def fake_load(path, *, map_location, weights_only):
        calls["path"] = path
        calls["map_location"] = map_location
        calls["weights_only"] = weights_only
        return {"observations": torch.zeros(1, 4, 84, 84)}

    monkeypatch.setattr(torch, "load", fake_load)

    payload = _safe_torch_load(tmp_path / "dataset.pt", map_location="cpu")

    assert payload["observations"].shape == (1, 4, 84, 84)
    assert calls["map_location"] == "cpu"
    assert calls["weights_only"] is True


def test_compare_atari_methods_trains_and_evaluates_sdam_baseline(
    monkeypatch,
    tmp_path,
):
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

    def fake_build_sdam_atari_model(received_config, env, *, verbose, device):
        calls["sdam_env"] = env.name
        calls["sdam_device"] = device
        return FakeModel("sdam")

    def fake_build_naturecnn_atari_model(received_config, env, *, verbose, device):
        calls["naturecnn_env"] = env.name
        calls["naturecnn_device"] = device
        return FakeModel("naturecnn")

    def fake_evaluate_atari_model(model, env, *, n_eval_episodes):
        return {
            "mean_reward": {
                "naturecnn": 5.0,
                "sdam": 10.0,
            }[model.name],
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
        device="cuda",
    )

    assert [row["method"] for row in rows] == ["naturecnn", "sdam"]
    assert rows[0]["mean_reward"] == 5.0
    assert rows[1]["mean_reward"] == 10.0
    assert calls["naturecnn_timesteps"] == 12
    assert calls["sdam_timesteps"] == 12
    assert calls["closed"] == ["env-1", "env-2"]
    assert calls["naturecnn_device"] == "cuda"
    assert calls["sdam_device"] == "cuda"
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

    def fake_build_sdam_atari_model(received_config, env, *, verbose, device):
        calls["build_model_config"] = received_config
        calls["build_model_env"] = env
        calls["build_model_verbose"] = verbose
        calls["build_model_device"] = device
        return fake_model

    monkeypatch.setattr(atari, "build_atari_env", fake_build_atari_env)
    monkeypatch.setattr(atari, "build_sdam_atari_model", fake_build_sdam_atari_model)

    save_path = tmp_path / "model"
    model = train_sdam_atari(
        config,
        total_timesteps=12,
        save_path=save_path,
        verbose=3,
        device="cuda",
    )

    assert model is fake_model
    assert calls["build_env_config"] is config
    assert calls["build_model_config"] is config
    assert calls["build_model_env"] is fake_env
    assert calls["build_model_verbose"] == 3
    assert calls["build_model_device"] == "cuda"
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

    def fake_build_sdam_atari_model(received_config, env, *, verbose, device):
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
        lambda received_config, env, *, verbose, device: FakeModel(),
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
    assert "--device" in result.stdout


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
    assert "--device" in result.stdout
    assert "--alternating-log-interval" not in result.stdout
    assert "sdam_alternating" not in result.stdout


def test_pretrain_atari_script_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/pretrain_atari_sdam.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--config" in result.stdout
    assert "--steps" in result.stdout
    assert "--train-steps" in result.stdout
    assert "--device" in result.stdout
    assert "--collect-log-interval" in result.stdout
    assert "--train-log-interval" in result.stdout
    assert "--save-path" in result.stdout


def test_train_main_vector_dqn_script_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/train_main_vector_dqn.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--config" in result.stdout
    assert "--timesteps" in result.stdout
    assert "--eval-episodes" in result.stdout
    assert "--vae-path" in result.stdout
    assert "--env-model-path" in result.stdout
    assert "--device" in result.stdout
    assert "--batch-size" in result.stdout
    assert "192-D memory vector observation" in result.stdout


def test_train_atari_dqn_baseline_script_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/train_atari_dqn_baseline.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--config" in result.stdout
    assert "--timesteps" in result.stdout
    assert "--eval-episodes" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--learning-rate" in result.stdout
    assert "--batch-size" in result.stdout
    assert "NatureCNN DQN" in result.stdout


def test_pretrain_main_env_model_script_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/pretrain_main_env_model.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--config" in result.stdout
    assert "--vae-path" in result.stdout
    assert "--save-path" in result.stdout
    assert "--episodes" in result.stdout
    assert "--train-steps" in result.stdout
    assert "--prediction-weight" in result.stdout
    assert "--device" in result.stdout


def test_run_main_atari_pipeline_script_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/run_main_atari_pipeline.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--output-dir" in result.stdout
    assert "--vae-episodes" in result.stdout
    assert "--vae-train-steps" in result.stdout
    assert "--env-episodes" in result.stdout
    assert "--env-train-steps" in result.stdout
    assert "--env-prediction-weight" in result.stdout
    assert "--rl-algo" in result.stdout
    assert "dqn" in result.stdout
    assert "ppo" in result.stdout
