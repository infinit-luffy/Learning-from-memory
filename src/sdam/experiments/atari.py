from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Any

import torch

from sdam.config import AtariSDAMConfig
from sdam.policies.sb3_atari import (
    SDAMAtariAutoEncoder,
    SDAMAtariFeaturesExtractor,
    atari_observations_to_sdam,
)


_SB3_EXTRA_MESSAGE = (
    'Stable-Baselines3 is required for Atari experiments. Install with: pip install -e ".[atari]"'
)


def _load_ppo():
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(_SB3_EXTRA_MESSAGE) from exc

    return PPO


def _load_atari_env_tools():
    try:
        from stable_baselines3.common.env_util import make_atari_env
        from stable_baselines3.common.vec_env import VecFrameStack
    except ImportError as exc:
        raise ImportError(_SB3_EXTRA_MESSAGE) from exc

    return make_atari_env, VecFrameStack


def _load_evaluate_policy():
    try:
        from stable_baselines3.common.evaluation import evaluate_policy
    except ImportError as exc:
        raise ImportError(_SB3_EXTRA_MESSAGE) from exc

    return evaluate_policy


def build_sdam_atari_policy_kwargs(config: AtariSDAMConfig) -> dict[str, Any]:
    return {
        "features_extractor_class": SDAMAtariFeaturesExtractor,
        "features_extractor_kwargs": {
            "static_dim": config.model.static_dim,
            "dynamic_dim": config.model.dynamic_dim,
            "assoc_dim": config.model.assoc_dim,
            "hidden_channels": config.model.hidden_channels,
            "features_dim": config.model.features_dim,
            "sequence_length": config.env.n_stack,
        },
    }


def build_atari_env(config: AtariSDAMConfig):
    make_atari_env, VecFrameStack = _load_atari_env_tools()
    env = make_atari_env(
        config.env.env_id,
        n_envs=config.env.n_envs,
        seed=config.env.seed,
        wrapper_kwargs={"terminal_on_life_loss": config.env.terminal_on_life_loss},
    )
    return VecFrameStack(env, n_stack=config.env.n_stack)


def build_sdam_atari_model(config: AtariSDAMConfig, env, verbose: int = 1):
    PPO = _load_ppo()
    return PPO(
        "CnnPolicy",
        env,
        policy_kwargs=build_sdam_atari_policy_kwargs(config),
        learning_rate=config.ppo.learning_rate,
        n_steps=config.ppo.n_steps,
        batch_size=config.ppo.batch_size,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_range=config.ppo.clip_range,
        verbose=verbose,
    )


def build_sdam_alternating_atari_model(config: AtariSDAMConfig, env, verbose: int = 1):
    return SDAMAlternatingPPO(
        "CnnPolicy",
        env,
        policy_kwargs=build_sdam_atari_policy_kwargs(config),
        learning_rate=config.ppo.learning_rate,
        n_steps=config.ppo.n_steps,
        batch_size=config.ppo.batch_size,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_range=config.ppo.clip_range,
        verbose=verbose,
        autoencoder_class=SDAMAtariAutoEncoder,
        autoencoder_kwargs={
            "sequence_length": config.env.n_stack,
            "static_dim": config.model.static_dim,
            "dynamic_dim": config.model.dynamic_dim,
            "assoc_dim": config.model.assoc_dim,
            "hidden_channels": config.model.hidden_channels,
            "reconstruction_weight": config.pretraining.reconstruction_weight,
            "prediction_weight": config.pretraining.prediction_weight,
        },
        alternating_interval=config.alternating.interval,
        alternating_updates=config.alternating.updates,
        auxiliary_batch_size=config.alternating.batch_size,
        auxiliary_learning_rate=config.alternating.learning_rate,
        pretrained_path=config.alternating.pretrained_path,
        reconstruction_weight=config.pretraining.reconstruction_weight,
        prediction_weight=config.pretraining.prediction_weight,
    )


def build_naturecnn_atari_model(config: AtariSDAMConfig, env, verbose: int = 1):
    PPO = _load_ppo()
    return PPO(
        "CnnPolicy",
        env,
        learning_rate=config.ppo.learning_rate,
        n_steps=config.ppo.n_steps,
        batch_size=config.ppo.batch_size,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_range=config.ppo.clip_range,
        verbose=verbose,
    )


def _unpack_reset(result):
    if isinstance(result, tuple):
        return result[0]
    return result


def _unpack_step(result):
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, reward, terminated or truncated, info
    return result


def _first_observation(observation, sequence_length: int) -> torch.Tensor:
    tensor = torch.as_tensor(observation)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.shape != (sequence_length, 84, 84):
        raise ValueError("Atari observations must have shape [T, 84, 84] or [N, T, 84, 84]")
    return tensor.detach().cpu().to(torch.uint8)


def collect_random_atari_sequences(
    env,
    steps: int,
    sequence_length: int,
    output_path: str | Path,
) -> Path:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    observation = _unpack_reset(env.reset())
    samples = []
    for _ in range(steps):
        action = env.action_space.sample()
        if getattr(env, "num_envs", None):
            action = [env.action_space.sample() for _ in range(env.num_envs)]
        observation, reward, done, info = _unpack_step(env.step(action))
        samples.append(_first_observation(observation, sequence_length))
        if bool(torch.as_tensor(done).any().item()) and not getattr(env, "num_envs", None):
            observation = _unpack_reset(env.reset())

    torch.save({"observations": torch.stack(samples, dim=0)}, output)
    return output


class SDAMAtariPretrainer:
    def __init__(self, config: AtariSDAMConfig) -> None:
        self.config = config
        self.autoencoder = SDAMAtariAutoEncoder(
            sequence_length=config.env.n_stack,
            static_dim=config.model.static_dim,
            dynamic_dim=config.model.dynamic_dim,
            assoc_dim=config.model.assoc_dim,
            hidden_channels=config.model.hidden_channels,
            reconstruction_weight=config.pretraining.reconstruction_weight,
            prediction_weight=config.pretraining.prediction_weight,
        )

    def train(
        self,
        dataset_path: str | Path,
        save_path: str | Path,
        train_steps: int | None = None,
    ) -> dict[str, str | float | int]:
        steps = train_steps if train_steps is not None else self.config.pretraining.train_steps
        if steps <= 0:
            raise ValueError("train_steps must be positive")

        payload = torch.load(Path(dataset_path), map_location="cpu")
        observations = payload["observations"]
        if observations.ndim != 4:
            raise ValueError("dataset observations must have shape [N, T, 84, 84]")
        if observations.shape[1] != self.config.env.n_stack:
            raise ValueError(f"dataset frame stack must be {self.config.env.n_stack}")

        optimizer = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=self.config.pretraining.learning_rate,
        )
        batch_size = min(self.config.pretraining.batch_size, observations.shape[0])
        last_loss = 0.0
        for _ in range(steps):
            indices = torch.randint(0, observations.shape[0], (batch_size,))
            batch = atari_observations_to_sdam(
                observations[indices],
                self.config.env.n_stack,
            )
            optimizer.zero_grad()
            outputs = self.autoencoder(batch)
            outputs["loss"].backward()
            optimizer.step()
            last_loss = float(outputs["loss"].detach().cpu().item())

        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "sdam_autoencoder.pt"
        torch.save(
            {
                "model_state_dict": self.autoencoder.state_dict(),
                "encoder_state_dict": self.autoencoder.encoder.state_dict(),
                "loss": last_loss,
            },
            checkpoint_path,
        )
        return {
            "checkpoint_path": str(checkpoint_path),
            "loss": last_loss,
            "train_steps": steps,
        }


class SDAMAlternatingPPO:
    def __init__(
        self,
        *ppo_args,
        autoencoder_class,
        autoencoder_kwargs: dict[str, Any],
        alternating_interval: int,
        alternating_updates: int,
        auxiliary_batch_size: int,
        auxiliary_learning_rate: float,
        pretrained_path: str = "",
        reconstruction_weight: float = 1.0,
        prediction_weight: float = 1.0,
        **ppo_kwargs,
    ) -> None:
        if alternating_interval <= 0:
            raise ValueError("alternating_interval must be positive")
        if alternating_updates <= 0:
            raise ValueError("alternating_updates must be positive")

        PPO = _load_ppo()
        self.model = PPO(*ppo_args, **ppo_kwargs)
        self.alternating_interval = alternating_interval
        self.alternating_updates = alternating_updates
        self.auxiliary_batch_size = auxiliary_batch_size
        self.reconstruction_weight = reconstruction_weight
        self.prediction_weight = prediction_weight
        self.autoencoder = autoencoder_class(**autoencoder_kwargs)
        self._tie_encoder_to_policy()
        if pretrained_path and Path(pretrained_path).exists():
            self.load_pretrained_autoencoder(pretrained_path)
        elif pretrained_path:
            warnings.warn(
                f"pretrained SDAM checkpoint not found: {pretrained_path}; "
                "continuing without pretraining weights",
                RuntimeWarning,
                stacklevel=2,
            )
        self.auxiliary_optimizer = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=auxiliary_learning_rate,
        )
        self.auxiliary_losses: list[float] = []

    def __getattr__(self, name: str):
        if name == "model":
            raise AttributeError(name)
        return getattr(self.model, name)

    def _tie_encoder_to_policy(self) -> None:
        features_extractor = getattr(self.model.policy, "features_extractor", None)
        if features_extractor is None:
            return
        encoder = getattr(features_extractor, "encoder", None)
        if encoder is not None:
            self.autoencoder.encoder = encoder

    def load_pretrained_autoencoder(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.autoencoder.load_state_dict(state_dict, strict=False)
        self._tie_encoder_to_policy()

    def learn(self, total_timesteps: int, **kwargs):
        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        num_envs = int(getattr(self.model.env, "num_envs", 1))
        n_steps = int(getattr(self.model, "n_steps", 1))
        chunk_size = max(1, self.alternating_interval * n_steps * num_envs)
        remaining = total_timesteps
        reset_num_timesteps = kwargs.pop("reset_num_timesteps", True)
        while remaining > 0:
            chunk = min(chunk_size, remaining)
            self.model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=reset_num_timesteps,
                **kwargs,
            )
            reset_num_timesteps = False
            remaining -= chunk
            observations = self._rollout_observations()
            if observations is not None:
                self.update_autoencoder(observations)
        return self

    def _rollout_observations(self) -> torch.Tensor | None:
        rollout_buffer = getattr(self.model, "rollout_buffer", None)
        observations = getattr(rollout_buffer, "observations", None)
        if observations is None:
            return None
        tensor = torch.as_tensor(observations)
        if tensor.ndim == 5:
            tensor = tensor.reshape(-1, *tensor.shape[2:])
        if tensor.ndim != 4:
            return None
        return tensor

    def update_autoencoder(self, observations: torch.Tensor) -> float:
        if observations.shape[0] == 0:
            return 0.0
        batch_size = min(self.auxiliary_batch_size, observations.shape[0])
        last_loss = 0.0
        for _ in range(self.alternating_updates):
            indices = torch.randint(0, observations.shape[0], (batch_size,))
            batch = atari_observations_to_sdam(
                observations[indices],
                self.autoencoder.sequence_length,
            )
            self.auxiliary_optimizer.zero_grad()
            outputs = self.autoencoder(batch)
            outputs["loss"].backward()
            self.auxiliary_optimizer.step()
            last_loss = float(outputs["loss"].detach().cpu().item())
            self.auxiliary_losses.append(last_loss)
        return last_loss

    def save(self, path):
        return self.model.save(path)


def evaluate_atari_model(model, env, n_eval_episodes: int = 10) -> dict[str, float | int]:
    if n_eval_episodes <= 0:
        raise ValueError("n_eval_episodes must be positive")
    evaluate_policy = _load_evaluate_policy()
    rewards, lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        return_episode_rewards=True,
    )
    mean_reward = sum(rewards) / len(rewards)
    mean_length = sum(lengths) / len(lengths)
    reward_variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
    return {
        "mean_reward": float(mean_reward),
        "std_reward": float(reward_variance ** 0.5),
        "mean_ep_length": float(mean_length),
        "episodes": len(rewards),
    }


def train_sdam_atari(
    config: AtariSDAMConfig,
    total_timesteps: int | None = None,
    save_path: str | Path | None = None,
    verbose: int = 1,
):
    steps = (
        total_timesteps
        if total_timesteps is not None
        else config.training.total_timesteps
    )
    if steps <= 0:
        raise ValueError("total_timesteps must be positive")

    env = None
    try:
        env = build_atari_env(config)
        model = build_sdam_atari_model(config, env, verbose=verbose)
        output_path = Path(save_path if save_path is not None else config.training.save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.learn(total_timesteps=steps)
        model.save(output_path)
        return model
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()


def compare_atari_methods(
    config: AtariSDAMConfig,
    total_timesteps: int,
    eval_episodes: int,
    output_dir: str | Path,
    methods: tuple[str, ...] = ("naturecnn", "sdam"),
    verbose: int = 1,
) -> list[dict[str, str | float | int]]:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float | int]] = []
    for method in methods:
        if method not in ("naturecnn", "sdam", "sdam_alternating"):
            raise ValueError(f"unknown Atari comparison method: {method}")
        env = None
        try:
            env = build_atari_env(config)
            if method == "naturecnn":
                model = build_naturecnn_atari_model(config, env, verbose=verbose)
            elif method == "sdam":
                model = build_sdam_atari_model(config, env, verbose=verbose)
            else:
                model = build_sdam_alternating_atari_model(config, env, verbose=verbose)

            model.learn(total_timesteps=total_timesteps)
            model_path = output_path / f"{method}.zip"
            model.save(model_path)
            metrics = evaluate_atari_model(model, env, n_eval_episodes=eval_episodes)
            rows.append({"method": method, **metrics, "model_path": str(model_path)})
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()

    write_comparison_outputs(rows, output_path)
    return rows


def write_comparison_outputs(
    rows: list[dict[str, str | float | int]],
    output_dir: str | Path,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "comparison.csv"
    fieldnames = [
        "method",
        "mean_reward",
        "std_reward",
        "mean_ep_length",
        "episodes",
        "model_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_path / "comparison.md").write_text(
        format_comparison_markdown(rows),
        encoding="utf-8",
    )


def format_comparison_markdown(rows: list[dict[str, str | float | int]]) -> str:
    lines = [
        "| Method | Mean Reward | Std Reward | Mean Episode Length | Episodes | Model Path |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {mean_reward:.3f} | {std_reward:.3f} | "
            "{mean_ep_length:.3f} | {episodes} | {model_path} |".format(**row)
        )
    return "\n".join(lines) + "\n"
