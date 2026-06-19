from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from sdam.config import AtariSDAMConfig
from sdam.policies.sb3_atari import SDAMAtariFeaturesExtractor


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
        if method not in ("naturecnn", "sdam"):
            raise ValueError(f"unknown Atari comparison method: {method}")
        env = None
        try:
            env = build_atari_env(config)
            if method == "naturecnn":
                model = build_naturecnn_atari_model(config, env, verbose=verbose)
            else:
                model = build_sdam_atari_model(config, env, verbose=verbose)

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
