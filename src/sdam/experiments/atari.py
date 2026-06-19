from __future__ import annotations

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
