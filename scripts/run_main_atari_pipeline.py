from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full origin/main-style Atari pipeline: train VAE, train "
            "ENV_MODEL_V2, then train a vector-state RL agent."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/atari/alien_sdam_ppo.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/alien/full_pipeline"))
    parser.add_argument("--vae-episodes", type=int, default=2000)
    parser.add_argument("--vae-train-steps", type=int, default=300)
    parser.add_argument("--vae-batch-size", type=int, default=128)
    parser.add_argument("--vae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--env-episodes", type=int, default=2000)
    parser.add_argument("--env-train-steps", type=int, default=300)
    parser.add_argument("--env-batch-size", type=int, default=128)
    parser.add_argument("--env-learning-rate", type=float, default=3e-4)
    parser.add_argument("--rl-algo", choices=("dqn", "ppo"), default="dqn")
    parser.add_argument("--timesteps", type=int, default=5000000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--rl-learning-rate", type=float, default=1e-4)
    parser.add_argument("--rl-batch-size", type=int, default=256)
    parser.add_argument("--rl-buffer-size", type=int, default=500000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--collect-log-interval", type=int, default=10)
    parser.add_argument("--train-log-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from sdam.config import load_atari_config
    from sdam.experiments.main_vector_dqn import run_main_atari_pipeline

    config = load_atari_config(args.config)
    logger = lambda message: print(message, flush=True)
    result = run_main_atari_pipeline(
        config=config,
        output_dir=args.output_dir,
        vae_episodes=args.vae_episodes,
        vae_train_steps=args.vae_train_steps,
        env_episodes=args.env_episodes,
        env_train_steps=args.env_train_steps,
        rl_algo=args.rl_algo,
        total_timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        device=args.device,
        vae_batch_size=args.vae_batch_size,
        env_batch_size=args.env_batch_size,
        vae_learning_rate=args.vae_learning_rate,
        env_learning_rate=args.env_learning_rate,
        rl_learning_rate=args.rl_learning_rate,
        rl_batch_size=args.rl_batch_size,
        rl_buffer_size=args.rl_buffer_size,
        verbose=args.verbose,
        collect_log_interval=args.collect_log_interval,
        train_log_interval=args.train_log_interval,
        logger=logger,
    )
    print(
        "main_pipeline "
        f"vae={result['vae']['checkpoint_path']} "
        f"env_model={result['env_model']['checkpoint_path']} "
        f"rl_model={result['rl']['model_path']} "
        f"mean_reward={result['rl']['mean_reward']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
