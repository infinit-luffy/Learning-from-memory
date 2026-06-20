from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the origin/main-style Atari agent: frozen pretrained SDAM/VAE "
            "representation -> 160-D vector observation -> SB3 DQN MlpPolicy."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/atari/alien_sdam_ppo.yaml"))
    parser.add_argument("--vae-path", type=Path, default=Path("vae_Alien.pth"))
    parser.add_argument("--env-model-path", type=Path, default=Path("env_Alien.pth"))
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/alien/main_vector_dqn"))
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=500000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from sdam.config import load_atari_config
    from sdam.experiments.main_vector_dqn import train_main_vector_dqn

    config = load_atari_config(args.config)
    row = train_main_vector_dqn(
        config=config,
        vae_path=args.vae_path,
        env_model_path=args.env_model_path,
        total_timesteps=args.timesteps,
        save_path=args.save_path,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        device=args.device,
        verbose=args.verbose,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_rate=args.learning_rate,
    )
    print(
        "main_vector_dqn "
        f"mean_reward={row['mean_reward']} "
        f"std_reward={row['std_reward']} "
        f"model_path={row['model_path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
