from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Stable-Baselines3 NatureCNN DQN Atari baseline."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/atari/alien_sdam_ppo.yaml"))
    parser.add_argument("--timesteps", type=int, default=5000000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/alien/naturecnn_dqn"))
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=500000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from sdam.config import load_atari_config
    from sdam.experiments.atari import train_naturecnn_dqn_atari

    config = load_atari_config(args.config)
    row = train_naturecnn_dqn_atari(
        config=config,
        total_timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        save_path=args.save_path,
        verbose=args.verbose,
        device=args.device,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
    )
    print(
        "naturecnn_dqn "
        f"mean_reward={row['mean_reward']} "
        f"std_reward={row['std_reward']} "
        f"model_path={row['model_path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
