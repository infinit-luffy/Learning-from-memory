from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate NatureCNN, SDAM, and alternating SDAM PPO on one Atari config."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/atari/sdam_ppo.yaml"))
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/atari/compare"))
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("naturecnn", "sdam", "sdam_alternating"),
        default=("naturecnn", "sdam"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from sdam.config import load_atari_config
    from sdam.experiments.atari import compare_atari_methods, format_comparison_markdown

    config = load_atari_config(args.config)
    rows = compare_atari_methods(
        config,
        total_timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        methods=tuple(args.methods),
        verbose=args.verbose,
        device=args.device,
    )
    print(format_comparison_markdown(rows))


if __name__ == "__main__":
    main()
