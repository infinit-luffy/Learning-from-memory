from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run random collection, SDAM pretraining, and Atari PPO comparison."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/atari/alien_sdam_alternating_ppo.yaml"),
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/alien/pipeline"))
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--collect-log-interval", type=int, default=1000)
    parser.add_argument("--train-log-interval", type=int, default=100)
    parser.add_argument("--alternating-log-interval", type=int, default=10)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("naturecnn", "sdam", "sdam_alternating"),
        default=("naturecnn", "sdam", "sdam_alternating"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from sdam.config import load_atari_config
    from sdam.experiments.atari import run_atari_sdam_pipeline

    logger = lambda message: print(message, flush=True)
    result = run_atari_sdam_pipeline(
        load_atari_config(args.config),
        output_dir=args.output_dir,
        collect_steps=args.steps,
        pretrain_steps=args.train_steps,
        total_timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        methods=tuple(args.methods),
        verbose=args.verbose,
        device=args.device,
        collect_log_interval=args.collect_log_interval,
        train_log_interval=args.train_log_interval,
        alternating_log_interval=args.alternating_log_interval,
        logger=logger,
    )
    print(f"dataset_path={result['dataset_path']}", flush=True)
    print(f"checkpoint_path={result['checkpoint_path']}", flush=True)
    print(f"comparison_dir={result['comparison_dir']}", flush=True)


if __name__ == "__main__":
    main()
