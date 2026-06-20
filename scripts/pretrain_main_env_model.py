from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain the origin/main-style ENV_MODEL_V2 checkpoint used by "
            "the 160-D Atari vector DQN pipeline."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/atari/alien_sdam_ppo.yaml"))
    parser.add_argument("--vae-path", type=Path, default=Path("vae_Alien.pth"))
    parser.add_argument("--save-path", type=Path, default=Path("env_Alien.pth"))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--collect-log-interval", type=int, default=10)
    parser.add_argument("--train-log-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from sdam.config import load_atari_config
    from sdam.experiments.main_vector_dqn import pretrain_main_env_model

    config = load_atari_config(args.config)
    logger = lambda message: print(message, flush=True)
    result = pretrain_main_env_model(
        config=config,
        vae_path=args.vae_path,
        save_path=args.save_path,
        episodes=args.episodes,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        dataset_path=args.dataset_path,
        collect_log_interval=args.collect_log_interval,
        train_log_interval=args.train_log_interval,
        logger=logger,
    )
    print(
        "main_env_model "
        f"checkpoint_path={result['checkpoint_path']} "
        f"loss={result['loss']} "
        f"updates={result['updates']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
