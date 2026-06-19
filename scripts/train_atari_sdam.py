from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/atari/sdam_ppo.yaml"),
    )
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from sdam.config import load_atari_config
    from sdam.experiments.atari import train_sdam_atari

    config = load_atari_config(args.config)
    train_sdam_atari(
        config,
        total_timesteps=args.timesteps,
        save_path=args.save_path,
        verbose=args.verbose,
        device=args.device,
    )


if __name__ == "__main__":
    main()
