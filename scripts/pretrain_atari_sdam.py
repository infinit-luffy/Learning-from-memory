from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect random Atari frame stacks and pretrain the SDAM autoencoder."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/atari/alien_sdam_pretrain.yaml"),
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--save-path", type=Path, default=Path("runs/alien/sdam_pretrain"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from sdam.config import load_atari_config
    from sdam.experiments.atari import (
        SDAMAtariPretrainer,
        build_atari_env,
        collect_random_atari_sequences,
    )

    config = load_atari_config(args.config)
    steps = args.steps if args.steps is not None else config.pretraining.collect_steps
    dataset_path = (
        args.dataset_path
        if args.dataset_path is not None
        else Path(config.pretraining.dataset_path)
    )

    env = None
    try:
        env = build_atari_env(config)
        collected_path = collect_random_atari_sequences(
            env,
            steps=steps,
            sequence_length=config.env.n_stack,
            output_path=dataset_path,
        )
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()

    result = SDAMAtariPretrainer(config).train(
        dataset_path=collected_path,
        save_path=args.save_path,
        train_steps=args.train_steps,
    )
    print(f"checkpoint_path={result['checkpoint_path']}")
    print(f"loss={result['loss']:.6f}")


if __name__ == "__main__":
    main()
