from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdam.config import load_config
from sdam.experiments import build_synthetic_components, train_one_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic/sdam.yaml"))
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    components = build_synthetic_components(config)
    steps = args.steps if args.steps is not None else config.training.train_steps
    for step in range(1, steps + 1):
        metrics = train_one_step(components)
        print(
            f"step={step} loss={metrics['loss']:.6f} "
            f"position_loss={metrics['position_loss']:.6f} "
            f"velocity_loss={metrics['velocity_loss']:.6f}"
        )


if __name__ == "__main__":
    main()
