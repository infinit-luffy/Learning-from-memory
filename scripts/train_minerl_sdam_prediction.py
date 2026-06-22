from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MineRL SDAM-3D scene prediction from preprocessed offline shards."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/minerl/navigate_sdam_prediction.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/minerl/navigate_sdam_prediction"))
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from sdam.config import load_minerl_prediction_config
    from sdam.experiments.minerl_prediction import train_minerl_prediction

    config = load_minerl_prediction_config(args.config)
    result = train_minerl_prediction(
        config=config,
        output_dir=args.output_dir,
        train_steps=args.train_steps,
        device=args.device,
        log_interval=args.log_interval,
    )
    train_metrics = result["train_metrics"]
    eval_metrics = result["eval_metrics"]
    print(
        "minerl_sdam_prediction "
        f"checkpoint={result['checkpoint_path']} "
        f"device={result['device']} "
        f"train_loss={train_metrics['total_loss']:.6f} "
        f"eval_loss={eval_metrics['total_loss']:.6f}"
    )


if __name__ == "__main__":
    main()
