from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdam.config import load_config
from sdam.experiments import build_synthetic_components, evaluate_one_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic/sdam.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    components = build_synthetic_components(config)
    metrics = evaluate_one_batch(components)
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")


if __name__ == "__main__":
    main()
