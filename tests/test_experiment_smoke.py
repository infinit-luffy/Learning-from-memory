from pathlib import Path

import pytest

from sdam.config import load_config
from sdam.experiments.synthetic import build_synthetic_components, evaluate_one_batch, train_one_step
from sdam.losses.flow_matching import latent_flow_matching_loss


def test_build_synthetic_components_and_train_one_step():
    config = load_config(Path("configs/synthetic/sdam.yaml"))
    components = build_synthetic_components(config)

    metrics = train_one_step(components)

    assert metrics["loss"] >= 0.0
    assert metrics["position_loss"] >= 0.0
    assert metrics["velocity_loss"] >= 0.0


def test_evaluate_one_batch_reports_memory_dim_and_losses():
    config = load_config(Path("configs/synthetic/sdam.yaml"))
    components = build_synthetic_components(config)

    metrics = evaluate_one_batch(components)

    assert metrics["memory_dim"] > 0
    assert metrics["position_mse"] >= 0.0
    assert metrics["velocity_mse"] >= 0.0


def test_latent_flow_matching_loss_is_future_extension_interface():
    with pytest.raises(NotImplementedError, match="Latent flow association is not implemented"):
        latent_flow_matching_loss()
