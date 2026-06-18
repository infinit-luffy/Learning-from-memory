from dataclasses import replace
from pathlib import Path

import pytest
import torch

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


def test_train_one_step_advances_batches():
    config = load_config(Path("configs/synthetic/sdam.yaml"))
    components = build_synthetic_components(config)

    train_one_step(components)
    first_target = components.last_batch_target_position.clone()
    train_one_step(components)
    second_target = components.last_batch_target_position.clone()

    assert not torch.allclose(first_target, second_target)


def test_train_one_step_updates_trainable_parameter():
    config = load_config(Path("configs/synthetic/sdam.yaml"))
    components = build_synthetic_components(config)
    before = [parameter.detach().clone() for parameter in _trainable_parameters(components)]

    train_one_step(components)

    after = _trainable_parameters(components)
    assert any(not torch.allclose(before_param, after_param) for before_param, after_param in zip(before, after))


def test_evaluate_one_batch_reports_memory_dim_and_losses():
    config = load_config(Path("configs/synthetic/sdam.yaml"))
    components = build_synthetic_components(config)

    metrics = evaluate_one_batch(components)

    assert metrics["memory_dim"] > 0
    assert metrics["position_mse"] >= 0.0
    assert metrics["velocity_mse"] >= 0.0


def test_evaluate_one_batch_uses_eval_mode_without_grads():
    config = load_config(Path("configs/synthetic/sdam.yaml"))
    components = build_synthetic_components(config)
    components.encoder.train()
    components.head.train()
    for parameter in _trainable_parameters(components):
        parameter.grad = None

    evaluate_one_batch(components)

    assert components.encoder.training is False
    assert components.head.training is False
    assert all(parameter.grad is None for parameter in _trainable_parameters(components))


@pytest.mark.parametrize("model_override", [{"q_dim": 1}, {"action_dim": 1}])
def test_build_synthetic_components_rejects_q_or_action_dims(model_override):
    config = load_config(Path("configs/synthetic/sdam.yaml"))
    config = replace(config, model=replace(config.model, **model_override))

    with pytest.raises(ValueError, match="synthetic experiment does not provide q or actions"):
        build_synthetic_components(config)


def test_latent_flow_matching_loss_is_future_extension_interface():
    with pytest.raises(NotImplementedError, match="Latent flow association is not implemented"):
        latent_flow_matching_loss()


def _trainable_parameters(components):
    return [
        parameter
        for module in (components.encoder, components.head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
