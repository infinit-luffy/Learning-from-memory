from dataclasses import replace
from pathlib import Path

import pytest
import torch

import sdam.experiments.synthetic as synthetic_experiment
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
    assert metrics["forward_latency_ms"] >= 0.0


def test_evaluate_one_batch_forward_latency_excludes_loss_time(monkeypatch):
    config = load_config(Path("configs/synthetic/sdam.yaml"))
    components = build_synthetic_components(config)
    original_loss = synthetic_experiment.position_velocity_loss
    ticks = iter([10.0, 10.25, 99.0])

    monkeypatch.setattr(synthetic_experiment.time, "perf_counter", lambda: next(ticks))

    def loss_with_observable_timer(*args, **kwargs):
        synthetic_experiment.time.perf_counter()
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(synthetic_experiment, "position_velocity_loss", loss_with_observable_timer)

    metrics = evaluate_one_batch(components)

    assert metrics["forward_latency_ms"] == pytest.approx(250.0)


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


import subprocess
import sys


def test_eval_script_runs_on_default_config():
    result = subprocess.run(
        [sys.executable, "scripts/eval_synthetic.py", "--config", "configs/synthetic/sdam.yaml"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "position_mse" in result.stdout
    assert "memory_dim" in result.stdout
    assert "forward_latency_ms" in result.stdout


def test_train_script_runs_for_one_step():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_synthetic.py",
            "--config",
            "configs/synthetic/sdam.yaml",
            "--steps",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "step=1" in result.stdout
    assert "loss=" in result.stdout
