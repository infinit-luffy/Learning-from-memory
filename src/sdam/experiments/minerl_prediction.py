from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader

from sdam.config import MineRLPredictionConfig
from sdam.data import MineRLSequenceDataset
from sdam.models import SDAM3DScenePredictor, sdam_3d_prediction_loss


@dataclass
class MineRLPredictionComponents:
    config: MineRLPredictionConfig
    loader: DataLoader
    model: SDAM3DScenePredictor
    optimizer: torch.optim.Optimizer
    device: torch.device
    loader_iter: Iterator[dict[str, torch.Tensor]] | None = None


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_minerl_prediction_components(
    config: MineRLPredictionConfig,
    device: str | None = None,
) -> MineRLPredictionComponents:
    resolved_device = resolve_device(device or config.training.device)
    dataset = MineRLSequenceDataset(
        dataset_path=config.data.dataset_path,
        sequence_length=config.data.sequence_length,
        image_size=config.data.image_size,
        action_dim=config.data.action_dim,
        change_threshold=config.data.change_threshold,
    )
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True)
    model = SDAM3DScenePredictor(
        channels=config.data.channels,
        image_size=config.data.image_size,
        action_dim=config.data.action_dim,
        static_dim=config.model.static_dim,
        dynamic_dim=config.model.dynamic_dim,
        assoc_dim=config.model.assoc_dim,
        hidden_channels=config.model.hidden_channels,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    return MineRLPredictionComponents(
        config=config,
        loader=loader,
        model=model,
        optimizer=optimizer,
        device=resolved_device,
    )


def train_one_step(components: MineRLPredictionComponents) -> dict[str, float]:
    components.model.train()
    batch = _next_batch(components)
    batch = _move_batch(batch, components.device)
    outputs = components.model(batch["obs"], batch["actions"], next_obs=batch["next_obs"])
    loss, metrics = sdam_3d_prediction_loss(
        outputs,
        batch,
        frame_weight=components.config.training.frame_loss_weight,
        latent_weight=components.config.training.latent_loss_weight,
        change_weight=components.config.training.change_loss_weight,
        recon_weight=components.config.training.recon_loss_weight,
    )
    components.optimizer.zero_grad()
    loss.backward()
    components.optimizer.step()
    return metrics


@torch.no_grad()
def evaluate_one_batch(components: MineRLPredictionComponents) -> dict[str, float]:
    components.model.eval()
    batch = _move_batch(next(iter(components.loader)), components.device)
    start = time.perf_counter()
    outputs = components.model(batch["obs"], batch["actions"], next_obs=batch["next_obs"])
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _, metrics = sdam_3d_prediction_loss(
        outputs,
        batch,
        frame_weight=components.config.training.frame_loss_weight,
        latent_weight=components.config.training.latent_loss_weight,
        change_weight=components.config.training.change_loss_weight,
        recon_weight=components.config.training.recon_loss_weight,
    )
    metrics["forward_latency_ms"] = elapsed_ms
    return metrics


def train_minerl_prediction(
    config: MineRLPredictionConfig,
    output_dir: str | Path,
    train_steps: int | None = None,
    device: str | None = None,
    log_interval: int = 100,
    logger=print,
) -> dict[str, object]:
    components = build_minerl_prediction_components(config, device=device)
    steps = train_steps or config.training.train_steps
    last_metrics: dict[str, float] = {}
    for step in range(1, steps + 1):
        last_metrics = train_one_step(components)
        if logger is not None and log_interval > 0 and (step == 1 or step % log_interval == 0):
            logger(
                "[minerl train] "
                f"step={step} loss={last_metrics['total_loss']:.6f} "
                f"frame={last_metrics['frame_loss']:.6f} "
                f"latent={last_metrics['latent_loss']:.6f} "
                f"change={last_metrics['change_loss']:.6f}"
            )

    eval_metrics = evaluate_one_batch(components)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "sdam_3d_prediction.pt"
    torch.save(
        {
            "model_state_dict": components.model.state_dict(),
            "config": config,
            "train_metrics": last_metrics,
            "eval_metrics": eval_metrics,
        },
        checkpoint_path,
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "train_steps": steps,
        "device": str(components.device),
        "train_metrics": last_metrics,
        "eval_metrics": eval_metrics,
    }


def _next_batch(components: MineRLPredictionComponents) -> dict[str, torch.Tensor]:
    if components.loader_iter is None:
        components.loader_iter = iter(components.loader)
    try:
        return next(components.loader_iter)
    except StopIteration:
        components.loader_iter = iter(components.loader)
        return next(components.loader_iter)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}
