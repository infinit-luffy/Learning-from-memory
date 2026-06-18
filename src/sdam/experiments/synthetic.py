from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import DataLoader

from sdam.config import SDAMConfig
from sdam.data import SyntheticVideoConfig, SyntheticVideoDataset
from sdam.losses import position_velocity_loss
from sdam.models import PositionVelocityHead, SDAMEncoder


@dataclass
class SyntheticComponents:
    config: SDAMConfig
    loader: DataLoader
    encoder: SDAMEncoder
    head: PositionVelocityHead
    optimizer: torch.optim.Optimizer
    loader_iter: Iterator[dict[str, torch.Tensor]] | None = None
    last_batch_target_position: torch.Tensor | None = None


def build_synthetic_components(config: SDAMConfig, seed: int = 0) -> SyntheticComponents:
    if config.model.q_dim > 0 or config.model.action_dim > 0:
        raise ValueError("synthetic experiment does not provide q or actions")

    dataset_config = SyntheticVideoConfig(
        image_size=config.data.image_size,
        channels=config.data.channels,
        sequence_length=config.data.sequence_length,
        dataset_size=config.data.dataset_size,
        object_size=config.data.object_size,
        clutter_count=config.data.clutter_count,
        min_speed=config.data.min_speed,
        max_speed=config.data.max_speed,
    )
    dataset = SyntheticVideoDataset(dataset_config, seed=seed)
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False)
    encoder = SDAMEncoder(
        in_channels=config.data.channels,
        sequence_length=config.data.sequence_length,
        hidden_channels=config.model.hidden_channels,
        static_dim=config.model.static_dim,
        dynamic_dim=config.model.dynamic_dim,
        assoc_dim=config.model.assoc_dim,
        q_dim=config.model.q_dim,
        action_dim=config.model.action_dim,
    )
    head = PositionVelocityHead(memory_dim=encoder.memory_dim)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=config.training.learning_rate,
    )
    return SyntheticComponents(config=config, loader=loader, encoder=encoder, head=head, optimizer=optimizer)


def _next_batch(components: SyntheticComponents) -> dict[str, torch.Tensor]:
    if components.loader_iter is None:
        components.loader_iter = iter(components.loader)

    try:
        return next(components.loader_iter)
    except StopIteration:
        components.loader_iter = iter(components.loader)
        return next(components.loader_iter)


def train_one_step(components: SyntheticComponents) -> dict[str, float]:
    components.encoder.train()
    components.head.train()
    batch = _next_batch(components)
    components.last_batch_target_position = batch["target_position"].detach().clone()
    components.optimizer.zero_grad()
    outputs = components.encoder(batch["obs"])
    predictions = components.head(outputs["memory"])
    loss, metrics = position_velocity_loss(
        predictions,
        batch,
        velocity_weight=components.config.training.velocity_loss_weight,
    )
    loss.backward()
    components.optimizer.step()
    return metrics


@torch.no_grad()
def evaluate_one_batch(components: SyntheticComponents) -> dict[str, float]:
    components.encoder.eval()
    components.head.eval()
    batch = next(iter(components.loader))
    outputs = components.encoder(batch["obs"])
    predictions = components.head(outputs["memory"])
    _, metrics = position_velocity_loss(
        predictions,
        batch,
        velocity_weight=components.config.training.velocity_loss_weight,
    )
    return {
        "position_mse": metrics["position_loss"],
        "velocity_mse": metrics["velocity_loss"],
        "memory_dim": float(components.encoder.memory_dim),
    }
