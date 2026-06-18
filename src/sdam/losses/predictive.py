from __future__ import annotations

import torch
import torch.nn.functional as F


def position_velocity_loss(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    velocity_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    position_loss = F.mse_loss(predictions["position"], batch["target_position"])
    velocity_loss = F.mse_loss(predictions["velocity"], batch["target_velocity"])
    loss = position_loss + velocity_weight * velocity_loss
    metrics = {
        "loss": float(loss.detach().item()),
        "position_loss": float(position_loss.detach().item()),
        "velocity_loss": float(velocity_loss.detach().item()),
    }
    return loss, metrics
