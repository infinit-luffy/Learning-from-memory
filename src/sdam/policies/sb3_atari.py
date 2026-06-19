from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def atari_observations_to_sdam(
    observations: torch.Tensor, sequence_length: int
) -> torch.Tensor:
    if observations.ndim != 4:
        raise ValueError("observations must have shape [B, T, H, W]")
    if observations.shape[1] != sequence_length:
        raise ValueError(f"observations frame stack must be {sequence_length}")

    converted = observations.float()
    if converted.numel() > 0 and converted.max().item() > 1.0:
        converted = converted / 255.0

    return converted.unsqueeze(2)
