from __future__ import annotations

import torch
from torch import nn

from sdam.models import SDAMEncoder


_SB3_IMPORT_ERROR = (
    'Stable-Baselines3 is required for Atari experiments. Install with: pip install -e ".[atari]"'
)


def atari_observations_to_sdam(
    observations: torch.Tensor, sequence_length: int
) -> torch.Tensor:
    if observations.ndim != 4:
        raise ValueError("observations must have shape [B, T, H, W]")
    if observations.shape[1] != sequence_length:
        raise ValueError(f"observations frame stack must be {sequence_length}")

    converted = observations.float()
    if not observations.dtype.is_floating_point:
        converted = converted / 255.0
    elif converted.numel() > 0 and converted.max().item() > 1.0:
        converted = converted / 255.0

    return converted.unsqueeze(2)


def _load_base_features_extractor():
    try:
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    except ImportError as exc:
        raise ImportError(_SB3_IMPORT_ERROR) from exc

    return BaseFeaturesExtractor


def _attach_loaded_base(instance: nn.Module, base_features_extractor: type[nn.Module]) -> None:
    if isinstance(instance, base_features_extractor):
        return

    current_class = instance.__class__
    optional_class = type(
        f"{current_class.__name__}With{base_features_extractor.__name__}",
        (current_class, base_features_extractor),
        {"__module__": current_class.__module__},
    )
    instance.__class__ = optional_class


class _OptionalBaseFeaturesExtractor(nn.Module):
    def __init__(self, observation_space, features_dim: int) -> None:
        base_features_extractor = _load_base_features_extractor()
        _attach_loaded_base(self, base_features_extractor)
        base_features_extractor.__init__(self, observation_space, features_dim)


class SDAMAtariFeaturesExtractor(_OptionalBaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        static_dim: int,
        dynamic_dim: int,
        assoc_dim: int,
        hidden_channels: int,
        features_dim: int,
        sequence_length: int,
    ) -> None:
        super().__init__(observation_space, features_dim)

        expected_shape = (sequence_length, 84, 84)
        if tuple(observation_space.shape) != expected_shape:
            raise ValueError(f"observation_space shape must be ({sequence_length}, 84, 84)")

        self.sequence_length = sequence_length
        self.encoder = SDAMEncoder(
            in_channels=1,
            sequence_length=sequence_length,
            hidden_channels=hidden_channels,
            static_dim=static_dim,
            dynamic_dim=dynamic_dim,
            assoc_dim=assoc_dim,
        )
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.memory_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        sdam_observations = atari_observations_to_sdam(observations, self.sequence_length)
        outputs = self.encoder(sdam_observations)
        return self.projector(outputs["memory"])
