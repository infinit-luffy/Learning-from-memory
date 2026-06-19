from __future__ import annotations

import torch
import torch.nn.functional as F
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


class _FrameDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels * 21 * 21),
            nn.ReLU(),
            nn.Unflatten(1, (hidden_channels, 21, 21)),
            nn.ConvTranspose2d(hidden_channels, hidden_channels // 2 or 1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels // 2 or 1, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.net(latents)


class SDAMAtariAutoEncoder(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        static_dim: int,
        dynamic_dim: int,
        assoc_dim: int,
        hidden_channels: int,
        reconstruction_weight: float = 1.0,
        prediction_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if reconstruction_weight < 0:
            raise ValueError("reconstruction_weight must be non-negative")
        if prediction_weight < 0:
            raise ValueError("prediction_weight must be non-negative")

        self.sequence_length = sequence_length
        self.reconstruction_weight = reconstruction_weight
        self.prediction_weight = prediction_weight
        self.encoder = SDAMEncoder(
            in_channels=1,
            sequence_length=sequence_length,
            hidden_channels=hidden_channels,
            static_dim=static_dim,
            dynamic_dim=dynamic_dim,
            assoc_dim=assoc_dim,
        )
        decoder_input_dim = static_dim + dynamic_dim
        self.reconstruction_head = _FrameDecoder(decoder_input_dim, hidden_channels)
        self.prediction_head = _FrameDecoder(decoder_input_dim, hidden_channels)

    def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        if observations.ndim != 5:
            raise ValueError("observations must have shape [B, T, C, H, W]")
        if observations.shape[1] != self.sequence_length:
            raise ValueError(f"observations time dimension must be {self.sequence_length}")
        if observations.shape[2:] != (1, 84, 84):
            raise ValueError("observations frame shape must be [1, 84, 84]")

        observations = observations.float()
        if observations.numel() > 0 and observations.max().item() > 1.0:
            observations = observations / 255.0

        encoded = self.encoder(observations)
        b = encoded["b"]
        z_seq = encoded["z_seq"]
        batch = observations.shape[0]
        zero_dynamic = z_seq.new_zeros(batch, 1, z_seq.shape[-1])
        reconstruction_z = torch.cat([zero_dynamic, z_seq], dim=1)
        repeated_b = b.unsqueeze(1).expand(-1, self.sequence_length, -1)
        reconstruction_latents = torch.cat([repeated_b, reconstruction_z], dim=-1)
        reconstruction = self.reconstruction_head(
            reconstruction_latents.reshape(batch * self.sequence_length, -1)
        ).reshape(batch, self.sequence_length, 1, 84, 84)

        prediction_latents = torch.cat([repeated_b[:, :-1], z_seq], dim=-1)
        prediction = self.prediction_head(
            prediction_latents.reshape(batch * (self.sequence_length - 1), -1)
        ).reshape(batch, self.sequence_length - 1, 1, 84, 84)

        reconstruction_loss = F.mse_loss(reconstruction, observations)
        prediction_loss = F.mse_loss(prediction, observations[:, 1:])
        loss = (
            self.reconstruction_weight * reconstruction_loss
            + self.prediction_weight * prediction_loss
        )
        return {
            "reconstruction": reconstruction,
            "prediction": prediction,
            "reconstruction_loss": reconstruction_loss,
            "prediction_loss": prediction_loss,
            "loss": loss,
            "b": b,
            "z_seq": z_seq,
        }
