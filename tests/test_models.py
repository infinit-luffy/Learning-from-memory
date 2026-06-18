import pytest
import torch

from sdam.models import DynamicEncoder, StaticEncoder


def make_obs(batch: int = 2, time: int = 5) -> torch.Tensor:
    return torch.rand(batch, time, 3, 32, 32)


def test_static_encoder_returns_static_latent_shape():
    encoder = StaticEncoder(in_channels=3, hidden_channels=8, static_dim=16)

    result = encoder(make_obs())

    assert result.shape == (2, 16)


def test_dynamic_encoder_returns_sequence_of_dynamic_latents():
    encoder = DynamicEncoder(in_channels=3, hidden_channels=8, dynamic_dim=12)

    result = encoder(make_obs())

    assert result.shape == (2, 4, 12)


def test_visual_encoders_reject_invalid_obs_rank():
    static_encoder = StaticEncoder(in_channels=3, hidden_channels=8, static_dim=16)
    dynamic_encoder = DynamicEncoder(in_channels=3, hidden_channels=8, dynamic_dim=12)

    with pytest.raises(ValueError, match="obs must have shape"):
        static_encoder(torch.rand(2, 3, 32, 32))

    with pytest.raises(ValueError, match="obs must have shape"):
        dynamic_encoder(torch.rand(2, 3, 32, 32))
