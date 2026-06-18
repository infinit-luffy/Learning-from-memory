import pytest
import torch

from sdam.models import AssociativeMemory, DynamicEncoder, PositionVelocityHead, SDAMEncoder, StaticEncoder


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


def test_associative_memory_returns_context_latent_without_optional_inputs():
    memory = AssociativeMemory(dynamic_dim=12, static_dim=16, assoc_dim=20)

    c = memory(z_seq=torch.rand(2, 4, 12), b=torch.rand(2, 16))

    assert c.shape == (2, 20)


def test_associative_memory_accepts_q_and_actions():
    memory = AssociativeMemory(dynamic_dim=12, static_dim=16, assoc_dim=20, q_dim=3, action_dim=2)

    c = memory(
        z_seq=torch.rand(2, 4, 12),
        b=torch.rand(2, 16),
        q=torch.rand(2, 3),
        actions=torch.rand(2, 4, 2),
    )

    assert c.shape == (2, 20)


def test_associative_memory_rejects_wrong_dynamic_dim():
    memory = AssociativeMemory(dynamic_dim=12, static_dim=16, assoc_dim=20)

    with pytest.raises(ValueError, match="z_seq last dimension must be 12"):
        memory(z_seq=torch.rand(2, 4, 11), b=torch.rand(2, 16))


def test_associative_memory_rejects_empty_dynamic_sequence():
    memory = AssociativeMemory(dynamic_dim=12, static_dim=16, assoc_dim=20)

    with pytest.raises(ValueError, match="z_seq sequence length must be positive"):
        memory(z_seq=torch.rand(2, 0, 12), b=torch.rand(2, 16))


def test_associative_memory_rejects_missing_q_when_required():
    memory = AssociativeMemory(dynamic_dim=12, static_dim=16, assoc_dim=20, q_dim=3)

    with pytest.raises(ValueError, match="q must have shape"):
        memory(z_seq=torch.rand(2, 4, 12), b=torch.rand(2, 16))


def test_associative_memory_rejects_unexpected_actions_when_disabled():
    memory = AssociativeMemory(dynamic_dim=12, static_dim=16, assoc_dim=20, action_dim=0)

    with pytest.raises(ValueError, match="actions were provided but action_dim is 0"):
        memory(
            z_seq=torch.rand(2, 4, 12),
            b=torch.rand(2, 16),
            actions=torch.rand(2, 4, 2),
        )


def test_sdam_encoder_returns_structured_memory_dict():
    encoder = SDAMEncoder(
        in_channels=3,
        sequence_length=5,
        hidden_channels=8,
        static_dim=16,
        dynamic_dim=12,
        assoc_dim=20,
    )

    outputs = encoder(make_obs())

    assert set(outputs.keys()) == {"b", "z_seq", "c", "memory", "aux"}
    assert outputs["b"].shape == (2, 16)
    assert outputs["z_seq"].shape == (2, 4, 12)
    assert outputs["c"].shape == (2, 20)
    assert outputs["memory"].shape == (2, encoder.memory_dim)


def test_sdam_encoder_accepts_time_aligned_q_and_actions():
    encoder = SDAMEncoder(
        in_channels=3,
        sequence_length=5,
        hidden_channels=8,
        static_dim=16,
        dynamic_dim=12,
        assoc_dim=20,
        q_dim=3,
        action_dim=2,
    )

    outputs = encoder(make_obs(), q=torch.rand(2, 5, 3), actions=torch.rand(2, 4, 2))

    assert outputs["memory"].shape == (2, encoder.memory_dim)


def test_position_velocity_head_predicts_four_values():
    encoder = SDAMEncoder(in_channels=3, sequence_length=5, hidden_channels=8, static_dim=16, dynamic_dim=12, assoc_dim=20)
    head = PositionVelocityHead(memory_dim=encoder.memory_dim, hidden_dim=32)

    prediction = head(encoder(make_obs())["memory"])

    assert prediction["position"].shape == (2, 2)
    assert prediction["velocity"].shape == (2, 2)
