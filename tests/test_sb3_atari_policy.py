import pytest

from sdam.policies import atari_observations_to_sdam

import torch


def test_uint_like_observations_are_scaled_and_channelled():
    observations = torch.full((2, 4, 84, 84), 255, dtype=torch.uint8)

    converted = atari_observations_to_sdam(observations, sequence_length=4)

    assert converted.shape == (2, 4, 1, 84, 84)
    assert converted.dtype == torch.float32
    assert torch.all(converted == 1.0)


def test_low_valued_uint_observations_are_scaled():
    observations = torch.ones((1, 4, 84, 84), dtype=torch.uint8)

    converted = atari_observations_to_sdam(observations, sequence_length=4)

    assert converted.shape == (1, 4, 1, 84, 84)
    assert converted.dtype == torch.float32
    assert torch.all(converted == 1.0 / 255.0)


def test_already_normalized_observations_are_not_scaled():
    observations = torch.full((2, 4, 84, 84), 0.5)

    converted = atari_observations_to_sdam(observations, sequence_length=4)

    assert torch.all(converted == 0.5)


def test_invalid_rank_raises_clear_error():
    observations = torch.zeros((2, 4, 84))

    with pytest.raises(ValueError, match="observations must have shape"):
        atari_observations_to_sdam(observations, sequence_length=4)


def test_wrong_frame_stack_raises_clear_error():
    observations = torch.zeros((2, 3, 84, 84))

    with pytest.raises(ValueError, match="observations frame stack must be 4"):
        atari_observations_to_sdam(observations, sequence_length=4)
