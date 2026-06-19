import builtins
import importlib
import sys
import types

import pytest
import torch
from torch import nn

from sdam.policies import atari_observations_to_sdam


class FakeBox:
    def __init__(self, shape):
        self.shape = shape


@pytest.fixture()
def fake_sb3(monkeypatch):
    stable_baselines3 = types.ModuleType("stable_baselines3")
    common = types.ModuleType("stable_baselines3.common")
    torch_layers = types.ModuleType("stable_baselines3.common.torch_layers")

    class BaseFeaturesExtractor(nn.Module):
        def __init__(self, observation_space, features_dim):
            super().__init__()
            self.observation_space = observation_space
            self.features_dim = features_dim

    torch_layers.BaseFeaturesExtractor = BaseFeaturesExtractor
    common.torch_layers = torch_layers
    stable_baselines3.common = common

    monkeypatch.setitem(sys.modules, "stable_baselines3", stable_baselines3)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", common)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.torch_layers", torch_layers)

    import sdam.policies.sb3_atari as sb3_atari

    return importlib.reload(sb3_atari)


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


def test_sdam_atari_features_extractor_returns_features(fake_sb3):
    extractor = fake_sb3.SDAMAtariFeaturesExtractor(
        FakeBox((4, 84, 84)),
        static_dim=8,
        dynamic_dim=8,
        assoc_dim=16,
        hidden_channels=4,
        features_dim=32,
        sequence_length=4,
    )

    features = extractor(torch.rand(2, 4, 84, 84))

    assert features.shape == (2, 32)
    assert extractor.features_dim == 32


def test_sdam_atari_features_extractor_requires_matching_observation_space(fake_sb3):
    with pytest.raises(ValueError, match="observation_space shape must be"):
        fake_sb3.SDAMAtariFeaturesExtractor(
            FakeBox((3, 84, 84)),
            static_dim=8,
            dynamic_dim=8,
            assoc_dim=16,
            hidden_channels=4,
            features_dim=32,
            sequence_length=4,
        )


def test_sdam_atari_policy_imports_without_sb3(monkeypatch):
    monkeypatch.delitem(sys.modules, "stable_baselines3", raising=False)
    monkeypatch.delitem(sys.modules, "stable_baselines3.common", raising=False)
    monkeypatch.delitem(sys.modules, "stable_baselines3.common.torch_layers", raising=False)
    real_import = builtins.__import__

    def import_without_sb3(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "stable_baselines3" or name.startswith("stable_baselines3."):
            raise ModuleNotFoundError("No module named 'stable_baselines3'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_sb3)

    import sdam.policies.sb3_atari as sb3_atari

    reloaded = importlib.reload(sb3_atari)

    with pytest.raises(ImportError, match="Stable-Baselines3 is required"):
        reloaded.SDAMAtariFeaturesExtractor(
            FakeBox((4, 84, 84)),
            static_dim=8,
            dynamic_dim=8,
            assoc_dim=16,
            hidden_channels=4,
            features_dim=32,
            sequence_length=4,
        )
