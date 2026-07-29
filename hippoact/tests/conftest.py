"""Test-wide fixtures.

We force ``HIPPOACT_FORCE_MOCK=1`` so no test tries to download DINOv2 weights.
"""
import os
os.environ.setdefault("HIPPOACT_FORCE_MOCK", "1")

import pytest
import torch

from hippoact.encoders.hippo_encoder import HippoActEncoder


@pytest.fixture(scope="session")
def device() -> str:
    return "cpu"       # tests are CPU-only; production runs on CUDA.


@pytest.fixture(scope="function")
def encoder(device) -> HippoActEncoder:
    torch.manual_seed(0)
    return HippoActEncoder(
        num_slots=8,           # small for speed
        slot_dim=64,
        proprio_dim=8,
        c_dim=64,
        t_window=2,
        image_size=56,         # 56 / 14 = 4 → 16 patch tokens
        patch_size=14,
        binding_layers=2,
        binding_heads=2,
        slot_iters=2,
        slot_hidden=64,
        router_hidden=16,
    ).to(device)


@pytest.fixture(scope="function")
def dummy_frame(device) -> torch.Tensor:
    return torch.randn(2, 3, 56, 56, device=device)


@pytest.fixture(scope="function")
def dummy_seq(device) -> tuple:
    imgs = torch.randn(2, 2, 3, 56, 56, device=device)
    proprio = torch.randn(2, 2, 8, device=device)
    return imgs, proprio
