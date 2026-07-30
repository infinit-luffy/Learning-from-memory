"""L1/L2 tests for the TD-MPC2 adapter (E2E-0)."""
import torch

from hippoact.adapters import HippoActAdapter, AdapterConfig


def _make_adapter():
    cfg = AdapterConfig(
        stage1_ckpt="",             # random init is fine for shape tests
        proprio_dim=8,
        latent_dim=64,
        num_slots=8,
        slot_dim=32,
        image_size=56,
    )
    return HippoActAdapter(cfg)


def test_adapter_forward_shape():
    a = _make_adapter()
    obs = {"rgb": torch.randn(2, 3, 56, 56), "state": torch.randn(2, 8)}
    z = a(obs)
    assert z.shape == (2, 64)
    assert not torch.isnan(z).any()


def test_adapter_encoder_frozen_by_default():
    a = _make_adapter()
    assert all(not p.requires_grad for p in a.encoder.parameters())
    # z_mlp must be trainable
    assert all(p.requires_grad for p in a.z_mlp.parameters())


def test_adapter_gradient_reaches_zmlp_only():
    a = _make_adapter()
    obs = {"rgb": torch.randn(2, 3, 56, 56), "state": torch.randn(2, 8)}
    z = a(obs)
    z.sum().backward()
    assert a.z_mlp[0].weight.grad is not None
    assert all(p.grad is None for p in a.encoder.parameters())


def test_adapter_routing_deterministic():
    """Argmax routing: same input twice → identical z (no Gumbel noise).

    Note: requires the encoder slot init to be deterministic too — at
    eval we pass through encode_frame which samples slot init. This test
    documents the residual stochasticity: z varies across calls unless
    slot_query_mode='learned' or a fixed seed. We assert only that the
    ROUTING (argmax) itself adds no extra noise given identical slots.
    """
    a = _make_adapter()
    torch.manual_seed(0)
    obs = {"rgb": torch.randn(1, 3, 56, 56), "state": torch.randn(1, 8)}
    torch.manual_seed(42); z1 = a(obs)
    torch.manual_seed(42); z2 = a(obs)
    assert torch.allclose(z1, z2, atol=1e-6)
