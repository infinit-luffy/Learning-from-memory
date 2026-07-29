"""L1 · Smoke tests. Run in <10 seconds."""
import torch


def test_import_package():
    import hippoact  # noqa: F401
    from hippoact import (
        HippoActEncoder, DinoV2Encoder, SlotAttention, SlotFeatureDecoder,
        SlotRouter, BindingTransformer, SafetyGate,  # noqa: F401
    )


def test_encoder_forward_no_nan(encoder, dummy_seq):
    imgs, proprio = dummy_seq
    out = encoder(imgs, proprio)
    assert out.c.shape == (2, 64)
    assert not torch.isnan(out.c).any(), "NaN in c_t"
    assert not torch.isinf(out.c).any(), "Inf in c_t"


def test_encoder_frame_forward_no_nan(encoder, dummy_frame):
    out = encoder.encode_frame(dummy_frame)
    assert out.slots.shape == (2, 8, 64)
    assert out.slot_recon.shape == (2, 16, encoder.dino.feat_dim)
    assert not torch.isnan(out.slots).any()


def test_encoder_backward_gradients_flow(encoder, dummy_seq):
    imgs, proprio = dummy_seq
    out = encoder(imgs, proprio)
    loss = out.c.pow(2).mean() + out.slot_recon.pow(2).mean()
    loss.backward()
    # Slot attn / router / binding should have gradients.
    grads = {n: p.grad is not None
             for n, p in encoder.named_parameters() if p.requires_grad}
    assert all(grads.values()), f"Missing gradients: {[k for k,v in grads.items() if not v]}"


def test_gumbel_router_output_is_hard(encoder):
    slots = torch.randn(4, 8, 64)
    g, logits = encoder.router(slots, hard=True)
    # One-hot: sum along last dim == 1, only 0 or 1 values.
    assert torch.allclose(g.sum(-1), torch.ones(4, 8))
    assert ((g == 0) | (g == 1)).all()
    assert logits.shape == (4, 8, 2)
