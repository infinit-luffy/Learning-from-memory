"""L2 · Module tests — shapes, frozen state, mask correctness."""
import torch

from hippoact.encoders.binding import BindingTransformer
from hippoact.encoders.dinov2 import DinoV2Encoder
from hippoact.encoders.slot_attention import SlotAttention
from hippoact.encoders.slot_decoder import SlotFeatureDecoder


def test_dinov2_all_params_frozen():
    d = DinoV2Encoder(image_size=56, patch_size=14)
    for p in d._backbone.parameters():
        assert not p.requires_grad


def test_slot_attention_output_shape():
    sa = SlotAttention(num_slots=8, slot_dim=64, input_dim=192)
    x = torch.randn(3, 25, 192)
    out = sa(x)
    assert out.shape == (3, 8, 64)


def test_slot_decoder_shapes_and_alpha_normalized():
    dec = SlotFeatureDecoder(slot_dim=64, out_dim=192, num_patches=25)
    slots = torch.randn(3, 8, 64)
    recon, alpha = dec(slots)
    assert recon.shape == (3, 25, 192)
    assert alpha.shape == (3, 8, 25)
    # alpha softmax over slot dim → sums to 1 per (batch, patch).
    assert torch.allclose(alpha.sum(dim=1), torch.ones(3, 25), atol=1e-5)


def test_binding_mask_effectively_hides_slow_slots():
    """P2 sanity: the key-padding mask must actually exclude slow slot content.

    We construct two batches identical in fast slots + proprio, but with
    completely different values at slow slot positions. Outputs must match.
    """
    torch.manual_seed(0)
    bt = BindingTransformer(
        slot_dim=64, proprio_dim=8, d_model=64,
        n_layers=2, n_heads=2, t_window=2, num_slots=8, dropout=0.0,
    )
    bt.eval()

    fast_seq = torch.randn(2, 2, 8, 64)
    proprio = torch.randn(2, 2, 8)
    fast_mask = torch.ones(2, 2, 8)
    fast_mask[:, :, 4:] = 0.0                        # last 4 slots are slow

    fast_seq_alt = fast_seq.clone()
    fast_seq_alt[:, :, 4:] = 999.0                   # trash the slow positions

    c1 = bt(fast_seq, proprio, fast_mask)
    c2 = bt(fast_seq_alt, proprio, fast_mask)

    assert torch.allclose(c1, c2, atol=1e-5), (
        "Mask leaked slow slot content into c_t — P3 slot-swap will be unsound"
    )


def test_binding_output_shape():
    bt = BindingTransformer(
        slot_dim=64, proprio_dim=8, d_model=64,
        n_layers=2, n_heads=2, t_window=4, num_slots=8, dropout=0.0,
    )
    fast_seq = torch.randn(3, 4, 8, 64)
    proprio = torch.randn(3, 4, 8)
    fast_mask = torch.ones(3, 4, 8)
    c = bt(fast_seq, proprio, fast_mask)
    assert c.shape == (3, 64)
