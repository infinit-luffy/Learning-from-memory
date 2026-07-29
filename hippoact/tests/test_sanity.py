"""L3 · Sanity tests — verify the method actually learns what we claim.

These are slow-ish (~30s each on CPU with tiny config). Run manually or
with ``pytest -m sanity`` when you touch the encoder.
"""
import pytest
import torch
import torch.nn.functional as F

from hippoact.augmentation.slot_swap import slot_swap_in_batch
from hippoact.losses import (
    action_align_infonce,
    route_prior_kl,
    slot_diversity_loss,
    slot_reconstruction_loss,
    slow_temporal_loss,
)


pytestmark = pytest.mark.sanity


def test_slot_reconstruction_loss_reduces_on_overfit(encoder):
    """S1 (mini): 200 steps should cut L_slot by ≥ 5×."""
    torch.manual_seed(0)
    img = torch.randn(1, 3, 56, 56)
    optim = torch.optim.AdamW(
        list(encoder.slot_attn.parameters())
        + list(encoder.slot_decoder.parameters()),
        lr=3e-4,
    )
    losses = []
    for _ in range(200):
        with torch.no_grad():
            target = encoder.dino(img)
        slots = encoder.slot_attn(target)
        recon, _ = encoder.slot_decoder(slots)
        loss = slot_reconstruction_loss(recon, target)
        optim.zero_grad(); loss.backward(); optim.step()
        losses.append(loss.item())
    assert losses[0] > 5 * losses[-1], (
        f"Slot loss did not decrease enough: {losses[0]:.4f} → {losses[-1]:.4f}"
    )


def test_route_prior_pushes_towards_prior(encoder):
    """L_route should decrease when routing becomes closer to prior [0.7, 0.3]."""
    slots = torch.randn(16, 8, 64)
    # Force routing towards 50/50 (uniform)
    _, logits_uniform = encoder.router(slots)
    l_uniform = route_prior_kl(logits_uniform, prior_slow=0.7)
    # Force routing towards our prior directly
    logits_close = torch.tensor([[0.7, 0.3]]).log().expand(16, 8, 2)
    l_close = route_prior_kl(logits_close, prior_slow=0.7)
    assert l_close < l_uniform


def test_slot_diversity_penalizes_collapse():
    identical = torch.randn(1, 8, 64).expand(1, 8, 64).clone()
    diverse = torch.randn(1, 8, 64) * 3.0
    l_bad = slot_diversity_loss(identical)
    l_good = slot_diversity_loss(diverse)
    assert l_bad > l_good


def test_slow_loss_rewards_router_matching_variance_target():
    """Router that classifies low-variance → slow and high-variance → fast
    should score lower than a router that flips those labels."""
    torch.manual_seed(0)
    # 8 slots; first 4 static (variance 0), last 4 wildly moving.
    slots_prev = torch.randn(4, 8, 16)
    slots_t = slots_prev.clone()
    slots_t[:, 4:] += 5.0                      # high variance on the last 4

    # 'Good' router: first 4 slow, last 4 fast.
    good = torch.tensor([[10.0, -10.0]] * 4 + [[-10.0, 10.0]] * 4)
    good = good.unsqueeze(0).expand(4, -1, -1).contiguous()
    # 'Bad' router: opposite.
    bad = torch.tensor([[-10.0, 10.0]] * 4 + [[10.0, -10.0]] * 4)
    bad = bad.unsqueeze(0).expand(4, -1, -1).contiguous()

    l_good = slow_temporal_loss(slots_t, slots_prev, good, prior_slow=0.5)
    l_bad  = slow_temporal_loss(slots_t, slots_prev, bad,  prior_slow=0.5)
    assert l_good.item() < l_bad.item(), f"good {l_good} !< bad {l_bad}"


def test_slow_loss_has_no_trivial_all_fast_minimum():
    """The regression fix: routing everything to fast should NOT minimize
    the loss when slots actually vary in temporal frequency."""
    torch.manual_seed(0)
    slots_prev = torch.zeros(2, 8, 16)
    slots_t = torch.zeros(2, 8, 16)
    slots_t[:, 4:] += 1.0                      # last 4 move, first 4 don't

    all_fast = torch.tensor([[-10.0, 10.0]]).expand(2, 8, 2).contiguous()
    prior_match = torch.zeros(2, 8, 2)         # neutral logits → prior-only

    l_fast = slow_temporal_loss(slots_t, slots_prev, all_fast, prior_slow=0.5)
    l_neutral = slow_temporal_loss(slots_t, slots_prev, prior_match, prior_slow=0.5)
    # All-fast should be *worse* than neutral because it mis-labels 4 slots.
    assert l_fast > l_neutral


def test_infonce_returns_zero_without_positive_pairs():
    c = torch.randn(8, 32)
    # actions all orthogonal → no positive pairs
    actions = torch.eye(8)
    l = action_align_infonce(c, actions, action_eps=0.001)
    assert l.item() == 0.0


def test_slot_swap_only_replaces_slow_positions():
    torch.manual_seed(0)
    slots = torch.randn(4, 8, 16)
    fast_mask = torch.zeros(4, 8)
    fast_mask[:, :4] = 1.0                 # first half fast, second half slow
    perm = torch.tensor([1, 0, 3, 2])
    swapped, indicator = slot_swap_in_batch(slots, fast_mask, perm=perm)

    # Fast positions must be untouched.
    assert torch.allclose(swapped[:, :4], slots[:, :4])
    # Slow positions should reflect the permutation.
    for b in range(4):
        assert torch.allclose(swapped[b, 4:], slots[perm[b], 4:])
    assert indicator[:, :4].sum().item() == 0
    assert indicator[:, 4:].sum().item() == 16
