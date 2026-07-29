"""All auxiliary losses used by HippoAct.

Notation matches paper §III. Every loss is written to be safe when the
regularizer would otherwise be undefined (e.g. no previous frame yet).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- P1


def slot_reconstruction_loss(
    recon: torch.Tensor, target_features: torch.Tensor
) -> torch.Tensor:
    """DINOSAUR-style feature reconstruction. Target is detached."""
    return F.mse_loss(recon, target_features.detach())


def slow_temporal_loss(
    slots_t: torch.Tensor,
    slots_prev: torch.Tensor,
    slow_mask: torch.Tensor,
) -> torch.Tensor:
    """Encourage slots routed to 'slow' to be time-invariant.

    slots_t, slots_prev: (B, K, D)
    slow_mask:           (B, K)   1 where slot is slow.
    """
    diff = (slots_t - slots_prev.detach()).pow(2).sum(dim=-1)  # (B, K)
    return (slow_mask * diff).sum(dim=-1).mean()


def route_prior_kl(
    router_logits: torch.Tensor, prior_slow: float = 0.7
) -> torch.Tensor:
    """KL of mean routing distribution towards a prior [prior_slow, 1-prior_slow]."""
    probs = router_logits.softmax(dim=-1)
    # Mean over batch/time/slot — router_logits may be (B, K, 2) or (B, T, K, 2)
    mean_probs = probs.reshape(-1, 2).mean(dim=0)              # (2,)
    prior = torch.tensor([prior_slow, 1.0 - prior_slow], device=probs.device)
    return (mean_probs * (mean_probs.clamp(min=1e-8).log() - prior.log())).sum()


def slot_diversity_loss(slots: torch.Tensor) -> torch.Tensor:
    """Discourage slot collapse via off-diagonal pairwise cosine squared."""
    z = F.normalize(slots, dim=-1)                             # (B, K, D)
    cos = torch.matmul(z, z.transpose(-1, -2))                 # (B, K, K)
    K = slots.shape[1]
    off_diag = cos ** 2 - torch.eye(K, device=slots.device).unsqueeze(0)
    return off_diag.clamp(min=0.0).sum(dim=(-1, -2)).mean() / max(K * (K - 1), 1)


# --------------------------------------------------------------------- P2


def forward_proprio_loss(
    c_t: torch.Tensor,
    action_t: torch.Tensor,
    proprio_next: torch.Tensor,
    predictor: nn.Module,
) -> torch.Tensor:
    """||q_{t+1} − predictor([c_t, a_t])||²."""
    pred = predictor(torch.cat([c_t, action_t], dim=-1))
    return F.mse_loss(pred, proprio_next)


def action_align_infonce(
    c_t: torch.Tensor,
    actions: torch.Tensor,
    action_eps: float = 0.05,
    tau: float = 0.1,
) -> torch.Tensor:
    """InfoNCE where positive pairs share cosine-similar actions.

    c_t, actions: (B, *)
    """
    if c_t.size(0) < 2:
        return c_t.new_zeros(())
    z = F.normalize(c_t, dim=-1)
    a = F.normalize(actions, dim=-1)
    sim_c = torch.matmul(z, z.t()) / tau                         # (B, B)
    with torch.no_grad():
        sim_a = torch.matmul(a, a.t())
        pos = (sim_a > 1.0 - action_eps).float()
        pos.fill_diagonal_(0.0)
        n_pos = pos.sum(dim=-1, keepdim=True).clamp(min=1.0)
        pos = pos / n_pos
        has_any_pos = pos.sum(dim=-1) > 0                        # (B,)
    log_prob = sim_c - sim_c.logsumexp(dim=-1, keepdim=True)
    loss_per_row = -(pos * log_prob).sum(dim=-1)                 # (B,)
    if has_any_pos.any():
        return loss_per_row[has_any_pos].mean()
    return c_t.new_zeros(())


# --------------------------------------------------------------------- P3


def slot_swap_consistency(
    c_orig: torch.Tensor,
    c_swap: torch.Tensor,
    action_orig: torch.Tensor,
    action_swap: torch.Tensor,
) -> torch.Tensor:
    """||c_swap − sg(c_orig)||² + KL(pi_swap ‖ sg(pi_orig)).

    Actions may be continuous — we compare via MSE if unnormalized;
    for discrete-log-prob use, users should replace with F.kl_div externally.
    """
    l_c = F.mse_loss(c_swap, c_orig.detach())
    l_a = F.mse_loss(action_swap, action_orig.detach())
    return l_c + l_a
