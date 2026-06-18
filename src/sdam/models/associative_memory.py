from __future__ import annotations

import torch
from torch import nn


class AssociativeMemory(nn.Module):
    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        assoc_dim: int,
        q_dim: int = 0,
        action_dim: int = 0,
    ) -> None:
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.assoc_dim = assoc_dim
        self.q_dim = q_dim
        self.action_dim = action_dim
        self.gru = nn.GRU(input_size=dynamic_dim, hidden_size=assoc_dim, batch_first=True)
        fusion_dim = assoc_dim + static_dim + q_dim + action_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, assoc_dim),
            nn.ReLU(),
            nn.Linear(assoc_dim, assoc_dim),
        )

    def forward(
        self,
        z_seq: torch.Tensor,
        b: torch.Tensor,
        q: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_seq.ndim != 3:
            raise ValueError("z_seq must have shape [B, K, dynamic_dim]")
        if z_seq.shape[-1] != self.dynamic_dim:
            raise ValueError(f"z_seq last dimension must be {self.dynamic_dim}")
        if z_seq.shape[1] <= 0:
            raise ValueError("z_seq sequence length must be positive")
        if b.ndim != 2 or b.shape[-1] != self.static_dim:
            raise ValueError(f"b must have shape [B, {self.static_dim}]")
        if b.shape[0] != z_seq.shape[0]:
            raise ValueError("b batch dimension must match z_seq")

        _, hidden = self.gru(z_seq)
        parts = [hidden[-1], b]

        if self.q_dim:
            if q is None or q.shape != (z_seq.shape[0], self.q_dim):
                raise ValueError(f"q must have shape [B, {self.q_dim}]")
            parts.append(q)
        elif q is not None:
            raise ValueError("q was provided but q_dim is 0")

        if self.action_dim:
            if actions is None or actions.shape != (z_seq.shape[0], z_seq.shape[1], self.action_dim):
                raise ValueError(f"actions must have shape [B, K, {self.action_dim}]")
            parts.append(actions.mean(dim=1))
        elif actions is not None:
            raise ValueError("actions were provided but action_dim is 0")

        return self.fusion(torch.cat(parts, dim=-1))
