from __future__ import annotations

import torch
from torch import nn

from sdam.models.associative_memory import AssociativeMemory
from sdam.models.dynamic_encoder import DynamicEncoder
from sdam.models.static_encoder import StaticEncoder


class SDAMEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        sequence_length: int,
        hidden_channels: int,
        static_dim: int,
        dynamic_dim: int,
        assoc_dim: int,
        q_dim: int = 0,
        action_dim: int = 0,
    ) -> None:
        super().__init__()
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.static_encoder = StaticEncoder(in_channels, hidden_channels, static_dim)
        self.dynamic_encoder = DynamicEncoder(in_channels, hidden_channels, dynamic_dim)
        self.associative_memory = AssociativeMemory(dynamic_dim, static_dim, assoc_dim, q_dim, action_dim)
        self.sequence_length = sequence_length
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.assoc_dim = assoc_dim
        self.q_dim = q_dim
        self.action_dim = action_dim
        self.memory_dim = static_dim + ((sequence_length - 1) * dynamic_dim) + assoc_dim + q_dim

    def forward(
        self,
        obs: torch.Tensor,
        q: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if obs.ndim != 5:
            raise ValueError("obs must have shape [B, T, C, H, W]")
        if obs.shape[1] < 2:
            raise ValueError("obs time dimension must be at least 2")
        if obs.shape[1] != self.sequence_length:
            raise ValueError(f"obs time dimension must be {self.sequence_length}")
        batch, time = obs.shape[:2]

        q_t = None
        if self.q_dim:
            if q is None or q.shape != (batch, time, self.q_dim):
                raise ValueError(f"q must have shape [B, T, {self.q_dim}]")
            q_t = q[:, -1]
        elif q is not None:
            raise ValueError("q was provided but q_dim is 0")

        if self.action_dim:
            if actions is None or actions.shape != (batch, time - 1, self.action_dim):
                raise ValueError(f"actions must have shape [B, T - 1, {self.action_dim}]")
        elif actions is not None:
            raise ValueError("actions were provided but action_dim is 0")

        b = self.static_encoder(obs)
        z_seq = self.dynamic_encoder(obs)
        c = self.associative_memory(z_seq=z_seq, b=b, q=q_t, actions=actions)
        memory_parts = [b, z_seq.flatten(start_dim=1), c]
        if q_t is not None:
            memory_parts.append(q_t)
        memory = torch.cat(memory_parts, dim=-1)
        return {"b": b, "z_seq": z_seq, "c": c, "memory": memory, "aux": {}}
