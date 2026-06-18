from __future__ import annotations

import torch
from torch import nn


class PolicyEncoderAdapter(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    @property
    def memory_dim(self) -> int:
        return int(self.encoder.memory_dim)

    def forward(self, obs: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(obs, **kwargs)
        return outputs["memory"]
