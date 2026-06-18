from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticVideoConfig:
    image_size: int
    channels: int
    sequence_length: int
    dataset_size: int
    object_size: int
    clutter_count: int
    min_speed: float
    max_speed: float


class SyntheticVideoDataset(Dataset):
    def __init__(self, config: SyntheticVideoConfig, seed: int = 0) -> None:
        if config.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.config = config
        self.seed = seed

    def __len__(self) -> int:
        return self.config.dataset_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        cfg = self.config
        background = self._make_background(generator)
        obs = background.unsqueeze(0).repeat(cfg.sequence_length, 1, 1, 1)
        mask = torch.zeros(cfg.sequence_length, 1, cfg.image_size, cfg.image_size)

        positions = self._target_positions(generator)
        color = torch.full((cfg.channels, 1, 1), 0.95)
        self._draw_static_clutter(obs, generator)

        for t, pos in enumerate(positions):
            x = int(pos[0].item())
            y = int(pos[1].item())
            obs[t, :, y : y + cfg.object_size, x : x + cfg.object_size] = color
            mask[t, :, y : y + cfg.object_size, x : x + cfg.object_size] = 1.0

        velocity = positions[-1] - positions[-2]
        return {
            "obs": obs.clamp(0.0, 1.0).float(),
            "target_positions": positions.float(),
            "target_position": positions[-1].float(),
            "target_velocity": velocity.float(),
            "dynamic_mask": mask.float(),
            "background": background.float(),
        }

    def _make_background(self, generator: torch.Generator) -> torch.Tensor:
        cfg = self.config
        low_res = torch.rand(cfg.channels, 4, 4, generator=generator) * 0.35
        return torch.nn.functional.interpolate(
            low_res.unsqueeze(0),
            size=(cfg.image_size, cfg.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _target_positions(self, generator: torch.Generator) -> torch.Tensor:
        cfg = self.config
        limit = cfg.image_size - cfg.object_size - 1
        pos = torch.rand(2, generator=generator) * limit
        direction = torch.randn(2, generator=generator)
        direction = direction / direction.norm().clamp_min(1e-6)
        speed = cfg.min_speed + torch.rand(1, generator=generator).item() * (cfg.max_speed - cfg.min_speed)
        velocity = direction * speed
        positions = []
        for _ in range(cfg.sequence_length):
            positions.append(pos.round().clamp(0, limit))
            next_pos = pos + velocity
            for axis in range(2):
                if next_pos[axis] < 0 or next_pos[axis] > limit:
                    velocity[axis] = -velocity[axis]
            pos = (pos + velocity).clamp(0, limit)
        return torch.stack(positions)

    def _draw_static_clutter(self, obs: torch.Tensor, generator: torch.Generator) -> None:
        cfg = self.config
        limit = cfg.image_size - cfg.object_size - 1
        for _ in range(cfg.clutter_count):
            x = int(torch.randint(0, limit + 1, (1,), generator=generator).item())
            y = int(torch.randint(0, limit + 1, (1,), generator=generator).item())
            color = torch.full((cfg.channels, 1, 1), 0.8)
            obs[:, :, y : y + cfg.object_size, x : x + cfg.object_size] = color
