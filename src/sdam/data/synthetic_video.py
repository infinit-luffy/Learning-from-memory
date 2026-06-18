from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


_MAX_TORCH_SEED = 2**63 - 1
_UINT64_MASK = 2**64 - 1


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
        for field_name in (
            "image_size",
            "channels",
            "sequence_length",
            "dataset_size",
            "object_size",
            "clutter_count",
        ):
            value = getattr(config, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an int")

        for field_name in ("min_speed", "max_speed"):
            value = getattr(config, field_name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{field_name} must be a number")
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")

        if config.image_size <= 0:
            raise ValueError("image_size must be positive")
        if config.channels <= 0:
            raise ValueError("channels must be positive")
        if config.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if config.dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if config.object_size <= 0:
            raise ValueError("object_size must be positive")
        if config.object_size == config.image_size:
            raise ValueError("object_size must be smaller than image_size")
        if config.object_size > config.image_size:
            raise ValueError("object_size must be <= image_size")
        placement_limit = config.image_size - config.object_size
        if placement_limit < 2:
            raise ValueError("image_size - object_size must be at least 2")
        if config.clutter_count < 0:
            raise ValueError("clutter_count must be >= 0")
        if config.min_speed <= 0 or config.min_speed > config.max_speed:
            raise ValueError("speed range must satisfy 0 < min_speed <= max_speed")
        self.config = config
        self.seed = seed

    def __len__(self) -> int:
        return self.config.dataset_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self._sample_seed(index))
        cfg = self.config
        background = self._make_background(generator)
        obs = background.unsqueeze(0).repeat(cfg.sequence_length, 1, 1, 1)
        mask = torch.zeros(cfg.sequence_length, 1, cfg.image_size, cfg.image_size)

        top_left_positions = self._target_top_left_positions(generator)
        target_positions = self._target_center_positions(top_left_positions)
        color = torch.full((cfg.channels, 1, 1), 0.95)
        self._draw_static_clutter(obs, generator)

        for t, pos in enumerate(top_left_positions):
            x = int(pos[0].item())
            y = int(pos[1].item())
            obs[t, :, y : y + cfg.object_size, x : x + cfg.object_size] = color
            mask[t, :, y : y + cfg.object_size, x : x + cfg.object_size] = 1.0

        velocity = target_positions[-1] - target_positions[-2]
        return {
            "obs": obs.clamp(0.0, 1.0).float(),
            "target_positions": target_positions.float(),
            "target_position": target_positions[-1].float(),
            "target_velocity": velocity.float(),
            "dynamic_mask": mask.float(),
            "background": background.float(),
        }

    def _sample_seed(self, index: int) -> int:
        value = (int(self.seed) * 0x9E3779B185EBCA87) & _UINT64_MASK
        value ^= (int(index) * 0xC2B2AE3D27D4EB4F) & _UINT64_MASK
        value ^= value >> 33
        value = (value * 0xFF51AFD7ED558CCD) & _UINT64_MASK
        value ^= value >> 33
        value = (value * 0xC4CEB9FE1A85EC53) & _UINT64_MASK
        value ^= value >> 33
        return value % _MAX_TORCH_SEED

    def _make_background(self, generator: torch.Generator) -> torch.Tensor:
        cfg = self.config
        low_res = torch.rand(cfg.channels, 4, 4, generator=generator) * 0.35
        return torch.nn.functional.interpolate(
            low_res.unsqueeze(0),
            size=(cfg.image_size, cfg.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def _target_top_left_positions(self, generator: torch.Generator) -> torch.Tensor:
        for _ in range(128):
            positions = self._target_top_left_positions_once(generator)
            if self._has_visible_motion(positions):
                return positions
        raise RuntimeError("failed to generate visible moving-object trajectory after 128 attempts")

    def _target_top_left_positions_once(self, generator: torch.Generator) -> torch.Tensor:
        cfg = self.config
        limit = cfg.image_size - cfg.object_size
        speed = cfg.min_speed + torch.rand(1, generator=generator).item() * (cfg.max_speed - cfg.min_speed)
        step = max(1, min(limit, int(round(speed))))

        pos = torch.randint(0, limit + 1, (2,), generator=generator, dtype=torch.long)
        velocity = None
        for _ in range(128):
            candidate = torch.randint(-step, step + 1, (2,), generator=generator, dtype=torch.long)
            if torch.any(candidate != 0):
                velocity = candidate
                break
        if velocity is None:
            velocity = torch.tensor([step, 0], dtype=torch.long)

        positions = []
        for _ in range(cfg.sequence_length):
            positions.append(pos.clone())
            next_pos = pos + velocity
            for axis in range(2):
                if next_pos[axis] < 0 or next_pos[axis] > limit:
                    velocity[axis] = -velocity[axis]
            pos = (pos + velocity).clamp(0, limit)
        return torch.stack(positions)

    def _has_visible_motion(self, positions: torch.Tensor) -> bool:
        return not torch.equal(positions[-1], positions[0]) and not torch.equal(positions[-1], positions[-2])

    def _target_center_positions(self, top_left_positions: torch.Tensor) -> torch.Tensor:
        offset = (self.config.object_size - 1) / 2
        return top_left_positions + offset

    def _draw_static_clutter(self, obs: torch.Tensor, generator: torch.Generator) -> None:
        cfg = self.config
        limit = cfg.image_size - cfg.object_size
        for _ in range(cfg.clutter_count):
            x = int(torch.randint(0, limit + 1, (1,), generator=generator).item())
            y = int(torch.randint(0, limit + 1, (1,), generator=generator).item())
            color = torch.full((cfg.channels, 1, 1), 0.8)
            obs[:, :, y : y + cfg.object_size, x : x + cfg.object_size] = color
