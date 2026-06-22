from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MineRLSequenceDataset(Dataset):
    """Offline MineRL-style sequence dataset backed by preprocessed `.pt` shards."""

    def __init__(
        self,
        dataset_path: str | Path,
        sequence_length: int,
        image_size: int,
        action_dim: int,
        change_threshold: float = 0.05,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.action_dim = action_dim
        self.change_threshold = change_threshold
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")

        shard_paths = self._resolve_shards(self.dataset_path)
        observations: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        for shard_path in shard_paths:
            payload = _safe_torch_load(shard_path)
            if not isinstance(payload, dict) or "obs" not in payload or "actions" not in payload:
                raise ValueError(f"MineRL shard must contain obs and actions: {shard_path}")
            observations.append(self._normalize_obs(payload["obs"], shard_path))
            actions.append(self._normalize_actions(payload["actions"], shard_path))

        self.obs = torch.cat(observations, dim=0)
        self.actions = torch.cat(actions, dim=0)

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sequence = self.obs[index]
        context = sequence[:-1]
        next_obs = sequence[-1]
        last_obs = context[-1]
        change = (next_obs - last_obs).abs().mean(dim=0, keepdim=True)
        change_mask = (change > self.change_threshold).float()
        return {
            "obs": context,
            "actions": self.actions[index],
            "next_obs": next_obs,
            "change_mask": change_mask,
        }

    @staticmethod
    def _resolve_shards(path: Path) -> list[Path]:
        if path.is_file():
            return [path]
        if not path.exists():
            raise FileNotFoundError(f"MineRL dataset path does not exist: {path}")
        shards = sorted(path.glob("*.pt"))
        if not shards:
            raise FileNotFoundError(f"no .pt MineRL shards found in: {path}")
        return shards

    def _normalize_obs(self, obs: object, shard_path: Path) -> torch.Tensor:
        tensor = torch.as_tensor(obs)
        if tensor.ndim != 5:
            raise ValueError(f"obs must have shape [N,T,C,H,W] in {shard_path}")
        if tensor.shape[1] != self.sequence_length:
            raise ValueError(f"obs sequence length must be {self.sequence_length} in {shard_path}")
        tensor = tensor.float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        if tensor.shape[2] not in (1, 3):
            raise ValueError(f"obs channels must be 1 or 3 in {shard_path}")
        if tensor.shape[-2:] != (self.image_size, self.image_size):
            flat = tensor.flatten(0, 1)
            flat = F.interpolate(
                flat,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            tensor = flat.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2], self.image_size, self.image_size)
        return tensor.clamp(0.0, 1.0).contiguous()

    def _normalize_actions(self, actions: object, shard_path: Path) -> torch.Tensor:
        tensor = torch.as_tensor(actions).float()
        expected_shape = (self.sequence_length - 1, self.action_dim)
        if tensor.ndim != 3 or tensor.shape[1:] != expected_shape:
            raise ValueError(f"actions must have shape [N,{expected_shape[0]},{expected_shape[1]}] in {shard_path}")
        return tensor.contiguous()


def _safe_torch_load(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
