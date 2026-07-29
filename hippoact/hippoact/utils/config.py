"""Thin YAML config loader on top of OmegaConf."""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    cfg = OmegaConf.load(Path(path))
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"expected top-level dict in {path}")
    return cfg
