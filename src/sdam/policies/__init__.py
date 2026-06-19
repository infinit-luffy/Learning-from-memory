"""Policy-facing adapters for SDAM encoders."""

from __future__ import annotations

from sdam.policies.sb3_atari import atari_observations_to_sdam

__all__ = ["PolicyEncoderAdapter", "atari_observations_to_sdam"]


def __getattr__(name: str) -> object:
    if name == "PolicyEncoderAdapter":
        from sdam.policies.encoder_adapter import PolicyEncoderAdapter

        return PolicyEncoderAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
