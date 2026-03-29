"""Compression profiles for experimental KV-cache operating points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mlx_turboquant.constants import SUPPORTED_KV_BITS

CompressionBackend = Literal["reference", "metal"]


@dataclass(frozen=True)
class CompressionProfile:
    """Configures an experimental compressed-cache operating point."""

    key_bits: int
    value_bits: int | None = None
    backend: CompressionBackend = "reference"
    seed: int = 42

    def __post_init__(self) -> None:
        value_bits = self.effective_value_bits
        if self.key_bits not in SUPPORTED_KV_BITS:
            raise ValueError(f"unsupported key_bits={self.key_bits}")
        if value_bits not in SUPPORTED_KV_BITS:
            raise ValueError(f"unsupported value_bits={value_bits}")

    @property
    def effective_value_bits(self) -> int:
        """Return the value-branch bit width, defaulting to key_bits."""
        return self.key_bits if self.value_bits is None else self.value_bits

    @property
    def cache_mode(self) -> str:
        """Stable label used in benchmark output and generation results."""
        if self.key_bits == self.effective_value_bits:
            label = f"{self.key_bits}bit"
        else:
            label = f"k{self.key_bits}v{self.effective_value_bits}bit"
        if self.backend != "reference":
            label = f"{label}-{self.backend}"
        return f"compressed-{label}"


def resolve_profiles(
    profiles: list[CompressionProfile] | None,
    kv_bits_list: list[int] | None,
    default_bits: list[int],
) -> list[CompressionProfile]:
    """Resolve a profiles list from the various input options.

    Priority: explicit profiles > kv_bits_list > default_bits.
    """
    if profiles is not None:
        return profiles
    bits = kv_bits_list if kv_bits_list is not None else default_bits
    return [CompressionProfile(b) for b in bits]


def default_experimental_profiles() -> list[CompressionProfile]:
    """Research candidates for quality/performance exploration.

    These profiles have NOT been validated against promotion thresholds.
    Run ``mlx-tq bench --gate`` to evaluate which profiles pass quality gates.
    """
    return [
        CompressionProfile(3),
        CompressionProfile(4),
        CompressionProfile(3, value_bits=4),
        CompressionProfile(4, backend="metal"),
    ]
