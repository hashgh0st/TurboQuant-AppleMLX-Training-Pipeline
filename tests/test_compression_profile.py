"""Tests for compression profile helpers."""

from __future__ import annotations

import pytest

from mlx_turboquant.integration.compression_profile import (
    CompressionProfile,
    default_experimental_profiles,
)


def test_profile_defaults_value_bits_to_key_bits() -> None:
    profile = CompressionProfile(3)
    assert profile.effective_value_bits == 3
    assert profile.cache_mode == "compressed-3bit"


def test_profile_labels_mixed_precision_and_backend() -> None:
    profile = CompressionProfile(3, value_bits=4, backend="metal")
    assert profile.effective_value_bits == 4
    assert profile.cache_mode == "compressed-k3v4bit-metal"


def test_invalid_bits_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported key_bits"):
        CompressionProfile(5)


def test_default_experimental_profiles_include_mixed_and_metal() -> None:
    labels = [profile.cache_mode for profile in default_experimental_profiles()]
    assert "compressed-k3v4bit" in labels
    assert "compressed-4bit-metal" in labels
