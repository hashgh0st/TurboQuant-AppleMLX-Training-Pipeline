"""Shared project constants used across CLI, tests, and examples."""

from __future__ import annotations

CANONICAL_SAMPLE_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
SUPPORTED_KV_BITS: tuple[int, ...] = (2, 3, 4)
