"""Latency benchmarks reusing generate_wrapper for timing.

Takes a loaded model+tokenizer (not a path) — model loading is expensive
and should happen once in the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlx_turboquant.integration.compression_profile import (
    CompressionProfile,
    resolve_profiles,
)
from mlx_turboquant.integration.generate_wrapper import (
    GenerationResult,
    generate_baseline,
    generate_with_compressed_cache,
)


def benchmark_latency(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int = 100,
    kv_bits_list: list[int] | None = None,
    profiles: list[CompressionProfile] | None = None,
    warmup: int = 1,
    runs: int = 3,
    model_name: str | None = None,
    calibrated_dir: Path | None = None,
) -> list[GenerationResult]:
    """Run generation multiple times for stable latency measurements.

    Returns all GenerationResult objects (one per run per config).
    Includes baseline + each kv_bits setting.
    """
    profiles = resolve_profiles(profiles, kv_bits_list, default_bits=[3, 4])

    results: list[GenerationResult] = []

    # Warmup + measure baseline
    for i in range(warmup + runs):
        r = generate_baseline(model, tokenizer, prompt, max_tokens=max_tokens)
        if i >= warmup:
            results.append(r)

    # Warmup + measure each bit width
    for profile in profiles:
        for i in range(warmup + runs):
            r = generate_with_compressed_cache(
                model,
                tokenizer,
                prompt,
                kv_bits=profile.key_bits,
                value_kv_bits=profile.effective_value_bits,
                backend=profile.backend,
                max_tokens=max_tokens,
                model_name=model_name,
                calibrated_dir=calibrated_dir,
            )
            if i >= warmup:
                results.append(r)

    return results
