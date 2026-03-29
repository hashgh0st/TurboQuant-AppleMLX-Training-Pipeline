"""Latency benchmarks reusing generate_wrapper for timing.

Takes a loaded model+tokenizer (not a path) — model loading is expensive
and should happen once in the CLI.
"""

from __future__ import annotations

from typing import Any

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
    warmup: int = 1,
    runs: int = 3,
) -> list[GenerationResult]:
    """Run generation multiple times for stable latency measurements.

    Returns all GenerationResult objects (one per run per config).
    Includes baseline + each kv_bits setting.
    """
    if kv_bits_list is None:
        kv_bits_list = [3, 4]

    results: list[GenerationResult] = []

    # Warmup + measure baseline
    for i in range(warmup + runs):
        r = generate_baseline(model, tokenizer, prompt, max_tokens=max_tokens)
        if i >= warmup:
            results.append(r)

    # Warmup + measure each bit width
    for bits in kv_bits_list:
        for i in range(warmup + runs):
            r = generate_with_compressed_cache(
                model,
                tokenizer,
                prompt,
                kv_bits=bits,
                max_tokens=max_tokens,
            )
            if i >= warmup:
                results.append(r)

    return results
