"""Quality benchmarks: compare compressed vs baseline token output at temp=0."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_turboquant.integration.compression_profile import (
    CompressionProfile,
    resolve_profiles,
)
from mlx_turboquant.integration.generate_wrapper import (
    generate_baseline,
    generate_with_compressed_cache,
)


@dataclass
class QualityResult:
    """Quality comparison for one (prompt, kv_bits) pair."""

    prompt_id: str
    cache_mode: str
    kv_bits: int
    value_kv_bits: int
    backend: str
    token_match_ratio: float
    first_divergence_position: int
    baseline_tokens: int
    compressed_tokens: int


def benchmark_quality(
    model: Any,
    tokenizer: Any,
    prompts: dict[str, str],
    *,
    kv_bits_list: list[int] | None = None,
    profiles: list[CompressionProfile] | None = None,
    max_tokens: int = 100,
    model_name: str | None = None,
    calibrated_dir: Path | None = None,
) -> list[QualityResult]:
    """For each prompt, generate baseline and compressed at temp=0, compare tokens.

    Token match ratio = (number of matching tokens) / min(baseline_len, compressed_len).
    First divergence position = index of first differing token (or -1 if identical).
    """
    profiles = resolve_profiles(profiles, kv_bits_list, default_bits=[2, 3, 4])

    results: list[QualityResult] = []

    for prompt_id, prompt_text in prompts.items():
        baseline = generate_baseline(
            model,
            tokenizer,
            prompt_text,
            max_tokens=max_tokens,
            temp=0.0,
        )

        for profile in profiles:
            compressed = generate_with_compressed_cache(
                model,
                tokenizer,
                prompt_text,
                kv_bits=profile.key_bits,
                value_kv_bits=profile.effective_value_bits,
                backend=profile.backend,
                max_tokens=max_tokens,
                temp=0.0,
                model_name=model_name,
                calibrated_dir=calibrated_dir,
            )

            # Single-pass comparison of raw token sequences
            min_len = min(len(baseline.tokens), len(compressed.tokens))
            matches = 0
            first_div = -1
            for idx in range(min_len):
                if baseline.tokens[idx] == compressed.tokens[idx]:
                    matches += 1
                elif first_div == -1:
                    first_div = idx
            match_ratio = matches / min_len if min_len > 0 else 0.0
            if min_len == 0:
                first_div = 0
            elif first_div == -1 and len(baseline.tokens) != len(compressed.tokens):
                first_div = min_len

            results.append(
                QualityResult(
                    prompt_id=prompt_id,
                    cache_mode=compressed.cache_mode,
                    kv_bits=profile.key_bits,
                    value_kv_bits=profile.effective_value_bits,
                    backend=profile.backend,
                    token_match_ratio=match_ratio,
                    first_divergence_position=first_div,
                    baseline_tokens=len(baseline.tokens),
                    compressed_tokens=len(compressed.tokens),
                )
            )

    return results
