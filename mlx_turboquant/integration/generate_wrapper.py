"""Generation wrapper threading compressed cache through MLX-LM's generate_step.

Provides generate_with_compressed_cache() and generate_baseline() for A/B
comparison of compressed vs standard KV cache.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

from mlx_turboquant.cache.memory_accounting import estimate_memory
from mlx_turboquant.integration.compression_profile import (
    CompressionBackend,
    CompressionProfile,
)
from mlx_turboquant.integration.mlx_lm_adapter import (
    introspect_model,
    make_compressed_cache,
)


@dataclass
class GenerationResult:
    """Result of a single generation run with metrics."""

    text: str
    tokens: list[int]  # raw token ids as generated
    tokens_generated: int
    ttft_ms: float  # time to first token (includes prefill + first decode step)
    decode_tokens_per_sec: float
    cache_bytes: int  # logical occupied bytes for the generated cache state
    cache_mode: str  # "baseline" | "compressed-{bits}bit"
    cache_allocated_bytes: int | None = None


def _cache_occupied_tokens(prompt_cache: list[Any]) -> int:
    """Return the occupied token count from the largest cache layer."""
    sizes: list[int] = []
    for layer in prompt_cache:
        size = getattr(layer, "size", None)
        if callable(size):
            sizes.append(int(size()))
    return max(sizes, default=0)


def _allocated_cache_bytes(prompt_cache: list[Any]) -> int:
    """Return the raw backing-buffer bytes currently allocated across all layers."""
    total = 0
    for layer in prompt_cache:
        allocated = getattr(layer, "allocated_nbytes", None)
        total += int(allocated) if allocated is not None else int(layer.nbytes)
    return total


def _logical_cache_bytes(
    model: Any,
    prompt_cache: list[Any],
    *,
    key_bits: int | None,
    value_bits: int | None = None,
) -> int:
    """Estimate occupied cache bytes from model geometry and live token count."""
    occupied_tokens = _cache_occupied_tokens(prompt_cache)
    if occupied_tokens == 0:
        return 0

    info = introspect_model(model)
    baseline_bytes = 2 * info.num_layers * info.num_kv_heads * info.head_dim * 2 * occupied_tokens
    if key_bits is None:
        return baseline_bytes

    report = estimate_memory(
        num_layers=info.num_layers,
        num_kv_heads=info.num_kv_heads,
        head_dim=info.head_dim,
        kv_bits=key_bits,
        seq_len=occupied_tokens,
        value_kv_bits=value_bits,
    )
    return report.compressed_bytes


def _compressed_cache_mode(profile: CompressionProfile) -> str:
    """Return the benchmark/report label for a compressed operating point."""
    return profile.cache_mode


def generate_with_compressed_cache(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    kv_bits: int = 3,
    value_kv_bits: int | None = None,
    backend: CompressionBackend = "reference",
    max_tokens: int = 256,
    temp: float = 0.0,
    seed: int = 42,
    sink_tokens: int = 0,
    model_name: str | None = None,
    calibrated_dir: Path | None = None,
    use_qjl: bool = False,
) -> GenerationResult:
    """Generate text using compressed KV cache.

    Creates a CompressedKVCache and passes it as prompt_cache to generate_step
    with kv_bits=None to disable MLX-LM's built-in affine quantization.
    """
    profile = CompressionProfile(
        kv_bits,
        value_bits=value_kv_bits,
        backend=backend,
        seed=seed,
    )
    prompt_tokens = mx.array(tokenizer.encode(prompt))
    cache = make_compressed_cache(
        model,
        kv_bits=profile.key_bits,
        value_kv_bits=profile.effective_value_bits,
        backend=profile.backend,
        seed=profile.seed,
        sink_tokens=sink_tokens,
        model_name=model_name,
        calibrated_dir=calibrated_dir,
        use_qjl=use_qjl,
    )

    return _run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        prompt_cache=cache,
        max_tokens=max_tokens,
        temp=temp,
        cache_mode=_compressed_cache_mode(profile),
        generate_step_kv_bits=None,  # disable MLX-LM's affine quant
        logical_kv_bits=profile.key_bits,
        logical_value_kv_bits=profile.effective_value_bits,
    )


def generate_baseline(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int = 256,
    temp: float = 0.0,
) -> GenerationResult:
    """Generate text using standard MLX-LM KV cache for A/B comparison."""
    prompt_tokens = mx.array(tokenizer.encode(prompt))
    cache = make_prompt_cache(model)

    return _run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        prompt_cache=cache,
        max_tokens=max_tokens,
        temp=temp,
        cache_mode="baseline",
        generate_step_kv_bits=None,
        logical_kv_bits=None,
    )


def _run_generation(
    model: Any,
    tokenizer: Any,
    prompt_tokens: mx.array,
    prompt_cache: list[Any],
    max_tokens: int,
    temp: float,
    cache_mode: str,
    generate_step_kv_bits: int | None,
    logical_kv_bits: int | None,
    logical_value_kv_bits: int | None = None,
) -> GenerationResult:
    """Shared generation loop for both compressed and baseline paths."""
    sampler = (
        (lambda x: mx.argmax(x, axis=-1))
        if temp == 0.0
        else (lambda x: mx.random.categorical(x * (1.0 / temp)))
    )

    t_start = time.perf_counter()

    tokens_out: list[int] = []
    first_token_time: float | None = None

    # generate_step yields (token_id: int via .item(), logprobs: mx.array)
    for token_id, _logprobs in generate_step(
        prompt=prompt_tokens,
        model=model,
        sampler=sampler,
        prompt_cache=prompt_cache,
        kv_bits=generate_step_kv_bits,
        max_tokens=max_tokens,
    ):
        if first_token_time is None:
            first_token_time = time.perf_counter()

        tokens_out.append(int(token_id))  # .item() returns int but mypy sees mx.array

        if token_id == tokenizer.eos_token_id:
            break

    t_end = time.perf_counter()

    # first_token_time captures TTFT (prefill + first decode step)
    ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else 0.0
    decode_time = t_end - (first_token_time or t_end)
    decode_tokens = max(len(tokens_out) - 1, 1)
    decode_tok_per_sec = decode_tokens / decode_time if decode_time > 0 else 0.0

    cache_bytes = _logical_cache_bytes(
        model,
        prompt_cache,
        key_bits=logical_kv_bits,
        value_bits=logical_value_kv_bits,
    )
    cache_allocated_bytes = _allocated_cache_bytes(prompt_cache)
    text = tokenizer.decode(tokens_out)

    return GenerationResult(
        text=text,
        tokens=tokens_out,
        tokens_generated=len(tokens_out),
        ttft_ms=ttft_ms,
        decode_tokens_per_sec=decode_tok_per_sec,
        cache_bytes=cache_bytes,
        cache_mode=cache_mode,
        cache_allocated_bytes=cache_allocated_bytes,
    )
