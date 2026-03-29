"""Generation wrapper threading compressed cache through MLX-LM's generate_step.

Provides generate_with_compressed_cache() and generate_baseline() for A/B
comparison of compressed vs standard KV cache.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

from mlx_turboquant.integration.mlx_lm_adapter import make_compressed_cache


@dataclass
class GenerationResult:
    """Result of a single generation run with metrics."""

    text: str
    tokens: list[int]  # raw token ids as generated
    tokens_generated: int
    ttft_ms: float  # time to first token (includes prefill + first decode step)
    decode_tokens_per_sec: float
    cache_bytes: int
    cache_mode: str  # "baseline" | "compressed-{bits}bit"


def generate_with_compressed_cache(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    kv_bits: int = 3,
    max_tokens: int = 256,
    temp: float = 0.0,
    seed: int = 42,
) -> GenerationResult:
    """Generate text using compressed KV cache.

    Creates a CompressedKVCache and passes it as prompt_cache to generate_step
    with kv_bits=None to disable MLX-LM's built-in affine quantization.
    """
    prompt_tokens = mx.array(tokenizer.encode(prompt))
    cache = make_compressed_cache(model, kv_bits=kv_bits, seed=seed)

    return _run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        prompt_cache=cache,
        max_tokens=max_tokens,
        temp=temp,
        cache_mode=f"compressed-{kv_bits}bit",
        kv_bits=None,  # disable MLX-LM's affine quant
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
        kv_bits=None,
    )


def _run_generation(
    model: Any,
    tokenizer: Any,
    prompt_tokens: mx.array,
    prompt_cache: list[Any],
    max_tokens: int,
    temp: float,
    cache_mode: str,
    kv_bits: int | None,
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
        kv_bits=kv_bits,
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

    cache_bytes = sum(c.nbytes for c in prompt_cache)
    text = tokenizer.decode(tokens_out)

    return GenerationResult(
        text=text,
        tokens=tokens_out,
        tokens_generated=len(tokens_out),
        ttft_ms=ttft_ms,
        decode_tokens_per_sec=decode_tok_per_sec,
        cache_bytes=cache_bytes,
        cache_mode=cache_mode,
    )
