"""Quickstart: compare baseline generation with experimental compressed KV cache.

Usage:
    python examples/quickstart.py

Requires: mlx-community/Qwen2.5-0.5B-Instruct-4bit (downloaded automatically)
"""

from mlx_lm import load

from mlx_turboquant.constants import CANONICAL_SAMPLE_MODEL
from mlx_turboquant.integration.generate_wrapper import (
    generate_baseline,
    generate_with_compressed_cache,
)

MODEL = CANONICAL_SAMPLE_MODEL
PROMPT = "Explain how KV cache compression works in transformer models."


def main() -> None:
    print(f"Loading {MODEL}...")
    model, tokenizer = load(MODEL)  # type: ignore[misc]

    print("\nGenerating with baseline cache...")
    baseline = generate_baseline(model, tokenizer, PROMPT, max_tokens=100)
    print(baseline.text)
    print(f"\n  Logical cache: {baseline.cache_bytes / 1024:.1f} KB")
    if baseline.cache_allocated_bytes not in (None, baseline.cache_bytes):
        print(f"  Allocated cache: {baseline.cache_allocated_bytes / 1024:.1f} KB")

    print("\nGenerating with experimental 3-bit compressed cache...")
    result = generate_with_compressed_cache(
        model,
        tokenizer,
        PROMPT,
        kv_bits=3,
        max_tokens=100,
    )
    print(result.text)
    print(f"\n  Tokens: {result.tokens_generated}")
    print(f"  TTFT: {result.ttft_ms:.1f} ms")
    print(f"  Decode: {result.decode_tokens_per_sec:.1f} tok/s")
    print(f"  Logical cache: {result.cache_bytes / 1024:.1f} KB")
    if result.cache_allocated_bytes not in (None, result.cache_bytes):
        print(f"  Allocated cache: {result.cache_allocated_bytes / 1024:.1f} KB")

    if baseline.cache_bytes > 0 and result.cache_bytes > 0:
        ratio = baseline.cache_bytes / result.cache_bytes
        print(f"\n  Logical compression: {ratio:.1f}x cache memory reduction")


if __name__ == "__main__":
    main()
