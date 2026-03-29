"""Compare 2-bit, 3-bit, and 4-bit KV cache compression side by side.

Usage:
    python examples/compare_bit_widths.py

Requires: mlx-community/Qwen2.5-0.5B-Instruct-4bit (downloaded automatically)
"""

from mlx_lm import load

from mlx_turboquant.constants import CANONICAL_SAMPLE_MODEL
from mlx_turboquant.integration.generate_wrapper import (
    generate_baseline,
    generate_with_compressed_cache,
)

MODEL = CANONICAL_SAMPLE_MODEL
PROMPT = "Write a Python function that checks if a number is prime."


def main() -> None:
    print(f"Loading {MODEL}...")
    model, tokenizer = load(MODEL)  # type: ignore[misc]

    print("\n=== BASELINE (fp16 cache) ===")
    baseline = generate_baseline(model, tokenizer, PROMPT, max_tokens=80)
    print(baseline.text)
    print(f"  Cache: {baseline.cache_bytes / 1024:.1f} KB")

    for bits in [4, 3, 2]:
        print(f"\n=== {bits}-BIT COMPRESSED ===")
        result = generate_with_compressed_cache(
            model,
            tokenizer,
            PROMPT,
            kv_bits=bits,
            max_tokens=80,
        )
        print(result.text)

        ratio = baseline.cache_bytes / result.cache_bytes if result.cache_bytes > 0 else 0
        print(f"  Cache: {result.cache_bytes / 1024:.1f} KB ({ratio:.1f}x compression)")
        print(f"  Decode: {result.decode_tokens_per_sec:.1f} tok/s")

        # Token match with baseline
        min_len = min(len(baseline.tokens), len(result.tokens))
        if min_len > 0:
            matches = sum(
                1
                for a, b in zip(baseline.tokens[:min_len], result.tokens[:min_len], strict=True)
                if a == b
            )
            print(f"  Token match: {matches}/{min_len} ({100 * matches / min_len:.0f}%)")


if __name__ == "__main__":
    main()
