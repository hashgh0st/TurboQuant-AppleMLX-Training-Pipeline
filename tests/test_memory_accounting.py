"""Tests for memory accounting."""

from __future__ import annotations

import mlx.core as mx

from mlx_turboquant.cache.cache_layout import CacheConfig, create_cache_layers
from mlx_turboquant.cache.memory_accounting import estimate_memory, measure_memory


class TestEstimate:
    def test_qwen_3b_3bit(self) -> None:
        """Qwen 2.5-3B: 36 layers, 2 KV heads, head_dim=128, 3-bit at 4096 tokens."""
        report = estimate_memory(
            num_layers=36, num_kv_heads=2, head_dim=128, kv_bits=3, seq_len=4096
        )
        # Baseline: 36 * 2 * 128 * 2 * 2 * 4096 bytes
        assert report.baseline_bytes == 36 * 2 * 128 * 2 * 2 * 4096
        assert report.compression_ratio > 4.0
        assert report.compressed_bytes < report.baseline_bytes

    def test_compression_ratio_3bit_dim128(self) -> None:
        """3-bit compression of dim=128 should achieve > 4x ratio."""
        report = estimate_memory(
            num_layers=1, num_kv_heads=1, head_dim=128, kv_bits=3, seq_len=1000
        )
        assert report.compression_ratio > 4.0, f"Ratio: {report.compression_ratio:.2f}"

    def test_higher_bits_lower_ratio(self) -> None:
        base = dict(num_layers=1, num_kv_heads=1, head_dim=128, seq_len=1000)
        r2 = estimate_memory(kv_bits=2, **base)
        r3 = estimate_memory(kv_bits=3, **base)
        r4 = estimate_memory(kv_bits=4, **base)
        assert r2.compression_ratio > r3.compression_ratio > r4.compression_ratio


class TestMeasure:
    def test_matches_estimate(self) -> None:
        """Live cache measurement should be within 10% of formula estimate."""
        config = CacheConfig(
            num_layers=4, num_kv_heads=2, head_dim=128,
            max_seq_len=4096, kv_bits=3,
        )
        layers = create_cache_layers(config)
        n_tokens = 100

        # Fill each layer with random KV data
        for layer in layers:
            k = mx.random.normal((1, 2, n_tokens, 128))
            v = mx.random.normal((1, 2, n_tokens, 128))
            layer.update_and_fetch(k, v)

        measured = measure_memory(layers)
        estimated = estimate_memory(
            num_layers=4, num_kv_heads=2, head_dim=128, kv_bits=3, seq_len=n_tokens
        )
        ratio = measured / estimated.compressed_bytes
        assert 0.9 < ratio < 1.1, f"Measured/estimated ratio: {ratio:.3f}"
