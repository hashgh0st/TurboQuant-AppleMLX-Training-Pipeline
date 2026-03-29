"""Tests for cache layout configuration and factory."""

from __future__ import annotations

from mlx_turboquant.cache.cache_layout import CacheConfig, create_cache_layers
from mlx_turboquant.cache.compressed_cache import CompressedKVCache


class TestCreateCacheLayers:
    def test_correct_count(self) -> None:
        config = CacheConfig(
            num_layers=36, num_kv_heads=2, head_dim=128,
            max_seq_len=4096, kv_bits=3,
        )
        layers = create_cache_layers(config)
        assert len(layers) == 36

    def test_returns_compressed_caches(self) -> None:
        config = CacheConfig(
            num_layers=4, num_kv_heads=2, head_dim=128,
            max_seq_len=4096, kv_bits=3,
        )
        layers = create_cache_layers(config)
        assert all(isinstance(c, CompressedKVCache) for c in layers)

    def test_all_empty_initially(self) -> None:
        config = CacheConfig(
            num_layers=4, num_kv_heads=2, head_dim=64,
            max_seq_len=4096, kv_bits=4,
        )
        layers = create_cache_layers(config)
        assert all(c.empty() for c in layers)
        assert all(c.offset == 0 for c in layers)
