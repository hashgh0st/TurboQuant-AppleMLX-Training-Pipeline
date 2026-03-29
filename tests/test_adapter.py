"""Tests for MLX-LM adapter — model introspection and cache creation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.nn as nn

from mlx_turboquant.cache.compressed_cache import CompressedKVCache
from mlx_turboquant.integration.mlx_lm_adapter import introspect_model, make_compressed_cache


@dataclass
class _MockModelArgs:
    """Mimics mlx_lm ModelArgs for Qwen2.5-0.5B."""

    hidden_size: int = 896
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768


class _MockModel(nn.Module):
    def __init__(self, args: Any = None) -> None:
        super().__init__()
        self.args = args or _MockModelArgs()


class TestIntrospect:
    def test_extracts_qwen_0_5b_params(self) -> None:
        model = _MockModel()
        info = introspect_model(model)
        assert info.num_layers == 24
        assert info.num_kv_heads == 2
        assert info.head_dim == 896 // 14  # = 64
        assert info.max_seq_len == 32768

    def test_extracts_qwen_7b_params(self) -> None:
        args = _MockModelArgs(
            hidden_size=3584,
            num_hidden_layers=28,
            num_attention_heads=28,
            num_key_value_heads=4,
        )
        model = _MockModel(args)
        info = introspect_model(model)
        assert info.num_layers == 28
        assert info.num_kv_heads == 4
        assert info.head_dim == 128  # 3584 / 28

    def test_default_max_seq_len(self) -> None:
        """Models without max_position_embeddings default to 4096."""

        @dataclass
        class _MinimalArgs:
            hidden_size: int = 128
            num_hidden_layers: int = 2
            num_attention_heads: int = 2
            num_key_value_heads: int = 2

        model = _MockModel(_MinimalArgs())
        info = introspect_model(model)
        assert info.max_seq_len == 4096


class TestMakeCompressedCache:
    def test_correct_layer_count(self) -> None:
        model = _MockModel()
        cache = make_compressed_cache(model, kv_bits=3)
        assert len(cache) == 24

    def test_returns_compressed_caches(self) -> None:
        model = _MockModel()
        cache = make_compressed_cache(model, kv_bits=3)
        assert all(isinstance(c, CompressedKVCache) for c in cache)

    def test_respects_kv_bits(self) -> None:
        model = _MockModel()
        cache = make_compressed_cache(model, kv_bits=4)
        assert cache[0].codec.config.bits == 4

    def test_all_caches_empty(self) -> None:
        model = _MockModel()
        cache = make_compressed_cache(model, kv_bits=3)
        assert all(c.empty() for c in cache)
        assert all(c.offset == 0 for c in cache)
