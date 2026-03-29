"""Cache configuration and per-layer cache creation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mlx_turboquant.cache.compressed_cache import CompressedKVCache
from mlx_turboquant.codec.stage1_codec import CodecConfig, Stage1Codec

BackendKind = Literal["reference", "metal"]


@dataclass
class CacheConfig:
    """Configuration for creating a set of compressed cache layers."""

    num_layers: int
    num_kv_heads: int  # GQA: may be < num_attention_heads
    head_dim: int
    max_seq_len: int
    kv_bits: int
    value_kv_bits: int | None = None
    seed: int = 42
    backend: BackendKind = "reference"
    sink_tokens: int = 0


def create_cache_layers(config: CacheConfig) -> list[CompressedKVCache]:
    """Create one CompressedKVCache per layer, each sharing the same codec config."""
    key_codec_config = CodecConfig(
        head_dim=config.head_dim,
        bits=config.kv_bits,
        seed=config.seed,
    )
    value_bits = config.kv_bits if config.value_kv_bits is None else config.value_kv_bits
    value_codec_config = CodecConfig(
        head_dim=config.head_dim,
        bits=value_bits,
        seed=config.seed,
    )
    key_codec = Stage1Codec(key_codec_config)
    value_codec = key_codec if value_bits == config.kv_bits else Stage1Codec(value_codec_config)
    use_metal = config.backend == "metal"
    return [
        CompressedKVCache(
            key_codec,
            value_codec=value_codec,
            use_metal=use_metal,
            sink_tokens=config.sink_tokens,
        )
        for _ in range(config.num_layers)
    ]
