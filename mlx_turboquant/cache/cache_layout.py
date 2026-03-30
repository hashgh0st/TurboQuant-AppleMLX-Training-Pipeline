"""Cache configuration and per-layer cache creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    model_name: str | None = None
    calibrated_dir: Path | None = None
    use_qjl: bool = False


def create_cache_layers(config: CacheConfig) -> list[CompressedKVCache]:
    """Create one CompressedKVCache per layer, each sharing the same codec config."""
    key_codec_config = CodecConfig(
        head_dim=config.head_dim,
        bits=config.kv_bits,
        seed=config.seed,
        model_name=config.model_name,
        kv_type="key",
        calibrated_dir=config.calibrated_dir,
        use_qjl=config.use_qjl,
    )
    value_bits = config.kv_bits if config.value_kv_bits is None else config.value_kv_bits
    value_codec_config = CodecConfig(
        head_dim=config.head_dim,
        bits=value_bits,
        seed=config.seed,
        model_name=config.model_name,
        kv_type="value",
        calibrated_dir=config.calibrated_dir,
        # QJL for keys only — values don't need unbiased inner products
    )
    key_codec = Stage1Codec(key_codec_config)
    # Separate codecs when bits differ OR when calibrated (different kv_type codebooks)
    share_codec = value_bits == config.kv_bits and config.model_name is None and not config.use_qjl
    value_codec = key_codec if share_codec else Stage1Codec(value_codec_config)
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
