"""Model introspection and compressed cache creation for MLX-LM models.

Reads model.args (a ModelArgs dataclass set by BaseModelArgs) to extract
architecture parameters, then creates CompressedKVCache layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlx_turboquant.cache.cache_layout import CacheConfig, create_cache_layers
from mlx_turboquant.cache.compressed_cache import CompressedKVCache


@dataclass
class ModelInfo:
    """Architecture parameters extracted from a loaded MLX-LM model."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int


def introspect_model(model: Any) -> ModelInfo:
    """Auto-detect model architecture parameters from model.args.

    Works with Qwen2, Qwen3, Llama, Mistral, and any MLX-LM model
    that stores a ModelArgs dataclass at model.args.
    """
    args = model.args
    head_dim = args.hidden_size // args.num_attention_heads
    return ModelInfo(
        num_layers=args.num_hidden_layers,
        num_kv_heads=args.num_key_value_heads,
        head_dim=head_dim,
        max_seq_len=getattr(args, "max_position_embeddings", 4096),
    )


def make_compressed_cache(
    model: Any,
    *,
    kv_bits: int = 3,
    max_seq_len: int = 4096,
    seed: int = 42,
) -> list[CompressedKVCache]:
    """Create compressed cache layers for a loaded MLX-LM model.

    Analogous to mlx_lm.models.cache.make_prompt_cache() but returns
    CompressedKVCache instances instead of KVCache.
    """
    info = introspect_model(model)
    config = CacheConfig(
        num_layers=info.num_layers,
        num_kv_heads=info.num_kv_heads,
        head_dim=info.head_dim,
        max_seq_len=max_seq_len,
        kv_bits=kv_bits,
        seed=seed,
    )
    return create_cache_layers(config)
