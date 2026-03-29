"""Memory accounting: formula-based estimates and actual measurement.

This module owns the calculation primitives. bench/memory.py owns iteration
and reporting over different configurations.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlx_turboquant.cache.compressed_cache import CompressedKVCache
from mlx_turboquant.codec.packbits import packed_dim


@dataclass
class MemoryReport:
    """Memory comparison between baseline fp16 and compressed KV cache."""

    baseline_bytes: int
    compressed_bytes: int
    compression_ratio: float
    bytes_per_token_baseline: int
    bytes_per_token_compressed: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    kv_bits: int
    seq_len: int
    value_kv_bits: int | None = None
    sink_tokens: int = 0
    cache_mode: str | None = None


def estimate_memory(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    kv_bits: int,
    seq_len: int,
    value_kv_bits: int | None = None,
    sink_tokens: int = 0,
) -> MemoryReport:
    """Formula-based memory estimate for baseline vs compressed cache.

    Baseline fp16: 2 (K+V) * layers * kv_heads * head_dim * 2 bytes * seq_len
    Compressed: sink tokens in FP16 + remaining tokens compressed.
    """
    # Baseline: fp16 storage for both K and V
    bpt_baseline = 2 * num_kv_heads * head_dim * 2  # bytes per token per layer (K+V, fp16)
    baseline = bpt_baseline * num_layers * seq_len

    # Sink tokens: stored in FP16 (same cost as baseline)
    actual_sink = min(sink_tokens, seq_len)
    sink_bytes = bpt_baseline * num_layers * actual_sink

    compressed_tokens = seq_len - actual_sink
    value_bits = kv_bits if value_kv_bits is None else value_kv_bits
    key_packed_bytes = packed_dim(head_dim, kv_bits) * 4  # uint32 = 4 bytes each
    value_packed_bytes = packed_dim(head_dim, value_bits) * 4
    norm_bytes = 4  # float32 corrected norms
    bpt_compressed = num_kv_heads * (
        (key_packed_bytes + norm_bytes) + (value_packed_bytes + norm_bytes)
    )
    compressed_portion = bpt_compressed * num_layers * compressed_tokens

    total_compressed = sink_bytes + compressed_portion
    ratio = baseline / total_compressed if total_compressed > 0 else float("inf")

    # Bytes per token across all layers (weighted average including sink)
    bpt_all_layers = total_compressed // seq_len if seq_len > 0 else bpt_compressed * num_layers

    return MemoryReport(
        baseline_bytes=baseline,
        compressed_bytes=total_compressed,
        compression_ratio=ratio,
        bytes_per_token_baseline=bpt_baseline * num_layers,
        bytes_per_token_compressed=bpt_all_layers,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_bits=kv_bits,
        value_kv_bits=value_bits,
        sink_tokens=actual_sink,
        seq_len=seq_len,
    )


def measure_memory(cache_layers: list[CompressedKVCache]) -> int:
    """Sum actual compressed bytes across all cache layers."""
    return sum(c.nbytes for c in cache_layers)
