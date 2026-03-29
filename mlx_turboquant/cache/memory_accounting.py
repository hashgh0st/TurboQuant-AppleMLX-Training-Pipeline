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


def estimate_memory(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    kv_bits: int,
    seq_len: int,
) -> MemoryReport:
    """Formula-based memory estimate for baseline vs compressed cache.

    Baseline fp16: 2 (K+V) * layers * kv_heads * head_dim * 2 bytes * seq_len
    Compressed: 2 (K+V) * layers * kv_heads * (packed_bytes + norm_bytes) * seq_len
    """
    # Baseline: fp16 storage for both K and V
    bpt_baseline = 2 * num_kv_heads * head_dim * 2  # bytes per token per layer (K+V, fp16)
    baseline = bpt_baseline * num_layers * seq_len

    # Compressed: packed uint32 + float16 norm per vector, for K and V
    packed_bytes = packed_dim(head_dim, kv_bits) * 4  # uint32 = 4 bytes each
    norm_bytes = 2  # float16
    bpt_compressed = 2 * num_kv_heads * (packed_bytes + norm_bytes)
    compressed = bpt_compressed * num_layers * seq_len

    ratio = baseline / compressed if compressed > 0 else float("inf")

    return MemoryReport(
        baseline_bytes=baseline,
        compressed_bytes=compressed,
        compression_ratio=ratio,
        bytes_per_token_baseline=bpt_baseline * num_layers,
        bytes_per_token_compressed=bpt_compressed * num_layers,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_bits=kv_bits,
        seq_len=seq_len,
    )


def measure_memory(cache_layers: list[CompressedKVCache]) -> int:
    """Sum actual compressed bytes across all cache layers."""
    return sum(c.nbytes for c in cache_layers)
