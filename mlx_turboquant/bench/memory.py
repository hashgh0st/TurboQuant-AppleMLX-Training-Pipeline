"""Memory benchmarks across sequence lengths and bit-widths.

Uses estimate_memory from memory_accounting — pure calculation, no model loading needed.
"""

from __future__ import annotations

from mlx_turboquant.cache.memory_accounting import MemoryReport, estimate_memory


def benchmark_memory(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_lengths: list[int] | None = None,
    kv_bits_list: list[int] | None = None,
) -> list[MemoryReport]:
    """Run estimate_memory across all (seq_len, kv_bits) combinations.

    Returns a list of MemoryReport, one per (seq_len, kv_bits) pair.
    """
    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096, 8192]
    if kv_bits_list is None:
        kv_bits_list = [2, 3, 4]

    results: list[MemoryReport] = []
    for seq_len in seq_lengths:
        for bits in kv_bits_list:
            report = estimate_memory(
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                kv_bits=bits,
                seq_len=seq_len,
            )
            results.append(report)
    return results
