"""Metal kernel for fused unpack + codebook lookup.

Combines two operations into a single Metal kernel:
  1. Extract bit-field from packed uint32 word
  2. Look up centroid value from codebook

Uses a **per-word** approach: each thread processes one packed uint32 word and
writes all values, with the codebook loaded into threadgroup shared memory.
This reduces thread count and global memory reads vs the per-element approach.
"""

from __future__ import annotations

import mlx.core as mx

from mlx_turboquant.codec.packbits import VALUES_PER_WORD

# Compiled kernel cache (created lazily on first use per bit-width)
_KERNEL_CACHE: dict[int, object] = {}


def _make_shader_source(bits: int) -> str:
    """Generate per-word Metal shader source for a given bit-width.

    Each thread:
      1. Loads codebook into threadgroup shared memory (first threads only)
      2. Reads one packed uint32 word
      3. Extracts all values and writes them via unrolled loop

    Templated from VALUES_PER_WORD and bits to stay in sync with packbits.
    """
    vpw = VALUES_PER_WORD[bits]
    mask = (1 << bits) - 1
    cb_size = 1 << bits

    # Unrolled lookups for each value in the word
    lookups = "\n        ".join(
        f"out[base + {i}] = shared_cb[(packed_word >> {i * bits}) & 0x{mask:X}];"
        for i in range(vpw)
    )

    return f"""
        // Load codebook into threadgroup shared memory (handles threadgroup < cb_size)
        threadgroup float shared_cb[{cb_size}];
        uint tid = thread_position_in_threadgroup.x;
        uint tg_size = threads_per_threadgroup.x;
        for (uint i = tid; i < {cb_size}u; i += tg_size) {{
            shared_cb[i] = codebook[i];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // One thread per packed word: unroll {vpw} lookups
        uint word = thread_position_in_grid.x;
        uint packed_word = packed[word];
        uint base = word * {vpw}u;
        {lookups}
    """


def _get_kernel(bits: int) -> object:
    """Get or create the Metal kernel for the given bit-width."""
    if bits not in _KERNEL_CACHE:
        if bits not in VALUES_PER_WORD:
            msg = f"Unsupported bit-width: {bits}. Must be 2, 3, or 4."
            raise ValueError(msg)
        _KERNEL_CACHE[bits] = mx.fast.metal_kernel(
            name=f"unpack_dequant_{bits}bit",
            input_names=["packed", "codebook"],
            output_names=["out"],
            source=_make_shader_source(bits),
        )
    return _KERNEL_CACHE[bits]


def metal_unpack_dequantize(
    packed: mx.array,
    codebook: mx.array,
    bits: int,
    head_dim: int,
) -> mx.array:
    """Fused unpack + codebook lookup via Metal kernel.

    Input: packed (..., packed_dim) uint32, codebook (2^bits,) float32
    Output: (..., head_dim) float32 — dequantized values

    One thread per packed word. Each thread processes all values in a word.
    """
    batch_shape = packed.shape[:-1]
    packed_dim_val = packed.shape[-1]
    vpw = VALUES_PER_WORD[bits]

    total_words = packed.size
    if total_words == 0:
        return mx.zeros((*batch_shape, head_dim), dtype=mx.float32)

    total_output_elements = total_words * vpw
    packed_flat = packed.reshape(-1)

    kernel = _get_kernel(bits)
    result = kernel(  # type: ignore[operator]
        inputs=[packed_flat, codebook],
        template=[],
        output_shapes=[(total_output_elements,)],
        output_dtypes=[mx.float32],
        grid=(total_words, 1, 1),
        threadgroup=(min(256, max(32, total_words)), 1, 1),
    )

    # Reshape and truncate to head_dim (handles 3-bit padding)
    full: mx.array = result[0].reshape(*batch_shape, packed_dim_val * vpw)
    return full[..., :head_dim]
