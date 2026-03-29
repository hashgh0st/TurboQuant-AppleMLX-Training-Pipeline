"""Metal kernel for fused unpack + codebook lookup.

Combines two operations into a single Metal kernel per element:
  1. Extract bit-field from packed uint32 word
  2. Look up centroid value from codebook

This eliminates the intermediate uint8 index array and halves memory bandwidth
compared to the reference path (unpack -> codebook[indices]).
"""

from __future__ import annotations

import mlx.core as mx

from mlx_turboquant.codec.packbits import VALUES_PER_WORD

# Compiled kernel cache (created lazily on first use per bit-width)
_KERNEL_CACHE: dict[int, object] = {}


def _make_shader_source(bits: int) -> str:
    """Generate Metal shader source for a given bit-width.

    Templated from VALUES_PER_WORD and bits to avoid copy-pasting constants
    across shader variants.
    """
    vpw = VALUES_PER_WORD[bits]
    mask = (1 << bits) - 1
    return f"""
        uint elem = thread_position_in_grid.x;
        uint word_idx = elem / {vpw};
        uint bit_offset = (elem % {vpw}) * {bits};
        uint index = (packed[word_idx] >> bit_offset) & 0x{mask:X};
        out[elem] = codebook[index];
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

    One thread per output element. Flattens input, runs kernel, reshapes.
    """
    batch_shape = packed.shape[:-1]
    packed_dim_val = packed.shape[-1]
    vpw = VALUES_PER_WORD[bits]

    # Compute total output elements without math.prod — use packed.size directly
    total_output_elements = (packed.size // packed_dim_val) * packed_dim_val * vpw

    packed_flat = packed.reshape(-1)

    kernel = _get_kernel(bits)
    result = kernel(  # type: ignore[operator]
        inputs=[packed_flat, codebook],
        template=[],
        output_shapes=[(total_output_elements,)],
        output_dtypes=[mx.float32],
        grid=(total_output_elements, 1, 1),
        threadgroup=(min(256, total_output_elements), 1, 1),
    )

    # Reshape and truncate to head_dim (handles 3-bit padding)
    full: mx.array = result[0].reshape(*batch_shape, packed_dim_val * vpw)
    return full[..., :head_dim]
