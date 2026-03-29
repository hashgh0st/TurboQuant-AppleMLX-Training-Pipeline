"""Bit-packing and unpacking for 2, 3, and 4-bit quantized indices.

Packing schemes:
  - 2-bit: 16 values per uint32 (32 bits, no waste)
  - 3-bit: 10 values per uint32 (30 bits used, 2 wasted)
  - 4-bit:  8 values per uint32 (32 bits, no waste)

All operations are fully vectorized using MLX bitwise ops — no Python loops.
"""

from __future__ import annotations

import math

import mlx.core as mx

# Values per uint32 word for each bit-width (public: used by kernels/metal_pack.py)
VALUES_PER_WORD: dict[int, int] = {2: 16, 3: 10, 4: 8}


def packed_dim(head_dim: int, bits: int) -> int:
    """Number of uint32 values needed to pack head_dim indices at given bit-width."""
    vpw = VALUES_PER_WORD[bits]
    return math.ceil(head_dim / vpw)


def pack(indices: mx.array, bits: int) -> mx.array:
    """Pack (..., head_dim) uint8 indices into (..., packed_dim) uint32.

    Each group of values_per_word indices is packed into one uint32 by
    shifting each value to its bit position and OR-ing them together.
    """
    vpw = VALUES_PER_WORD[bits]
    *batch_shape, head_dim_val = indices.shape
    padded_dim = vpw * math.ceil(head_dim_val / vpw)

    # Pad to multiple of values_per_word if needed
    if padded_dim > head_dim_val:
        pad_width = padded_dim - head_dim_val
        indices = mx.pad(indices, [(0, 0)] * len(batch_shape) + [(0, pad_width)])

    # Reshape to (..., num_words, vpw) and cast to uint32
    num_words = padded_dim // vpw
    grouped = indices.reshape(*batch_shape, num_words, vpw).astype(mx.uint32)

    # Create shift amounts: [0, bits, 2*bits, ...] for each position in group
    shifts = mx.array([i * bits for i in range(vpw)], dtype=mx.uint32)

    # Shift each value to its bit position and reduce across the group.
    # Sum is equivalent to OR here because shifts place values in non-overlapping
    # bit positions, so no two values contribute to the same bit.
    shifted = mx.left_shift(grouped, shifts)
    return mx.sum(shifted, axis=-1)


def unpack(packed: mx.array, bits: int, head_dim: int) -> mx.array:
    """Unpack (..., packed_dim) uint32 back to (..., head_dim) uint8.

    Extracts each value by right-shifting and masking with the bit-width mask.
    """
    vpw = VALUES_PER_WORD[bits]
    mask = mx.array((1 << bits) - 1, dtype=mx.uint32)
    *batch_shape, num_words = packed.shape

    # Create shift amounts for extraction
    shifts = mx.array([i * bits for i in range(vpw)], dtype=mx.uint32)

    # Expand packed to (..., num_words, 1) and broadcast with shifts
    expanded = mx.expand_dims(packed, axis=-1)  # (..., num_words, 1)
    shifted = mx.right_shift(expanded, shifts)  # (..., num_words, vpw)
    extracted = mx.bitwise_and(shifted, mask).astype(mx.uint8)

    # Reshape to (..., num_words * vpw) and truncate to head_dim
    flat = extracted.reshape(*batch_shape, num_words * vpw)
    return flat[..., :head_dim]


# --- 1-bit sign packing for QJL ---

SIGNS_PER_WORD = 32  # 1 bit x 32 per uint32


def signs_packed_dim(head_dim: int) -> int:
    """Number of uint32 values needed to pack head_dim sign bits."""
    return math.ceil(head_dim / SIGNS_PER_WORD)


def pack_signs(signs: mx.array) -> mx.array:
    """Pack (..., head_dim) uint8 {0,1} sign bits into (..., packed_dim) uint32."""
    *batch_shape, head_dim_val = signs.shape
    padded_dim = SIGNS_PER_WORD * math.ceil(head_dim_val / SIGNS_PER_WORD)

    if padded_dim > head_dim_val:
        pad_width = padded_dim - head_dim_val
        signs = mx.pad(signs, [(0, 0)] * len(batch_shape) + [(0, pad_width)])

    num_words = padded_dim // SIGNS_PER_WORD
    grouped = signs.reshape(*batch_shape, num_words, SIGNS_PER_WORD).astype(mx.uint32)
    shifts = mx.arange(SIGNS_PER_WORD, dtype=mx.uint32)
    return mx.sum(mx.left_shift(grouped, shifts), axis=-1)


def unpack_signs(packed: mx.array, head_dim: int) -> mx.array:
    """Unpack (..., packed_dim) uint32 to (..., head_dim) float32 signs in {-1, +1}."""
    *batch_shape, num_words = packed.shape
    shifts = mx.arange(SIGNS_PER_WORD, dtype=mx.uint32)
    expanded = mx.expand_dims(packed, axis=-1)
    bits = mx.bitwise_and(mx.right_shift(expanded, shifts), mx.array(1, dtype=mx.uint32))
    flat = bits.reshape(*batch_shape, num_words * SIGNS_PER_WORD)
    # Map {0, 1} → {-1, +1}
    return (2.0 * flat[..., :head_dim].astype(mx.float32) - 1.0)
