"""Stage-1 TurboQuant-inspired codec: normalize → rotate → quantize → pack.

This is the core encode/decode pipeline. It is fully vectorized with no Python
loops on the hot path. The encode pipeline:
  1. Compute per-vector L2 norms
  2. Normalize to unit vectors
  3. Apply randomized Hadamard rotation (data-oblivious)
  4. Scalar-quantize each coordinate using precomputed Lloyd-Max codebook
  5. Pack indices into uint32 arrays

The decode pipeline reverses this: unpack → dequantize → inverse rotate → rescale.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from mlx_turboquant.codec.codebooks import load_codebook
from mlx_turboquant.codec.packbits import pack, unpack
from mlx_turboquant.codec.transforms import (
    TransformState,
    create_transform,
    forward_transform,
    inverse_transform,
)


@dataclass
class CodecConfig:
    """Configuration for the Stage-1 codec."""

    head_dim: int
    bits: int
    seed: int = 42


@dataclass
class CompressedTensor:
    """Compressed representation of KV vectors."""

    packed: mx.array  # (..., packed_dim) uint32 — quantized indices
    norms: mx.array  # (...,) float16 — per-vector L2 norms
    config: CodecConfig


class Stage1Codec:
    """Stage-1 KV-cache compression codec.

    Uses randomized Hadamard rotation + Lloyd-Max optimal codebooks to compress
    vectors to 2-4 bits per coordinate. All operations are fully vectorized.
    """

    def __init__(self, config: CodecConfig) -> None:
        self.config = config
        cb = load_codebook(config.head_dim, config.bits)
        self.centroids = mx.array(cb.centroids, dtype=mx.float32)
        self.boundaries = mx.array(cb.boundaries, dtype=mx.float32)
        self.transform: TransformState = create_transform(config.head_dim, config.seed)

        # Decision thresholds are the internal boundaries (already optimal from Lloyd-Max).
        # boundaries has 2^bits+1 entries: [-1, b1, b2, ..., 1]
        # Internal boundaries [b1, ..., b_{2^bits-1}] are the decision points.
        self._thresholds = self.boundaries[1:-1]
        self._max_index = (1 << config.bits) - 1

    def encode(self, x: mx.array) -> CompressedTensor:
        """Compress KV vectors. Fully vectorized, no Python loops.

        Input: (..., head_dim) float16 or float32
        Returns: CompressedTensor with packed indices and norms
        """
        # 1. Compute per-vector L2 norms
        norms = mx.linalg.norm(x, axis=-1)

        # 2. Normalize to unit vectors (guard against zero vectors)
        x_normed = x / (norms[..., None] + 1e-8)

        # 3. Randomized Hadamard rotation
        x_rot = forward_transform(x_normed, self.transform)

        # 4. Quantize: count how many thresholds each coordinate exceeds.
        # This gives the bin index directly — the sum of (2^bits - 1) booleans
        # is bounded to [0, 2^bits - 1], so no clipping is needed.
        indices = mx.sum(x_rot[..., None] > self._thresholds, axis=-1).astype(mx.uint8)

        # 5. Pack into uint32
        packed = pack(indices, self.config.bits)

        return CompressedTensor(
            packed=packed,
            norms=norms.astype(mx.float16),
            config=self.config,
        )

    def decode(self, ct: CompressedTensor) -> mx.array:
        """Decompress to float32. Fully vectorized, no Python loops.

        Input: CompressedTensor (must have been encoded by a codec with matching config)
        Returns: (..., head_dim) float32
        """
        if ct.config != self.config:
            msg = f"Config mismatch: codec={self.config!r}, tensor={ct.config!r}"
            raise ValueError(msg)

        # 1. Unpack indices
        indices = unpack(ct.packed, ct.config.bits, ct.config.head_dim)

        # 2. Dequantize: map indices to centroid values via fancy indexing
        x_rot = self.centroids[indices]

        # 3. Inverse Hadamard rotation
        x_normed = inverse_transform(x_rot, self.transform)

        # 4. Rescale by original norms
        return x_normed * ct.norms.astype(mx.float32)[..., None]

    def encode_decode(self, x: mx.array) -> mx.array:
        """Round-trip compression for quality measurement."""
        return self.decode(self.encode(x))
