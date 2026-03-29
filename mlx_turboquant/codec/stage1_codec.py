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
from pathlib import Path

import mlx.core as mx

from mlx_turboquant.codec.codebooks import KVType, load_codebook_with_fallback
from mlx_turboquant.codec.packbits import pack, unpack
from mlx_turboquant.codec.transforms import (
    TransformState,
    create_transform,
    forward_transform,
    inverse_transform,
)
from mlx_turboquant.kernels.metal_pack import metal_unpack_dequantize


@dataclass
class CodecConfig:
    """Configuration for the Stage-1 codec."""

    head_dim: int
    bits: int
    seed: int = 42
    model_name: str | None = None
    kv_type: KVType = "key"
    calibrated_dir: Path | None = None


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
        cb = load_codebook_with_fallback(
            config.head_dim,
            config.bits,
            kv_type=config.kv_type,
            model_name=config.model_name,
            calibrated_dir=config.calibrated_dir,
        )
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
        # 1. Compute in float32 to avoid norm overflow in float16
        # (float16 sum-of-squares overflows for large activations like RoPE'd keys)
        x_f32 = x.astype(mx.float32) if x.dtype != mx.float32 else x
        norms = mx.linalg.norm(x_f32, axis=-1)

        # 2. Normalize to unit vectors (guard against zero vectors)
        x_normed = x_f32 / (norms[..., None] + 1e-8)

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

    def decode(self, ct: CompressedTensor, *, use_metal: bool = False) -> mx.array:
        """Decompress to float32. Fully vectorized, no Python loops.

        Input: CompressedTensor (must have been encoded by a codec with matching config)
        Returns: (..., head_dim) float32

        When use_metal=True, steps 1-2 (unpack+dequant) use a fused Metal kernel
        for ~1.6x speedup over the pure-MLX reference path.
        """
        if ct.config != self.config:
            msg = f"Config mismatch: codec={self.config!r}, tensor={ct.config!r}"
            raise ValueError(msg)

        if use_metal:
            x_rot = metal_unpack_dequantize(
                ct.packed, self.centroids, ct.config.bits, ct.config.head_dim
            )
        else:
            # Reference path (steps 1-2)
            indices = unpack(ct.packed, ct.config.bits, ct.config.head_dim)
            x_rot = self.centroids[indices]

        # 3. Inverse Hadamard rotation
        x_normed = inverse_transform(x_rot, self.transform)

        # 4. Rescale by original norms
        return x_normed * ct.norms.astype(mx.float32)[..., None]

    def encode_decode(self, x: mx.array, *, use_metal: bool = False) -> mx.array:
        """Round-trip compression for quality measurement."""
        return self.decode(self.encode(x), use_metal=use_metal)
