"""TurboQuant codec: normalize → rotate → quantize → pack, with optional QJL.

Stage 1 (MSE-optimal): normalize, randomized Hadamard rotation, per-coordinate
Lloyd-Max quantization, bit-packing. Fast but **biased** for inner products.

Stage 2 (QJL correction): computes the residual from Stage 1, projects through
a random matrix S, stores 1-bit signs + residual norm. During decode, the QJL
correction is added back: x_hat = x_hat_mse + sqrt(pi/2)/d * gamma * S^T * signs.
This makes inner product estimation **unbiased**, fixing attention score errors.

When ``use_qjl=True``, "b-bit" means (b-1) bits for MSE + 1 bit for QJL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

from mlx_turboquant.codec.codebooks import KVType, load_codebook_with_fallback
from mlx_turboquant.codec.packbits import pack, pack_signs, unpack, unpack_signs
from mlx_turboquant.codec.transforms import (
    TransformState,
    create_transform,
    forward_transform,
    inverse_transform,
)
from mlx_turboquant.kernels.metal_pack import metal_unpack_dequantize

_QJL_SCALE = math.sqrt(math.pi / 2)


@dataclass
class CodecConfig:
    """Configuration for the codec."""

    head_dim: int
    bits: int
    seed: int = 42
    model_name: str | None = None
    kv_type: KVType = "key"
    calibrated_dir: Path | None = None
    use_qjl: bool = False
    qjl_seed: int = 137


@dataclass
class CompressedTensor:
    """Compressed representation of KV vectors."""

    packed: mx.array  # (..., packed_dim) uint32 — MSE quantized indices
    norms: mx.array  # (...,) float16 — per-vector L2 norms
    config: CodecConfig
    qjl_packed: mx.array | None = field(default=None, repr=False)  # (..., signs_packed_dim) uint32
    residual_norms: mx.array | None = field(default=None, repr=False)  # (...,) float16


class Stage1Codec:
    """TurboQuant KV-cache compression codec.

    Stage 1: randomized Hadamard rotation + Lloyd-Max codebooks (MSE-optimal).
    Stage 2 (when ``use_qjl=True``): QJL residual correction for unbiased inner products.
    """

    def __init__(self, config: CodecConfig) -> None:
        self.config = config

        # MSE stage uses (bits-1) when QJL is enabled, full bits otherwise
        mse_bits = config.bits - 1 if config.use_qjl else config.bits
        if mse_bits < 1:
            msg = f"QJL requires bits >= 2 (got {config.bits})"
            raise ValueError(msg)

        cb = load_codebook_with_fallback(
            config.head_dim,
            mse_bits,
            kv_type=config.kv_type,
            model_name=config.model_name,
            calibrated_dir=config.calibrated_dir,
        )
        self.centroids = mx.array(cb.centroids, dtype=mx.float32)
        self.boundaries = mx.array(cb.boundaries, dtype=mx.float32)
        self.transform: TransformState = create_transform(config.head_dim, config.seed)
        self._mse_bits = mse_bits
        self._thresholds = self.boundaries[1:-1]
        self._max_index = (1 << mse_bits) - 1

        # QJL random projection matrix (shared across all vectors)
        self._s_matrix: mx.array | None = None
        self._qjl_decode_scale = 0.0
        if config.use_qjl:
            key = mx.random.key(config.qjl_seed)
            self._s_matrix = mx.random.normal(key=key, shape=(config.head_dim, config.head_dim))
            self._qjl_decode_scale = _QJL_SCALE / config.head_dim

    @property
    def mse_bits(self) -> int:
        """Effective MSE bit-width (bits-1 when QJL is on)."""
        return self._mse_bits

    def encode(self, x: mx.array) -> CompressedTensor:
        """Compress KV vectors. Fully vectorized, no Python loops.

        Input: (..., head_dim) float16 or float32
        Returns: CompressedTensor with packed indices, norms, and optional QJL data
        """
        # float32 to avoid norm overflow on RoPE'd keys
        x_f32 = x.astype(mx.float32) if x.dtype != mx.float32 else x
        norms = mx.linalg.norm(x_f32, axis=-1)
        x_normed = x_f32 / (norms[..., None] + 1e-8)
        x_rot = forward_transform(x_normed, self.transform)

        # MSE quantize (at mse_bits, which is bits-1 when QJL is on)
        indices = mx.sum(x_rot[..., None] > self._thresholds, axis=-1).astype(mx.uint8)
        packed = pack(indices, self._mse_bits)

        qjl_packed = None
        residual_norms = None

        if self.config.use_qjl:
            assert self._s_matrix is not None
            # Residual in rotated domain, then inverse-rotate
            x_rot_mse = self.centroids[indices]
            r_rot = x_rot - x_rot_mse
            r = inverse_transform(r_rot, self.transform)
            # QJL: project through S, take sign
            proj = mx.matmul(r, self._s_matrix.T)
            signs = (proj >= 0).astype(mx.uint8)
            qjl_packed = pack_signs(signs)
            residual_norms = mx.linalg.norm(r, axis=-1).astype(mx.float16)

        return CompressedTensor(
            packed=packed,
            norms=norms.astype(mx.float16),
            config=self.config,
            qjl_packed=qjl_packed,
            residual_norms=residual_norms,
        )

    def decode(self, ct: CompressedTensor, *, use_metal: bool = False) -> mx.array:
        """Decompress to float32. Fully vectorized, no Python loops.

        When QJL is enabled, adds the correction: sqrt(pi/2)/d * gamma * S^T * signs
        to make inner product estimation unbiased.
        """
        if use_metal and not self.config.use_qjl:
            x_rot = metal_unpack_dequantize(
                ct.packed, self.centroids, self._mse_bits, ct.config.head_dim
            )
        else:
            indices = unpack(ct.packed, self._mse_bits, ct.config.head_dim)
            x_rot = self.centroids[indices]

        x_normed = inverse_transform(x_rot, self.transform)

        # QJL correction: x_hat = x_hat_mse + sqrt(pi/2)/d * gamma * S^T * signs
        if self.config.use_qjl and ct.qjl_packed is not None and ct.residual_norms is not None:
            assert self._s_matrix is not None
            signs = unpack_signs(ct.qjl_packed, ct.config.head_dim)
            # S^T @ signs: (..., d) @ (d, d)^T = (..., d)
            correction = mx.matmul(signs, self._s_matrix)
            x_normed = x_normed + self._qjl_decode_scale * ct.residual_norms.astype(
                mx.float32
            )[..., None] * correction

        return x_normed * ct.norms.astype(mx.float32)[..., None]

    def encode_decode(self, x: mx.array, *, use_metal: bool = False) -> mx.array:
        """Round-trip compression for quality measurement."""
        return self.decode(self.encode(x), use_metal=use_metal)
