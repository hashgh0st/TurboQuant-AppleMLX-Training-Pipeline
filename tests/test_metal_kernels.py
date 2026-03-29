"""Tests for Metal kernel fused unpack + dequantize."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_turboquant.codec.codebooks import load_codebook
from mlx_turboquant.codec.packbits import pack, unpack
from mlx_turboquant.kernels.metal_pack import metal_unpack_dequantize


def _reference_unpack_dequant(
    packed: mx.array, codebook: mx.array, bits: int, head_dim: int
) -> mx.array:
    """Reference path: unpack then codebook lookup."""
    indices = unpack(packed, bits, head_dim)
    return codebook[indices]


@pytest.fixture(params=[64, 128])
def head_dim(request: pytest.FixtureRequest) -> int:
    return request.param  # type: ignore[no-any-return]


class TestParity:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_matches_reference(self, bits: int, head_dim: int) -> None:
        """Metal kernel output must match pure-MLX unpack + codebook[indices]."""
        cb = load_codebook(head_dim, bits)
        codebook = mx.array(cb.centroids, dtype=mx.float32)

        max_val = (1 << bits) - 1
        indices = (mx.random.uniform(shape=(head_dim,)) * (max_val + 1)).astype(mx.uint8)
        indices = mx.clip(indices, 0, max_val)
        packed = pack(indices, bits)

        metal_result = metal_unpack_dequantize(packed, codebook, bits, head_dim)
        ref_result = _reference_unpack_dequant(packed, codebook, bits, head_dim)
        mx.eval(metal_result, ref_result)

        diff = float(mx.max(mx.abs(metal_result - ref_result)))
        assert diff == 0.0, f"bits={bits}, head_dim={head_dim}: max diff={diff}"

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_batched_matches_reference(self, bits: int, head_dim: int) -> None:
        """Metal kernel works on batched inputs."""
        cb = load_codebook(head_dim, bits)
        codebook = mx.array(cb.centroids, dtype=mx.float32)

        max_val = (1 << bits) - 1
        indices = (mx.random.uniform(shape=(4, 8, head_dim)) * (max_val + 1)).astype(mx.uint8)
        indices = mx.clip(indices, 0, max_val)
        packed = pack(indices, bits)

        metal_result = metal_unpack_dequantize(packed, codebook, bits, head_dim)
        ref_result = _reference_unpack_dequant(packed, codebook, bits, head_dim)
        mx.eval(metal_result, ref_result)

        diff = float(mx.max(mx.abs(metal_result - ref_result)))
        assert diff == 0.0, f"bits={bits}: batched max diff={diff}"


class TestShapes:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_output_shape(self, bits: int, head_dim: int) -> None:
        cb = load_codebook(head_dim, bits)
        codebook = mx.array(cb.centroids, dtype=mx.float32)
        packed = pack(mx.zeros((head_dim,), dtype=mx.uint8), bits)
        result = metal_unpack_dequantize(packed, codebook, bits, head_dim)
        mx.eval(result)
        assert result.shape == (head_dim,)

    def test_batched_shape(self) -> None:
        cb = load_codebook(128, 3)
        codebook = mx.array(cb.centroids, dtype=mx.float32)
        packed = pack(mx.zeros((2, 3, 128), dtype=mx.uint8), 3)
        result = metal_unpack_dequantize(packed, codebook, 3, 128)
        mx.eval(result)
        assert result.shape == (2, 3, 128)


class TestEdgeCases:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_all_zeros(self, bits: int) -> None:
        cb = load_codebook(128, bits)
        codebook = mx.array(cb.centroids, dtype=mx.float32)
        packed = pack(mx.zeros((128,), dtype=mx.uint8), bits)
        result = metal_unpack_dequantize(packed, codebook, bits, 128)
        ref = _reference_unpack_dequant(packed, codebook, bits, 128)
        mx.eval(result, ref)
        assert float(mx.max(mx.abs(result - ref))) == 0.0

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_all_max(self, bits: int) -> None:
        cb = load_codebook(128, bits)
        codebook = mx.array(cb.centroids, dtype=mx.float32)
        max_val = (1 << bits) - 1
        packed = pack(mx.full((128,), max_val, dtype=mx.uint8), bits)
        result = metal_unpack_dequantize(packed, codebook, bits, 128)
        ref = _reference_unpack_dequant(packed, codebook, bits, 128)
        mx.eval(result, ref)
        assert float(mx.max(mx.abs(result - ref))) == 0.0
