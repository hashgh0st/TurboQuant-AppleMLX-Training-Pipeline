"""Tests for bit-packing and unpacking."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_turboquant.codec.packbits import pack, packed_dim, unpack


class TestPackedDim:
    def test_2bit(self) -> None:
        assert packed_dim(128, 2) == 8  # 128 / 16 = 8
        assert packed_dim(64, 2) == 4  # 64 / 16 = 4

    def test_3bit(self) -> None:
        assert packed_dim(128, 3) == 13  # ceil(128/10) = 13
        assert packed_dim(64, 3) == 7  # ceil(64/10) = 7

    def test_4bit(self) -> None:
        assert packed_dim(128, 4) == 16  # 128 / 8 = 16
        assert packed_dim(64, 4) == 8  # 64 / 8 = 8


class TestRoundtrip:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_exact_recovery(self, bits: int, head_dim: int) -> None:
        """pack then unpack should recover exact original indices."""
        max_val = (1 << bits) - 1
        indices = (mx.random.uniform(shape=(head_dim,)) * (max_val + 1)).astype(mx.uint8)
        indices = mx.clip(indices, 0, max_val)

        p = pack(indices, bits)
        recovered = unpack(p, bits, head_dim)
        mx.eval(recovered)

        assert recovered.shape == indices.shape
        diff = int(mx.sum(indices != recovered))
        assert diff == 0, f"bits={bits}, head_dim={head_dim}: {diff} mismatches"

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_batched(self, bits: int) -> None:
        """Pack/unpack should work on batched inputs."""
        head_dim = 128
        max_val = (1 << bits) - 1
        batch = (mx.random.uniform(shape=(4, 8, head_dim)) * (max_val + 1)).astype(mx.uint8)
        batch = mx.clip(batch, 0, max_val)

        p = pack(batch, bits)
        recovered = unpack(p, bits, head_dim)
        mx.eval(recovered)

        assert recovered.shape == batch.shape
        diff = int(mx.sum(batch != recovered))
        assert diff == 0, f"bits={bits}: {diff} mismatches in batch"


class TestPackedShape:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_output_shape(self, bits: int) -> None:
        head_dim = 128
        indices = mx.zeros((head_dim,), dtype=mx.uint8)
        p = pack(indices, bits)
        mx.eval(p)
        assert p.shape == (packed_dim(head_dim, bits),)
        assert p.dtype == mx.uint32

    def test_batched_shape(self) -> None:
        indices = mx.zeros((2, 3, 128), dtype=mx.uint8)
        p = pack(indices, 3)
        mx.eval(p)
        assert p.shape == (2, 3, 13)


class TestEdgeCases:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_all_zeros(self, bits: int) -> None:
        indices = mx.zeros((128,), dtype=mx.uint8)
        p = pack(indices, bits)
        recovered = unpack(p, bits, 128)
        mx.eval(recovered)
        assert int(mx.sum(recovered)) == 0

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_all_max(self, bits: int) -> None:
        max_val = (1 << bits) - 1
        indices = mx.full((128,), max_val, dtype=mx.uint8)
        p = pack(indices, bits)
        recovered = unpack(p, bits, 128)
        mx.eval(recovered)
        assert int(mx.sum(recovered != max_val)) == 0

    def test_3bit_padding_dim_not_multiple_of_10(self) -> None:
        """head_dim=128 is not a multiple of 10 — packing must handle padding."""
        indices = mx.array(list(range(8)) * 16, dtype=mx.uint8)[:128]
        assert indices.shape == (128,)
        p = pack(indices, 3)
        recovered = unpack(p, 3, 128)
        mx.eval(recovered)
        diff = int(mx.sum(indices != recovered))
        assert diff == 0


class TestCompressionRatio:
    def test_3bit_dim128_size(self) -> None:
        """3-bit packing of 128 values should produce 13 uint32s = 52 bytes."""
        indices = mx.zeros((128,), dtype=mx.uint8)
        p = pack(indices, 3)
        mx.eval(p)
        assert p.shape == (13,)
