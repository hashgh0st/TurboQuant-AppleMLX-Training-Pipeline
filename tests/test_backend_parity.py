"""Tests for Metal vs reference backend parity."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_turboquant.cache.cache_layout import CacheConfig, create_cache_layers
from mlx_turboquant.codec.stage1_codec import CodecConfig, Stage1Codec


class TestCodecParity:
    """Metal decode must match reference decode for the same compressed input."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_decode_matches(self, bits: int, head_dim: int) -> None:
        codec = Stage1Codec(CodecConfig(head_dim=head_dim, bits=bits))
        x = mx.random.normal((10, head_dim))
        ct = codec.encode(x)

        ref = codec.decode(ct, use_metal=False)
        metal = codec.decode(ct, use_metal=True)
        mx.eval(ref, metal)

        diff = float(mx.max(mx.abs(ref - metal)))
        assert diff == 0.0, f"bits={bits}, head_dim={head_dim}: max diff={diff}"

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_encode_decode_metal(self, bits: int) -> None:
        """Full round-trip with Metal decode produces valid output."""
        codec = Stage1Codec(CodecConfig(head_dim=128, bits=bits))
        x = mx.random.normal((50, 128))
        reconstructed = codec.encode_decode(x, use_metal=True)
        mx.eval(reconstructed)
        mse = float(mx.mean((x - reconstructed) ** 2))
        norm_sq = float(mx.mean(x**2))
        nmse = mse / (norm_sq + 1e-10)
        assert nmse < 0.15, f"Metal encode_decode NMSE={nmse:.4f}"


class TestCacheParity:
    """CompressedKVCache with Metal backend must match reference backend."""

    @pytest.mark.parametrize("bits", [3, 4])
    def test_cache_output_matches(self, bits: int) -> None:
        config_ref = CacheConfig(
            num_layers=1,
            num_kv_heads=2,
            head_dim=128,
            max_seq_len=4096,
            kv_bits=bits,
            backend="reference",
        )
        config_metal = CacheConfig(
            num_layers=1,
            num_kv_heads=2,
            head_dim=128,
            max_seq_len=4096,
            kv_bits=bits,
            backend="metal",
        )
        cache_ref = create_cache_layers(config_ref)[0]
        cache_metal = create_cache_layers(config_metal)[0]

        keys = mx.random.normal((1, 2, 10, 128))
        values = mx.random.normal((1, 2, 10, 128))

        k_ref, v_ref = cache_ref.update_and_fetch(keys, values)
        k_metal, v_metal = cache_metal.update_and_fetch(keys, values)
        mx.eval(k_ref, v_ref, k_metal, v_metal)

        k_diff = float(mx.max(mx.abs(k_ref - k_metal)))
        v_diff = float(mx.max(mx.abs(v_ref - v_metal)))
        assert k_diff == 0.0, f"Key diff: {k_diff}"
        assert v_diff == 0.0, f"Value diff: {v_diff}"

    def test_metal_cache_has_use_metal_flag(self) -> None:
        config = CacheConfig(
            num_layers=2,
            num_kv_heads=2,
            head_dim=64,
            max_seq_len=4096,
            kv_bits=3,
            backend="metal",
        )
        layers = create_cache_layers(config)
        assert all(c.use_metal for c in layers)

    def test_reference_cache_no_metal_flag(self) -> None:
        config = CacheConfig(
            num_layers=2,
            num_kv_heads=2,
            head_dim=64,
            max_seq_len=4096,
            kv_bits=3,
            backend="reference",
        )
        layers = create_cache_layers(config)
        assert all(not c.use_metal for c in layers)
