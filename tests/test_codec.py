"""Tests for the Stage-1 codec: full encode/decode pipeline."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_turboquant.codec.stage1_codec import CodecConfig, Stage1Codec


@pytest.fixture(params=[64, 128])
def head_dim(request: pytest.FixtureRequest) -> int:
    return request.param  # type: ignore[no-any-return]


def _normalized_mse(original: mx.array, reconstructed: mx.array) -> float:
    """Compute MSE / ||original||^2, averaged over all vectors."""
    mse = mx.mean((original - reconstructed) ** 2, axis=-1)
    norm_sq = mx.mean(original**2, axis=-1)
    ratio = mx.mean(mse / (norm_sq + 1e-10))
    mx.eval(ratio)
    return float(ratio)


class TestShapes:
    def test_encode_decode_shapes(self, head_dim: int) -> None:
        codec = Stage1Codec(CodecConfig(head_dim=head_dim, bits=3))
        x = mx.random.normal((head_dim,))
        ct = codec.encode(x)
        mx.eval(ct.packed, ct.norms)
        assert ct.norms.shape == ()
        decoded = codec.decode(ct)
        mx.eval(decoded)
        assert decoded.shape == (head_dim,)

    def test_batched_shapes(self, head_dim: int) -> None:
        codec = Stage1Codec(CodecConfig(head_dim=head_dim, bits=3))
        x = mx.random.normal((4, 8, head_dim))
        ct = codec.encode(x)
        mx.eval(ct.packed, ct.norms)
        assert ct.norms.shape == (4, 8)
        decoded = codec.decode(ct)
        mx.eval(decoded)
        assert decoded.shape == (4, 8, head_dim)


class TestQuality:
    def test_4bit_mse(self, head_dim: int) -> None:
        codec = Stage1Codec(CodecConfig(head_dim=head_dim, bits=4))
        x = mx.random.normal((100, head_dim))
        reconstructed = codec.encode_decode(x)
        mx.eval(reconstructed)
        nmse = _normalized_mse(x, reconstructed)
        # Theoretical: dim * codebook_distortion ≈ 0.009 for dim=128
        assert nmse < 0.015, f"4-bit NMSE={nmse:.6f} (expected < 0.015)"

    def test_3bit_mse(self, head_dim: int) -> None:
        codec = Stage1Codec(CodecConfig(head_dim=head_dim, bits=3))
        x = mx.random.normal((100, head_dim))
        reconstructed = codec.encode_decode(x)
        mx.eval(reconstructed)
        nmse = _normalized_mse(x, reconstructed)
        # Theoretical: dim * codebook_distortion ≈ 0.033 for dim=128
        assert nmse < 0.05, f"3-bit NMSE={nmse:.6f} (expected < 0.05)"

    def test_2bit_mse(self, head_dim: int) -> None:
        codec = Stage1Codec(CodecConfig(head_dim=head_dim, bits=2))
        x = mx.random.normal((100, head_dim))
        reconstructed = codec.encode_decode(x)
        mx.eval(reconstructed)
        nmse = _normalized_mse(x, reconstructed)
        # Theoretical: dim * codebook_distortion ≈ 0.115 for dim=128
        assert nmse < 0.15, f"2-bit NMSE={nmse:.6f} (expected < 0.15)"

    def test_higher_bits_lower_error(self) -> None:
        x = mx.random.normal((200, 128))
        mse_2 = _normalized_mse(x, Stage1Codec(CodecConfig(128, 2)).encode_decode(x))
        mse_3 = _normalized_mse(x, Stage1Codec(CodecConfig(128, 3)).encode_decode(x))
        mse_4 = _normalized_mse(x, Stage1Codec(CodecConfig(128, 4)).encode_decode(x))
        assert mse_4 < mse_3 < mse_2, f"Expected 4-bit < 3-bit < 2-bit: {mse_4}, {mse_3}, {mse_2}"


class TestInnerProduct:
    def test_inner_product_preservation(self) -> None:
        """Attention depends on inner products: <q, k> ≈ <q, decode(encode(k))>.

        Use cosine similarity of the IP vectors as the metric, which avoids
        blow-up when individual inner products cross zero.
        """
        codec = Stage1Codec(CodecConfig(head_dim=128, bits=3))
        q = mx.random.normal((200, 128))
        k = mx.random.normal((200, 128))
        k_reconstructed = codec.encode_decode(k)
        mx.eval(k_reconstructed)

        ip_original = mx.sum(q * k, axis=-1)
        ip_compressed = mx.sum(q * k_reconstructed, axis=-1)
        mx.eval(ip_original, ip_compressed)

        # Cosine similarity between the two IP vectors (should be very close to 1)
        cos_sim = mx.sum(ip_original * ip_compressed) / (
            mx.linalg.norm(ip_original) * mx.linalg.norm(ip_compressed) + 1e-10
        )
        mx.eval(cos_sim)
        assert float(cos_sim) > 0.95, (
            f"IP cosine similarity: {float(cos_sim):.4f} (expected > 0.95)"
        )

        # Also check normalized absolute error: mean|error| / mean|original|
        abs_error = float(mx.mean(mx.abs(ip_original - ip_compressed)))
        abs_scale = float(mx.mean(mx.abs(ip_original)))
        norm_error = abs_error / (abs_scale + 1e-10)
        assert norm_error < 0.25, f"Normalized IP error: {norm_error:.4f} (expected < 0.25)"


class TestDeterminism:
    def test_same_input_same_output(self) -> None:
        codec = Stage1Codec(CodecConfig(head_dim=128, bits=3))
        x = mx.random.normal((10, 128))
        r1 = codec.encode_decode(x)
        r2 = codec.encode_decode(x)
        mx.eval(r1, r2)
        diff = float(mx.max(mx.abs(r1 - r2)))
        assert diff == 0.0, f"Non-deterministic: max diff = {diff}"


class TestEdgeCases:
    def test_zero_vectors(self, head_dim: int) -> None:
        """Zero vectors should not produce NaN."""
        codec = Stage1Codec(CodecConfig(head_dim=head_dim, bits=3))
        x = mx.zeros((5, head_dim))
        reconstructed = codec.encode_decode(x)
        mx.eval(reconstructed)
        assert not bool(mx.any(mx.isnan(reconstructed))), "NaN in zero vector reconstruction"

    def test_single_vector(self, head_dim: int) -> None:
        """Should work on a single unbatched vector."""
        codec = Stage1Codec(CodecConfig(head_dim=head_dim, bits=3))
        x = mx.random.normal((head_dim,))
        reconstructed = codec.encode_decode(x)
        mx.eval(reconstructed)
        assert reconstructed.shape == (head_dim,)
