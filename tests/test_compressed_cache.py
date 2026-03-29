"""Tests for CompressedKVCache protocol compliance and correctness."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_turboquant.cache.compressed_cache import CompressedKVCache
from mlx_turboquant.codec.stage1_codec import CodecConfig, Stage1Codec

# Qwen 2.5 GQA config: 2 KV heads, head_dim 128
HEAD_DIM = 128
KV_HEADS = 2
BITS = 3
B = 1  # batch size


@pytest.fixture()
def cache() -> CompressedKVCache:
    codec = Stage1Codec(CodecConfig(head_dim=HEAD_DIM, bits=BITS))
    return CompressedKVCache(codec)


def _random_kv(n_steps: int) -> tuple[mx.array, mx.array]:
    """Generate random KV tensors matching Qwen shape."""
    shape = (B, KV_HEADS, n_steps, HEAD_DIM)
    return mx.random.normal(shape), mx.random.normal(shape)


class TestUpdateAndFetch:
    def test_single_token(self, cache: CompressedKVCache) -> None:
        keys, values = _random_kv(1)
        k_out, v_out = cache.update_and_fetch(keys, values)
        mx.eval(k_out, v_out)
        assert k_out.shape == (B, KV_HEADS, 1, HEAD_DIM)
        assert v_out.shape == (B, KV_HEADS, 1, HEAD_DIM)

    def test_prefill(self, cache: CompressedKVCache) -> None:
        keys, values = _random_kv(100)
        k_out, v_out = cache.update_and_fetch(keys, values)
        mx.eval(k_out, v_out)
        assert k_out.shape == (B, KV_HEADS, 100, HEAD_DIM)
        assert v_out.shape == (B, KV_HEADS, 100, HEAD_DIM)

    def test_incremental(self, cache: CompressedKVCache) -> None:
        """Prefill then add tokens one at a time."""
        keys, values = _random_kv(50)
        cache.update_and_fetch(keys, values)
        for i in range(10):
            k, v = _random_kv(1)
            k_out, v_out = cache.update_and_fetch(k, v)
            mx.eval(k_out, v_out)
            expected_len = 50 + i + 1
            assert k_out.shape == (B, KV_HEADS, expected_len, HEAD_DIM)

    def test_growth_across_step_boundary(self, cache: CompressedKVCache) -> None:
        """Fill past the 256-step allocation boundary."""
        keys, values = _random_kv(200)
        cache.update_and_fetch(keys, values)
        k2, v2 = _random_kv(100)
        k_out, v_out = cache.update_and_fetch(k2, v2)
        mx.eval(k_out, v_out)
        assert k_out.shape == (B, KV_HEADS, 300, HEAD_DIM)


class TestOffset:
    def test_starts_at_zero(self, cache: CompressedKVCache) -> None:
        assert cache.offset == 0

    def test_increments_on_update(self, cache: CompressedKVCache) -> None:
        cache.update_and_fetch(*_random_kv(50))
        assert cache.offset == 50
        cache.update_and_fetch(*_random_kv(1))
        assert cache.offset == 51

    def test_is_plain_attribute(self, cache: CompressedKVCache) -> None:
        """offset must be a plain int, not a property -- matching KVCache."""
        assert isinstance(cache.offset, int)
        assert "offset" not in type(cache).__dict__


class TestProtocolCompliance:
    def test_no_bits_attribute(self, cache: CompressedKVCache) -> None:
        assert not hasattr(cache, "bits")

    def test_no_to_quantized(self, cache: CompressedKVCache) -> None:
        assert not hasattr(cache, "to_quantized")

    def test_no_group_size(self, cache: CompressedKVCache) -> None:
        assert not hasattr(cache, "group_size")

    def test_is_trimmable(self, cache: CompressedKVCache) -> None:
        assert cache.is_trimmable()

    def test_trim(self, cache: CompressedKVCache) -> None:
        cache.update_and_fetch(*_random_kv(100))
        trimmed = cache.trim(30)
        assert trimmed == 30
        assert cache.offset == 70

    def test_trim_capped(self, cache: CompressedKVCache) -> None:
        cache.update_and_fetch(*_random_kv(10))
        trimmed = cache.trim(100)
        assert trimmed == 10
        assert cache.offset == 0

    def test_empty_after_trimming_to_zero(self, cache: CompressedKVCache) -> None:
        cache.update_and_fetch(*_random_kv(10))
        cache.trim(10)
        assert cache.empty()

    def test_empty_initially(self, cache: CompressedKVCache) -> None:
        assert cache.empty()

    def test_not_empty_after_update(self, cache: CompressedKVCache) -> None:
        cache.update_and_fetch(*_random_kv(1))
        assert not cache.empty()

    def test_size(self, cache: CompressedKVCache) -> None:
        cache.update_and_fetch(*_random_kv(42))
        assert cache.size() == 42

    def test_make_mask_callable(self, cache: CompressedKVCache) -> None:
        cache.update_and_fetch(*_random_kv(10))
        result = cache.make_mask(N=1, return_array=True, window_size=None)
        assert result is None or isinstance(result, (str, mx.array))


class TestState:
    def test_state_returns_tuple(self, cache: CompressedKVCache) -> None:
        cache.update_and_fetch(*_random_kv(10))
        k, v = cache.state
        assert isinstance(k, mx.array)
        assert isinstance(v, mx.array)
        assert k.shape == (B, KV_HEADS, 10, HEAD_DIM)

    def test_state_empty_cache(self, cache: CompressedKVCache) -> None:
        k, v = cache.state
        assert k is None and v is None

    def test_meta_state(self, cache: CompressedKVCache) -> None:
        cache.update_and_fetch(*_random_kv(10))
        ms = cache.meta_state
        assert isinstance(ms, tuple)
        assert ms[0] == "10"

    def test_setting_state_none_clears_cache(self, cache: CompressedKVCache) -> None:
        cache.update_and_fetch(*_random_kv(10))
        cache.state = (None, None)
        assert cache.size() == 0
        assert cache.empty()
        assert cache.nbytes == 0
        assert cache.state == (None, None)

    def test_setting_partial_state_raises(self, cache: CompressedKVCache) -> None:
        keys, _values = _random_kv(10)
        with pytest.raises(ValueError, match="both keys and values, or both None"):
            cache.state = (keys, None)


class TestCompression:
    def test_nbytes_less_than_fp16(self, cache: CompressedKVCache) -> None:
        n_tokens = 100
        cache.update_and_fetch(*_random_kv(n_tokens))
        compressed_bytes = cache.nbytes
        fp16_bytes = 2 * B * KV_HEADS * n_tokens * HEAD_DIM * 2
        assert compressed_bytes < fp16_bytes, f"Compressed {compressed_bytes} >= fp16 {fp16_bytes}"

    def test_decompressed_quality(self, cache: CompressedKVCache) -> None:
        """Decompressed output should have NMSE within codec bounds."""
        keys, values = _random_kv(50)
        k_out, v_out = cache.update_and_fetch(keys, values)
        mx.eval(k_out, v_out)
        mse_k = float(mx.mean((keys - k_out) ** 2))
        norm_k = float(mx.mean(keys**2))
        nmse_k = mse_k / (norm_k + 1e-10)
        assert nmse_k < 0.05, f"Key NMSE={nmse_k:.4f}"

    def test_nbytes_zero_when_empty(self, cache: CompressedKVCache) -> None:
        assert cache.nbytes == 0


class TestAttentionSink:
    """Tests for attention sink (FP16 initial tokens)."""

    @pytest.fixture()
    def sink_cache(self) -> CompressedKVCache:
        codec = Stage1Codec(CodecConfig(head_dim=HEAD_DIM, bits=BITS))
        return CompressedKVCache(codec, sink_tokens=4)

    def test_sink_tokens_stored_in_fp16(self, sink_cache: CompressedKVCache) -> None:
        """First sink_tokens tokens should bypass compression."""
        keys, values = _random_kv(4)
        k_out, v_out = sink_cache.update_and_fetch(keys, values)
        mx.eval(k_out, v_out)
        # Sink stores in FP16 — should be near-lossless (only fp16 rounding)
        mse = float(mx.mean((keys.astype(mx.float32) - k_out) ** 2))
        norm = float(mx.mean(keys.astype(mx.float32) ** 2))
        nmse = mse / (norm + 1e-10)
        assert nmse < 1e-4, f"Sink NMSE={nmse:.6f} — expected near-lossless"

    def test_sink_plus_compressed(self, sink_cache: CompressedKVCache) -> None:
        """Prefill that spans sink + compressed boundary."""
        keys, values = _random_kv(10)
        k_out, v_out = sink_cache.update_and_fetch(keys, values)
        mx.eval(k_out, v_out)
        assert k_out.shape == (B, KV_HEADS, 10, HEAD_DIM)
        assert sink_cache.offset == 10
        assert sink_cache._sink_filled == 4

    def test_sink_incremental(self, sink_cache: CompressedKVCache) -> None:
        """Add tokens one at a time across the sink boundary."""
        for i in range(8):
            k, v = _random_kv(1)
            k_out, v_out = sink_cache.update_and_fetch(k, v)
            mx.eval(k_out, v_out)
            assert k_out.shape == (B, KV_HEADS, i + 1, HEAD_DIM)
        assert sink_cache._sink_filled == 4
        assert sink_cache.offset == 8

    def test_sink_memory_accounting(self, sink_cache: CompressedKVCache) -> None:
        """nbytes should include FP16 sink + compressed."""
        sink_cache.update_and_fetch(*_random_kv(10))
        total_bytes = sink_cache.nbytes
        # Sink: 4 tokens * 2 heads * 128 dim * 2 bytes (fp16) * 2 (K+V) = 4096
        expected_sink = 4 * KV_HEADS * HEAD_DIM * 2 * 2
        assert total_bytes > expected_sink, "Total should exceed sink-only bytes"
        # Should be less than all-FP16
        fp16_bytes = 2 * B * KV_HEADS * 10 * HEAD_DIM * 2
        assert total_bytes < fp16_bytes

    def test_sink_zero_disables(self) -> None:
        """sink_tokens=0 matches v0.1.0 behavior."""
        codec = Stage1Codec(CodecConfig(head_dim=HEAD_DIM, bits=BITS))
        cache_no_sink = CompressedKVCache(codec, sink_tokens=0)
        cache_default = CompressedKVCache(codec)
        keys, values = _random_kv(10)
        k1, v1 = cache_no_sink.update_and_fetch(keys, values)
        k2, v2 = cache_default.update_and_fetch(keys, values)
        mx.eval(k1, v1, k2, v2)
        assert cache_no_sink.nbytes == cache_default.nbytes
        assert cache_no_sink._sink_filled == 0

    def test_sink_quality_improvement(self) -> None:
        """Sink should improve quality for the first few tokens at low bit-width."""
        codec = Stage1Codec(CodecConfig(head_dim=HEAD_DIM, bits=2))
        cache_no_sink = CompressedKVCache(codec, sink_tokens=0)
        cache_with_sink = CompressedKVCache(codec, sink_tokens=4)

        keys, values = _random_kv(20)
        k_no, _ = cache_no_sink.update_and_fetch(keys, values)
        k_yes, _ = cache_with_sink.update_and_fetch(keys, values)
        mx.eval(k_no, k_yes)

        # Compute MSE over just the first 4 tokens
        mse_no = float(mx.mean((keys[:, :, :4, :].astype(mx.float32) - k_no[:, :, :4, :]) ** 2))
        mse_yes = float(mx.mean((keys[:, :, :4, :].astype(mx.float32) - k_yes[:, :, :4, :]) ** 2))
        assert mse_yes < mse_no, f"Sink MSE={mse_yes:.6f} should be < no-sink MSE={mse_no:.6f}"

    def test_sink_trim(self, sink_cache: CompressedKVCache) -> None:
        """Trimming into the sink region adjusts _sink_filled."""
        sink_cache.update_and_fetch(*_random_kv(10))
        assert sink_cache._sink_filled == 4
        sink_cache.trim(8)
        assert sink_cache.offset == 2
        assert sink_cache._sink_filled == 2

    def test_sink_state_roundtrip(self, sink_cache: CompressedKVCache) -> None:
        """state setter re-populates sink correctly."""
        sink_cache.update_and_fetch(*_random_kv(10))
        k, v = sink_cache.state
        assert k is not None and v is not None
        sink_cache.state = (None, None)
        assert sink_cache.empty()
        assert sink_cache._sink_filled == 0
        sink_cache.state = (k, v)
        assert sink_cache.offset == 10
        assert sink_cache._sink_filled == 4

    def test_sink_empty(self, sink_cache: CompressedKVCache) -> None:
        """Empty cache with sink_tokens > 0 should report empty."""
        assert sink_cache.empty()
        assert sink_cache._sink_filled == 0
        assert sink_cache.nbytes == 0


class TestIncrementalDecode:
    """Tests for incremental decode optimization."""

    def test_incremental_matches_full_decode(self) -> None:
        """Incremental decode should produce the same output as full decode."""
        codec = Stage1Codec(CodecConfig(head_dim=HEAD_DIM, bits=BITS))
        cache = CompressedKVCache(codec)

        # Prefill
        keys_pf, values_pf = _random_kv(50)
        k_out, v_out = cache.update_and_fetch(keys_pf, values_pf)
        mx.eval(k_out, v_out)

        # Add 10 tokens one at a time
        all_keys = [keys_pf]
        all_values = [values_pf]
        for _ in range(10):
            k, v = _random_kv(1)
            all_keys.append(k)
            all_values.append(v)
            k_out, v_out = cache.update_and_fetch(k, v)

        mx.eval(k_out, v_out)

        # Compare against a fresh cache doing full decode
        codec2 = Stage1Codec(CodecConfig(head_dim=HEAD_DIM, bits=BITS))
        cache2 = CompressedKVCache(codec2)
        all_k = mx.concatenate(all_keys, axis=2)
        all_v = mx.concatenate(all_values, axis=2)
        k_full, v_full = cache2.update_and_fetch(all_k, all_v)
        mx.eval(k_full, v_full)

        # Should be identical (same codec, same data)
        assert k_out.shape == k_full.shape == (B, KV_HEADS, 60, HEAD_DIM)
        mse = float(mx.mean((k_out - k_full) ** 2))
        assert mse < 1e-10, f"Incremental vs full decode MSE={mse}"

    def test_decode_cache_persists(self) -> None:
        """Decoded cache should be stored in _decoded_keys/_decoded_values."""
        codec = Stage1Codec(CodecConfig(head_dim=HEAD_DIM, bits=BITS))
        cache = CompressedKVCache(codec)
        cache.update_and_fetch(*_random_kv(10))
        assert cache._decoded_keys is not None
        assert cache._decoded_keys.shape == (B, KV_HEADS, 10, HEAD_DIM)

    def test_trim_truncates_decode_cache(self) -> None:
        """Trimming should truncate the decoded cache."""
        codec = Stage1Codec(CodecConfig(head_dim=HEAD_DIM, bits=BITS))
        cache = CompressedKVCache(codec)
        cache.update_and_fetch(*_random_kv(20))
        assert cache._decoded_keys is not None
        cache.trim(5)
        assert cache._decoded_keys.shape[2] == 15

    def test_reset_clears_decode_cache(self) -> None:
        """Reset should clear the decoded cache."""
        codec = Stage1Codec(CodecConfig(head_dim=HEAD_DIM, bits=BITS))
        cache = CompressedKVCache(codec)
        cache.update_and_fetch(*_random_kv(10))
        cache._reset()
        assert cache._decoded_keys is None
        assert cache._decoded_values is None

    def test_incremental_with_sink(self) -> None:
        """Incremental decode works correctly with attention sink."""
        codec = Stage1Codec(CodecConfig(head_dim=HEAD_DIM, bits=BITS))
        cache = CompressedKVCache(codec, sink_tokens=4)

        # Add tokens one at a time across the sink boundary
        for i in range(8):
            k, v = _random_kv(1)
            k_out, v_out = cache.update_and_fetch(k, v)
            mx.eval(k_out, v_out)
            assert k_out.shape == (B, KV_HEADS, i + 1, HEAD_DIM)

        assert cache._decoded_keys is not None
        assert cache._decoded_keys.shape == (B, KV_HEADS, 8, HEAD_DIM)


class TestDim64:
    """Test with head_dim=64 (Qwen 2.5-0.5B)."""

    def test_update_and_fetch(self) -> None:
        codec = Stage1Codec(CodecConfig(head_dim=64, bits=3))
        cache = CompressedKVCache(codec)
        keys = mx.random.normal((1, 2, 10, 64))
        values = mx.random.normal((1, 2, 10, 64))
        k_out, v_out = cache.update_and_fetch(keys, values)
        mx.eval(k_out, v_out)
        assert k_out.shape == (1, 2, 10, 64)
        assert cache.offset == 10
