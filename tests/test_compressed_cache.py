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
