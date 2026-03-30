"""CompressedKVCache: duck-type compatible with MLX-LM's KVCache protocol.

Stores KV vectors in compressed form using Stage1Codec. Uses **incremental decode**:
previously decoded output is cached so autoregressive decode is O(1) per step
instead of O(seq_len). Optional **attention sink** keeps the first N tokens in
uncompressed FP16 for quality preservation.

CRITICAL: This class must NOT have `bits`, `group_size`, or `to_quantized` attributes.
MLX-LM checks hasattr(cache, 'bits') to dispatch to quantized SDPA, which is
incompatible with our rotation + codebook approach.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from mlx_turboquant.codec.packbits import packed_dim, signs_packed_dim
from mlx_turboquant.codec.stage1_codec import CompressedTensor, Stage1Codec


class CompressedKVCache:
    """KV cache storing vectors in compressed form via Stage1Codec.

    Duck-types to mlx_lm.models.cache.KVCache protocol.

    When ``sink_tokens > 0``, the first *sink_tokens* tokens are kept in
    uncompressed FP16 (the "attention sink").  Only tokens beyond the sink
    window are compressed.  This preserves quality on the initial tokens
    that accumulate disproportionate attention mass (StreamingLLM insight).
    """

    step = 256  # allocation granularity, matching KVCache

    def __init__(
        self,
        codec: Stage1Codec,
        value_codec: Stage1Codec | None = None,
        *,
        use_metal: bool = False,
        sink_tokens: int = 0,
    ) -> None:
        self.key_codec = codec
        self.value_codec = codec if value_codec is None else value_codec
        self.use_metal = use_metal
        self.sink_tokens = sink_tokens
        self.offset: int = 0
        # Attention sink: FP16 storage for the first sink_tokens tokens
        self._sink_keys: mx.array | None = None
        self._sink_values: mx.array | None = None
        self._sink_filled: int = 0
        # Compressed storage for tokens beyond the sink
        self._packed_keys: mx.array | None = None
        self._key_norms: mx.array | None = None
        self._packed_values: mx.array | None = None
        self._value_norms: mx.array | None = None
        # QJL fields (keys only — values don't need unbiased inner products)
        self._qjl_packed_keys: mx.array | None = None
        self._key_residual_norms: mx.array | None = None
        self._use_qjl = self.key_codec.config.use_qjl
        # Incremental decode cache
        self._decoded_keys: mx.array | None = None
        self._decoded_values: mx.array | None = None
        if self.key_codec.config.head_dim != self.value_codec.config.head_dim:
            raise ValueError("key and value codecs must share head_dim")
        # Use mse_bits for packed_dim (bits-1 when QJL is on)
        self._key_pdim = packed_dim(self.key_codec.config.head_dim, self.key_codec.mse_bits)
        self._value_pdim = packed_dim(self.value_codec.config.head_dim, self.value_codec.mse_bits)
        self._key_qjl_pdim = signs_packed_dim(self.key_codec.config.head_dim)

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """Compress new KV, append to storage, return all decompressed KV.

        Input: keys/values (B, n_kv_heads, num_steps, head_dim)
        Returns: (all_keys, all_values) decompressed (B, n_kv_heads, offset, head_dim)

        When ``sink_tokens > 0``, the first tokens go into the FP16 sink
        buffer; only subsequent tokens are compressed.
        """
        B, n_kv_heads, num_steps, head_dim = keys.shape
        old_sink_filled = self._sink_filled
        compressed_prev = self.offset - old_sink_filled

        sink_remaining = max(0, self.sink_tokens - old_sink_filled)
        sink_take = min(num_steps, sink_remaining)
        compress_steps = num_steps - sink_take

        if sink_take > 0:
            if self._sink_keys is None:
                self._sink_keys = mx.zeros(
                    (B, n_kv_heads, self.sink_tokens, head_dim), dtype=mx.float16
                )
                self._sink_values = mx.zeros(
                    (B, n_kv_heads, self.sink_tokens, head_dim), dtype=mx.float16
                )
            assert self._sink_keys is not None and self._sink_values is not None
            sf = old_sink_filled
            self._sink_keys[:, :, sf : sf + sink_take, :] = keys[:, :, :sink_take, :].astype(
                mx.float16
            )
            self._sink_values[:, :, sf : sf + sink_take, :] = values[:, :, :sink_take, :].astype(
                mx.float16
            )
            self._sink_filled = sf + sink_take

        if compress_steps > 0:
            k_compress = keys[:, :, sink_take:, :]
            v_compress = values[:, :, sink_take:, :]

            k_flat = k_compress.reshape(B * n_kv_heads, compress_steps, head_dim)
            v_flat = v_compress.reshape(B * n_kv_heads, compress_steps, head_dim)
            ct_k = self.key_codec.encode(k_flat)
            ct_v = self.value_codec.encode(v_flat)

            pk = ct_k.packed.reshape(B, n_kv_heads, compress_steps, self._key_pdim)
            kn = ct_k.norms.reshape(B, n_kv_heads, compress_steps)
            pv = ct_v.packed.reshape(B, n_kv_heads, compress_steps, self._value_pdim)
            vn = ct_v.norms.reshape(B, n_kv_heads, compress_steps)

            new_compressed_end = compressed_prev + compress_steps
            if self._packed_keys is None or new_compressed_end > self._packed_keys.shape[2]:
                n_alloc = ((self.step + compress_steps - 1) // self.step) * self.step
                new_pk = mx.zeros((B, n_kv_heads, n_alloc, self._key_pdim), dtype=mx.uint32)
                new_kn = mx.zeros((B, n_kv_heads, n_alloc), dtype=mx.float32)
                new_pv = mx.zeros((B, n_kv_heads, n_alloc, self._value_pdim), dtype=mx.uint32)
                new_vn = mx.zeros((B, n_kv_heads, n_alloc), dtype=mx.float32)
                if (
                    self._packed_keys is not None
                    and self._key_norms is not None
                    and self._packed_values is not None
                    and self._value_norms is not None
                ):
                    if compressed_prev % self.step != 0:
                        self._packed_keys = self._packed_keys[:, :, :compressed_prev, :]
                        self._key_norms = self._key_norms[:, :, :compressed_prev]
                        self._packed_values = self._packed_values[:, :, :compressed_prev, :]
                        self._value_norms = self._value_norms[:, :, :compressed_prev]
                    self._packed_keys = mx.concatenate([self._packed_keys, new_pk], axis=2)
                    self._key_norms = mx.concatenate([self._key_norms, new_kn], axis=2)
                    self._packed_values = mx.concatenate([self._packed_values, new_pv], axis=2)
                    self._value_norms = mx.concatenate([self._value_norms, new_vn], axis=2)
                else:
                    self._packed_keys = new_pk
                    self._key_norms = new_kn
                    self._packed_values = new_pv
                    self._value_norms = new_vn

                # QJL storage for keys
                if self._use_qjl:
                    new_qjl = mx.zeros(
                        (B, n_kv_heads, n_alloc, self._key_qjl_pdim), dtype=mx.uint32
                    )
                    new_krn = mx.zeros((B, n_kv_heads, n_alloc), dtype=mx.float16)
                    if self._qjl_packed_keys is not None:
                        assert self._key_residual_norms is not None
                        if compressed_prev % self.step != 0:
                            self._qjl_packed_keys = self._qjl_packed_keys[:, :, :compressed_prev, :]
                            self._key_residual_norms = self._key_residual_norms[:, :, :compressed_prev]
                        self._qjl_packed_keys = mx.concatenate(
                            [self._qjl_packed_keys, new_qjl], axis=2
                        )
                        self._key_residual_norms = mx.concatenate(
                            [self._key_residual_norms, new_krn], axis=2
                        )
                    else:
                        self._qjl_packed_keys = new_qjl
                        self._key_residual_norms = new_krn

            assert self._packed_keys is not None
            assert self._key_norms is not None
            assert self._packed_values is not None
            assert self._value_norms is not None
            self._packed_keys[:, :, compressed_prev:new_compressed_end, :] = pk
            self._key_norms[:, :, compressed_prev:new_compressed_end] = kn
            self._packed_values[:, :, compressed_prev:new_compressed_end, :] = pv
            self._value_norms[:, :, compressed_prev:new_compressed_end] = vn

            if self._use_qjl and ct_k.qjl_packed is not None:
                qjl_pk = ct_k.qjl_packed.reshape(B, n_kv_heads, compress_steps, self._key_qjl_pdim)
                krn = ct_k.residual_norms.reshape(B, n_kv_heads, compress_steps)  # type: ignore[union-attr]
                assert self._qjl_packed_keys is not None
                assert self._key_residual_norms is not None
                self._qjl_packed_keys[:, :, compressed_prev:new_compressed_end, :] = qjl_pk
                self._key_residual_norms[:, :, compressed_prev:new_compressed_end] = krn

        self.offset += num_steps
        return self._decode_incremental(B, n_kv_heads, head_dim, sink_take, compress_steps)

    def _decode_incremental(
        self,
        B: int,
        n_kv_heads: int,
        head_dim: int,
        sink_take: int,
        compress_steps: int,
    ) -> tuple[mx.array, mx.array]:
        """Decode only newly added tokens and concatenate with cached output.

        During autoregressive generation (num_steps=1), this decodes one token
        instead of the entire sequence — O(1) instead of O(seq_len).
        """
        new_k: list[mx.array] = []
        new_v: list[mx.array] = []

        if sink_take > 0:
            sf_start = self._sink_filled - sink_take
            assert self._sink_keys is not None and self._sink_values is not None
            new_k.append(self._sink_keys[:, :, sf_start : self._sink_filled, :].astype(mx.float32))
            new_v.append(
                self._sink_values[:, :, sf_start : self._sink_filled, :].astype(mx.float32)
            )

        if compress_steps > 0:
            compressed_end = self.offset - self._sink_filled
            compressed_start = compressed_end - compress_steps
            k_dec, v_dec = self._decode_compressed_slice(
                B, n_kv_heads, head_dim, compressed_start, compressed_end
            )
            new_k.append(k_dec)
            new_v.append(v_dec)

        if not new_k:
            new_keys = mx.zeros((B, n_kv_heads, 0, head_dim), dtype=mx.float32)
            new_vals = new_keys
        elif len(new_k) == 1:
            new_keys, new_vals = new_k[0], new_v[0]
        else:
            new_keys = mx.concatenate(new_k, axis=2)
            new_vals = mx.concatenate(new_v, axis=2)

        if self._decoded_keys is None:
            self._decoded_keys = new_keys
            self._decoded_values = new_vals
        else:
            assert self._decoded_values is not None
            self._decoded_keys = mx.concatenate([self._decoded_keys, new_keys], axis=2)
            self._decoded_values = mx.concatenate([self._decoded_values, new_vals], axis=2)

        assert self._decoded_keys is not None and self._decoded_values is not None
        return self._decoded_keys, self._decoded_values

    def _decode_compressed_slice(
        self,
        B: int,
        n_kv_heads: int,
        head_dim: int,
        start: int,
        end: int,
    ) -> tuple[mx.array, mx.array]:
        """Decode a slice [start:end] of compressed storage to float32."""
        n = end - start
        assert self._packed_keys is not None
        assert self._key_norms is not None
        assert self._packed_values is not None
        assert self._value_norms is not None

        pk_flat = self._packed_keys[:, :, start:end, :].reshape(B * n_kv_heads, n, self._key_pdim)
        kn_flat = self._key_norms[:, :, start:end].reshape(B * n_kv_heads, n)
        pv_flat = self._packed_values[:, :, start:end, :].reshape(
            B * n_kv_heads, n, self._value_pdim
        )
        vn_flat = self._value_norms[:, :, start:end].reshape(B * n_kv_heads, n)

        qjl_flat = None
        krn_flat = None
        if self._use_qjl and self._qjl_packed_keys is not None:
            qjl_flat = self._qjl_packed_keys[:, :, start:end, :].reshape(
                B * n_kv_heads, n, self._key_qjl_pdim
            )
            assert self._key_residual_norms is not None
            krn_flat = self._key_residual_norms[:, :, start:end].reshape(B * n_kv_heads, n)

        ct_k = CompressedTensor(
            packed=pk_flat,
            norms=kn_flat,
            config=self.key_codec.config,
            qjl_packed=qjl_flat,
            residual_norms=krn_flat,
        )
        ct_v = CompressedTensor(packed=pv_flat, norms=vn_flat, config=self.value_codec.config)

        keys_dec = self.key_codec.decode(ct_k, use_metal=self.use_metal).reshape(
            B, n_kv_heads, n, head_dim
        )
        vals_dec = self.value_codec.decode(ct_v, use_metal=self.use_metal).reshape(
            B, n_kv_heads, n, head_dim
        )
        return keys_dec, vals_dec

    def _decode_all(self, B: int, n_kv_heads: int, head_dim: int) -> tuple[mx.array, mx.array]:
        """Full decode from compressed storage (fallback for state property)."""
        parts_k: list[mx.array] = []
        parts_v: list[mx.array] = []

        if self._sink_filled > 0:
            assert self._sink_keys is not None and self._sink_values is not None
            parts_k.append(self._sink_keys[:, :, : self._sink_filled, :].astype(mx.float32))
            parts_v.append(self._sink_values[:, :, : self._sink_filled, :].astype(mx.float32))

        compressed_len = self.offset - self._sink_filled
        if compressed_len > 0:
            k_dec, v_dec = self._decode_compressed_slice(B, n_kv_heads, head_dim, 0, compressed_len)
            parts_k.append(k_dec)
            parts_v.append(v_dec)

        if not parts_k:
            empty = mx.zeros((B, n_kv_heads, 0, head_dim), dtype=mx.float32)
            return empty, empty
        if len(parts_k) == 1:
            return parts_k[0], parts_v[0]
        return mx.concatenate(parts_k, axis=2), mx.concatenate(parts_v, axis=2)

    @property
    def state(self) -> tuple[mx.array | None, mx.array | None]:
        """Return decompressed K, V for serialization compatibility."""
        if self.offset == 0:
            return None, None
        # Use incremental decode cache if available
        if self._decoded_keys is not None:
            return self._decoded_keys, self._decoded_values
        # Fallback: full decode from compressed storage
        dims = self._get_dims()
        if dims is None:
            return None, None
        B, n_kv_heads = dims
        head_dim = self.key_codec.config.head_dim
        return self._decode_all(B, n_kv_heads, head_dim)

    @state.setter
    def state(self, v: tuple[mx.array | None, mx.array | None]) -> None:
        """Load from decompressed state by re-compressing."""
        keys, values = v
        if keys is None and values is None:
            self._reset()
            return
        if keys is None or values is None:
            raise ValueError("state requires both keys and values, or both None")
        self._reset()
        self.update_and_fetch(keys, values)

    def _reset(self) -> None:
        """Clear all stored state."""
        self.offset = 0
        self._sink_filled = 0
        self._sink_keys = None
        self._sink_values = None
        self._packed_keys = None
        self._key_norms = None
        self._packed_values = None
        self._value_norms = None
        self._qjl_packed_keys = None
        self._key_residual_norms = None
        self._decoded_keys = None
        self._decoded_values = None

    @property
    def meta_state(self) -> tuple[str, ...]:
        """Serialization metadata."""
        return (
            str(self.offset),
            str(self.key_codec.config.head_dim),
            str(self.key_codec.config.bits),
            str(self.value_codec.config.bits),
            str(self.key_codec.config.seed),
            str(self.value_codec.config.seed),
        )

    @meta_state.setter
    def meta_state(self, v: tuple[str, ...]) -> None:
        self.offset = int(v[0])

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        # Sink can't hold more tokens than total offset
        self._sink_filled = min(self._sink_filled, self.offset)
        # Truncate decoded cache to match
        if self._decoded_keys is not None:
            if self.offset > 0:
                self._decoded_keys = self._decoded_keys[:, :, : self.offset, :]
                self._decoded_values = self._decoded_values[:, :, : self.offset, :]  # type: ignore[index]
            else:
                self._decoded_keys = None
                self._decoded_values = None
        return n

    def make_mask(self, N: int, return_array: bool = True, window_size: Any = None) -> Any:
        """Delegate to MLX-LM's mask creation."""
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(
            N, offset=self.offset, return_array=return_array, window_size=window_size
        )

    def empty(self) -> bool:
        return self.offset == 0

    def size(self) -> int:
        return self.offset

    def _get_dims(self) -> tuple[int, int] | None:
        """Return (B, n_kv_heads) from whichever buffer exists, or None."""
        if self._sink_keys is not None:
            return self._sink_keys.shape[0], self._sink_keys.shape[1]
        if self._packed_keys is not None:
            return self._packed_keys.shape[0], self._packed_keys.shape[1]
        return None

    @property
    def nbytes(self) -> int:
        """Actual bytes stored: FP16 sink + compressed remainder.

        Uses pure arithmetic instead of slicing arrays to avoid
        creating graph nodes on every access.
        """
        if self.offset == 0:
            return 0
        dims = self._get_dims()
        if dims is None:
            return 0
        B, n_kv_heads = dims
        head_dim = self.key_codec.config.head_dim

        sink_bytes = 2 * B * n_kv_heads * self._sink_filled * head_dim * 2
        compressed_len = self.offset - self._sink_filled
        key_packed_bytes = B * n_kv_heads * compressed_len * self._key_pdim * 4
        key_norm_bytes = B * n_kv_heads * compressed_len * 4  # float32 corrected norms
        value_packed_bytes = B * n_kv_heads * compressed_len * self._value_pdim * 4
        value_norm_bytes = B * n_kv_heads * compressed_len * 4  # float32 corrected norms

        return (
            sink_bytes + key_packed_bytes + key_norm_bytes + value_packed_bytes + value_norm_bytes
        )

    @property
    def allocated_nbytes(self) -> int:
        """Backing-buffer bytes currently allocated (sink + compressed)."""
        alloc = 0
        if self._sink_keys is not None:
            B, n_kv_heads = self._sink_keys.shape[0], self._sink_keys.shape[1]
            head_dim = self.key_codec.config.head_dim
            alloc += 2 * B * n_kv_heads * self.sink_tokens * head_dim * 2  # K+V fp16
        if self._packed_keys is not None:
            B, n_kv_heads = self._packed_keys.shape[0], self._packed_keys.shape[1]
            alloc_steps = self._packed_keys.shape[2]
            alloc += B * n_kv_heads * alloc_steps * self._key_pdim * 4
            alloc += B * n_kv_heads * alloc_steps * 4  # float32 key norms
            alloc += B * n_kv_heads * alloc_steps * self._value_pdim * 4
            alloc += B * n_kv_heads * alloc_steps * 4  # float32 value norms
        return alloc
