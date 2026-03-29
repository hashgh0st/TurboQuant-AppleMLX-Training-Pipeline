"""CompressedKVCache: duck-type compatible with MLX-LM's KVCache protocol.

Stores KV vectors in compressed form using Stage1Codec. On each update_and_fetch,
new vectors are compressed and appended. All stored vectors are decompressed into
a transient buffer for SDPA. The decompressed buffer is freed after the forward pass;
only compressed data persists between steps.

CRITICAL: This class must NOT have `bits`, `group_size`, or `to_quantized` attributes.
MLX-LM checks hasattr(cache, 'bits') to dispatch to quantized SDPA, which is
incompatible with our rotation + codebook approach.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from mlx_turboquant.codec.packbits import packed_dim
from mlx_turboquant.codec.stage1_codec import CompressedTensor, Stage1Codec


class CompressedKVCache:
    """KV cache storing vectors in compressed form via Stage1Codec.

    Duck-types to mlx_lm.models.cache.KVCache protocol.
    """

    step = 256  # allocation granularity, matching KVCache

    def __init__(self, codec: Stage1Codec, *, use_metal: bool = False) -> None:
        self.codec = codec
        self.use_metal = use_metal
        self.offset: int = 0
        self._packed_keys: mx.array | None = None
        self._key_norms: mx.array | None = None
        self._packed_values: mx.array | None = None
        self._value_norms: mx.array | None = None
        self._pdim = packed_dim(codec.config.head_dim, codec.config.bits)

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Compress new KV, append to storage, return all decompressed KV.

        Input: keys/values (B, n_kv_heads, num_steps, head_dim)
        Returns: (all_keys, all_values) decompressed (B, n_kv_heads, offset, head_dim)
        """
        B, n_kv_heads, num_steps, head_dim = keys.shape
        prev = self.offset

        # Encode new keys and values
        # Reshape to (B*n_kv_heads, num_steps, head_dim) for codec
        k_flat = keys.reshape(B * n_kv_heads, num_steps, head_dim)
        v_flat = values.reshape(B * n_kv_heads, num_steps, head_dim)
        ct_k = self.codec.encode(k_flat)
        ct_v = self.codec.encode(v_flat)

        # Reshape packed back to (B, n_kv_heads, num_steps, packed_dim/norms)
        pk = ct_k.packed.reshape(B, n_kv_heads, num_steps, self._pdim)
        kn = ct_k.norms.reshape(B, n_kv_heads, num_steps)
        pv = ct_v.packed.reshape(B, n_kv_heads, num_steps, self._pdim)
        vn = ct_v.norms.reshape(B, n_kv_heads, num_steps)

        # Allocate or grow storage
        if self._packed_keys is None or (prev + num_steps) > self._packed_keys.shape[2]:
            n_alloc = ((self.step + num_steps - 1) // self.step) * self.step
            new_pk = mx.zeros((B, n_kv_heads, n_alloc, self._pdim), dtype=mx.uint32)
            new_kn = mx.zeros((B, n_kv_heads, n_alloc), dtype=mx.float16)
            new_pv = mx.zeros((B, n_kv_heads, n_alloc, self._pdim), dtype=mx.uint32)
            new_vn = mx.zeros((B, n_kv_heads, n_alloc), dtype=mx.float16)
            if (
                self._packed_keys is not None
                and self._key_norms is not None
                and self._packed_values is not None
                and self._value_norms is not None
            ):
                if prev % self.step != 0:
                    self._packed_keys = self._packed_keys[:, :, :prev, :]
                    self._key_norms = self._key_norms[:, :, :prev]
                    self._packed_values = self._packed_values[:, :, :prev, :]
                    self._value_norms = self._value_norms[:, :, :prev]
                self._packed_keys = mx.concatenate([self._packed_keys, new_pk], axis=2)
                self._key_norms = mx.concatenate([self._key_norms, new_kn], axis=2)
                self._packed_values = mx.concatenate([self._packed_values, new_pv], axis=2)
                self._value_norms = mx.concatenate([self._value_norms, new_vn], axis=2)
            else:
                self._packed_keys = new_pk
                self._key_norms = new_kn
                self._packed_values = new_pv
                self._value_norms = new_vn

        # Write new data at [prev:prev+num_steps]
        self.offset += num_steps
        assert self._packed_keys is not None
        assert self._key_norms is not None
        assert self._packed_values is not None
        assert self._value_norms is not None
        self._packed_keys[:, :, prev : self.offset, :] = pk
        self._key_norms[:, :, prev : self.offset] = kn
        self._packed_values[:, :, prev : self.offset, :] = pv
        self._value_norms[:, :, prev : self.offset] = vn

        # Decode all stored data -> transient decompressed tensors
        return self._decode_all(B, n_kv_heads, head_dim)

    def _decode_all(
        self, B: int, n_kv_heads: int, head_dim: int
    ) -> tuple[mx.array, mx.array]:
        """Decode all stored compressed data up to self.offset."""
        assert self._packed_keys is not None
        assert self._key_norms is not None
        assert self._packed_values is not None
        assert self._value_norms is not None

        pk = self._packed_keys[:, :, : self.offset, :]
        kn = self._key_norms[:, :, : self.offset]
        pv = self._packed_values[:, :, : self.offset, :]
        vn = self._value_norms[:, :, : self.offset]

        # Flatten for codec: (B*n_kv_heads, offset, packed_dim)
        pk_flat = pk.reshape(B * n_kv_heads, self.offset, self._pdim)
        kn_flat = kn.reshape(B * n_kv_heads, self.offset)
        pv_flat = pv.reshape(B * n_kv_heads, self.offset, self._pdim)
        vn_flat = vn.reshape(B * n_kv_heads, self.offset)

        ct_k = CompressedTensor(packed=pk_flat, norms=kn_flat, config=self.codec.config)
        ct_v = CompressedTensor(packed=pv_flat, norms=vn_flat, config=self.codec.config)

        keys_dec = self.codec.decode(ct_k, use_metal=self.use_metal).reshape(
            B, n_kv_heads, self.offset, head_dim
        )
        vals_dec = self.codec.decode(ct_v, use_metal=self.use_metal).reshape(
            B, n_kv_heads, self.offset, head_dim
        )
        return keys_dec, vals_dec

    @property
    def state(self) -> tuple[mx.array | None, mx.array | None]:
        """Return decompressed K, V for serialization compatibility."""
        if self._packed_keys is None:
            return None, None
        B = self._packed_keys.shape[0]
        n_kv_heads = self._packed_keys.shape[1]
        head_dim = self.codec.config.head_dim
        return self._decode_all(B, n_kv_heads, head_dim)

    @state.setter
    def state(self, v: tuple[mx.array | None, mx.array | None]) -> None:
        """Load from decompressed state by re-compressing."""
        keys, values = v
        if keys is None or values is None:
            return
        self.offset = 0
        self._packed_keys = None
        self._key_norms = None
        self._packed_values = None
        self._value_norms = None
        self.update_and_fetch(keys, values)

    @property
    def meta_state(self) -> tuple[str, ...]:
        """Serialization metadata."""
        cfg = self.codec.config
        return (str(self.offset), str(cfg.head_dim), str(cfg.bits), str(cfg.seed))

    @meta_state.setter
    def meta_state(self, v: tuple[str, ...]) -> None:
        self.offset = int(v[0])

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, N: int, return_array: bool = True, window_size: Any = None) -> Any:
        """Delegate to MLX-LM's mask creation."""
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(
            N, offset=self.offset, return_array=return_array, window_size=window_size
        )

    def empty(self) -> bool:
        return self._packed_keys is None

    def size(self) -> int:
        return self.offset

    @property
    def nbytes(self) -> int:
        """Actual compressed bytes stored (not decompressed size).

        Uses pure arithmetic instead of slicing arrays to avoid
        creating graph nodes on every access.
        """
        if self._packed_keys is None:
            return 0
        B = self._packed_keys.shape[0]
        n_kv_heads = self._packed_keys.shape[1]
        # K and V each: packed uint32 (4 bytes) + float16 norm (2 bytes) per token per head
        packed_bytes = B * n_kv_heads * self.offset * self._pdim * 4
        norm_bytes = B * n_kv_heads * self.offset * 2
        return 2 * (packed_bytes + norm_bytes)  # 2x for K + V
