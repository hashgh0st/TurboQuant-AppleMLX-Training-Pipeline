"""Tests for generation wrapper cache metrics."""

from __future__ import annotations

import mlx_turboquant.integration.generate_wrapper as generate_wrapper


class _FakeArgs:
    hidden_size = 128
    num_attention_heads = 1
    num_hidden_layers = 2
    num_key_value_heads = 1


class _FakeModel:
    args = _FakeArgs()


class _FakeTokenizer:
    eos_token_id = -1

    def encode(self, _prompt: str) -> list[int]:
        return [11, 12, 13]

    def decode(self, tokens: list[int]) -> str:
        return " ".join(str(token) for token in tokens)


class _FakeCache:
    def __init__(self, *, nbytes: int, allocated_nbytes: int | None = None) -> None:
        self._size = 0
        self._nbytes = nbytes
        self.allocated_nbytes = allocated_nbytes if allocated_nbytes is not None else nbytes

    def size(self) -> int:
        return self._size

    @property
    def nbytes(self) -> int:
        return self._nbytes


def _fake_generate_step(*, prompt, model, prompt_cache, **_kwargs):
    del model
    occupied_tokens = int(prompt.size) + 2
    for layer in prompt_cache:
        layer._size = occupied_tokens
    yield 101, None
    yield 102, None


def test_baseline_uses_logical_cache_bytes_not_allocated_buffers(
    monkeypatch,
) -> None:
    caches = [_FakeCache(nbytes=16384), _FakeCache(nbytes=16384)]
    monkeypatch.setattr(generate_wrapper, "make_prompt_cache", lambda _model: caches)
    monkeypatch.setattr(generate_wrapper, "generate_step", _fake_generate_step)

    result = generate_wrapper.generate_baseline(
        _FakeModel(),
        _FakeTokenizer(),
        "hello",
        max_tokens=2,
    )

    assert result.cache_bytes == 5120
    assert result.cache_allocated_bytes == 32768


def test_compressed_uses_logical_bytes_and_tracks_allocated_bytes(
    monkeypatch,
) -> None:
    caches = [
        _FakeCache(nbytes=120, allocated_nbytes=640),
        _FakeCache(nbytes=120, allocated_nbytes=640),
    ]
    monkeypatch.setattr(generate_wrapper, "make_compressed_cache", lambda *_args, **_kwargs: caches)
    monkeypatch.setattr(generate_wrapper, "generate_step", _fake_generate_step)

    result = generate_wrapper.generate_with_compressed_cache(
        _FakeModel(),
        _FakeTokenizer(),
        "hello",
        kv_bits=3,
        max_tokens=2,
    )

    assert result.cache_bytes == 1080
    assert result.cache_allocated_bytes == 1280
