"""Tests for randomized Hadamard transform."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_turboquant.codec.transforms import (
    create_transform,
    forward_transform,
    inverse_transform,
)


@pytest.fixture(params=[64, 128])
def head_dim(request: pytest.FixtureRequest) -> int:
    return request.param  # type: ignore[no-any-return]


class TestRoundtrip:
    def test_forward_inverse(self, head_dim: int) -> None:
        """inverse(forward(x)) should recover x within float32 tolerance."""
        state = create_transform(head_dim)
        x = mx.random.normal((head_dim,))
        reconstructed = inverse_transform(forward_transform(x, state), state)
        mx.eval(reconstructed)
        diff = float(mx.max(mx.abs(x - reconstructed)))
        assert diff < 1e-5, f"Round-trip error: {diff}"

    def test_inverse_forward(self, head_dim: int) -> None:
        """forward(inverse(x)) should also recover x."""
        state = create_transform(head_dim)
        x = mx.random.normal((head_dim,))
        reconstructed = forward_transform(inverse_transform(x, state), state)
        mx.eval(reconstructed)
        diff = float(mx.max(mx.abs(x - reconstructed)))
        assert diff < 1e-5, f"Round-trip error: {diff}"


class TestNormPreservation:
    def test_preserves_norm(self, head_dim: int) -> None:
        """The transform should preserve L2 norms (orthogonal rotation)."""
        state = create_transform(head_dim)
        x = mx.random.normal((head_dim,))
        y = forward_transform(x, state)
        mx.eval(y)
        norm_x = float(mx.linalg.norm(x))
        norm_y = float(mx.linalg.norm(y))
        ratio = norm_y / norm_x
        assert abs(ratio - 1.0) < 1e-5, f"Norm ratio: {ratio}"

    def test_preserves_norm_batched(self, head_dim: int) -> None:
        state = create_transform(head_dim)
        x = mx.random.normal((4, 8, head_dim))
        y = forward_transform(x, state)
        mx.eval(y)
        norms_x = mx.linalg.norm(x, axis=-1)
        norms_y = mx.linalg.norm(y, axis=-1)
        ratio = norms_y / norms_x
        mx.eval(ratio)
        max_deviation = float(mx.max(mx.abs(ratio - 1.0)))
        assert max_deviation < 1e-4, f"Max norm deviation: {max_deviation}"


class TestDeterminism:
    def test_same_seed_same_signs(self) -> None:
        s1 = create_transform(128, seed=42)
        s2 = create_transform(128, seed=42)
        mx.eval(s1.signs_pre, s1.signs_post, s2.signs_pre, s2.signs_post)
        assert float(mx.max(mx.abs(s1.signs_pre - s2.signs_pre))) == 0.0
        assert float(mx.max(mx.abs(s1.signs_post - s2.signs_post))) == 0.0

    def test_different_seeds_differ(self) -> None:
        s1 = create_transform(128, seed=42)
        s2 = create_transform(128, seed=99)
        mx.eval(s1.signs_pre, s2.signs_pre)
        assert float(mx.max(mx.abs(s1.signs_pre - s2.signs_pre))) > 0.0


class TestBatched:
    def test_shape_preserved(self, head_dim: int) -> None:
        state = create_transform(head_dim)
        x = mx.random.normal((2, 3, head_dim))
        y = forward_transform(x, state)
        mx.eval(y)
        assert y.shape == x.shape

    def test_single_vs_batched_consistent(self, head_dim: int) -> None:
        """Batched transform should match element-wise application."""
        state = create_transform(head_dim)
        batch = mx.random.normal((3, head_dim))
        batched_result = forward_transform(batch, state)
        mx.eval(batched_result)
        for i in range(3):
            single_result = forward_transform(batch[i], state)
            mx.eval(single_result)
            diff = float(mx.max(mx.abs(batched_result[i] - single_result)))
            assert diff < 1e-6, f"Batch/single mismatch at index {i}: {diff}"
