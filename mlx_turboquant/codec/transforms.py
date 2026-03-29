"""Randomized Hadamard transform for data-oblivious coordinate rotation.

After this transform, each coordinate of a unit-norm vector follows Beta(d/2, d/2),
regardless of the input distribution. This enables data-oblivious codebook quantization.

Uses mx.hadamard_transform which auto-scales by 1/sqrt(d) and is exactly self-inverse.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class TransformState:
    """Random sign vectors for the Hadamard rotation."""

    signs_pre: mx.array  # ±1, shape (head_dim,) — applied before Hadamard
    signs_post: mx.array  # ±1, shape (head_dim,) — applied after Hadamard


def create_transform(head_dim: int, seed: int = 42) -> TransformState:
    """Create deterministic random rotation state.

    Generates two independent Rademacher (±1) sign vectors using the given seed.
    The resulting transform is an orthogonal rotation: it preserves norms and
    inner products exactly.
    """
    key = mx.random.key(seed)
    k1, k2 = mx.random.split(key)
    signs_pre = 2 * mx.random.bernoulli(key=k1, shape=(head_dim,)).astype(mx.float32) - 1
    signs_post = 2 * mx.random.bernoulli(key=k2, shape=(head_dim,)).astype(mx.float32) - 1
    return TransformState(signs_pre=signs_pre, signs_post=signs_post)


def forward_transform(x: mx.array, state: TransformState) -> mx.array:
    """Apply randomized Hadamard rotation: x → signs_post * H(signs_pre * x).

    Operates on the last axis. Supports batched inputs (..., head_dim).
    H = mx.hadamard_transform with default scale=1/sqrt(d), making it self-inverse.
    """
    return state.signs_post * mx.hadamard_transform(state.signs_pre * x)


def inverse_transform(x: mx.array, state: TransformState) -> mx.array:
    """Exact inverse: x → signs_pre * H(signs_post * x).

    Since H is self-inverse (with auto-scaling) and sign flips are self-inverse,
    the inverse just swaps which signs are applied pre vs post.
    """
    return state.signs_pre * mx.hadamard_transform(state.signs_post * x)
