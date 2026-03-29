"""Model-specific codebook calibration via sample-based Lloyd-Max.

Collects post-rotation KV coordinate distributions from real model activations
and fits optimal codebooks to the empirical distribution. This closes the gap
between the theoretical Beta(d/2, d/2) assumption and actual model behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache, make_prompt_cache

from mlx_turboquant.bench.prompts import BENCHMARK_PROMPTS
from mlx_turboquant.codec.codebooks import CodebookData
from mlx_turboquant.codec.transforms import TransformState, create_transform, forward_transform


@dataclass
class CalibrationConfig:
    """Configuration for codebook calibration."""

    head_dim: int
    bits_list: tuple[int, ...] = (2, 3, 4)
    seed: int = 42
    max_tokens: int = 256
    lloyd_max_iterations: int = 300


def build_empirical_codebook(
    samples: np.ndarray,
    dim: int,
    bits: int,
    iterations: int = 300,
) -> CodebookData:
    """Fit Lloyd-Max codebook from empirical 1-D samples.

    Sample-based Lloyd-Max (1-D k-means with ordered clusters):
      1. Initialize boundaries uniformly on [-1, 1]
      2. Assign samples to bins via np.digitize
      3. Centroid = mean(samples in bin), empty bin -> midpoint fallback
      4. Boundary = midpoint of adjacent centroids
    """
    num_levels = 1 << bits
    samples = np.clip(samples.ravel(), -1.0, 1.0)

    boundaries = np.linspace(-1.0, 1.0, num_levels + 1)
    centroids = np.zeros(num_levels)

    for _ in range(iterations):
        prev_centroids = centroids.copy()
        bin_indices = np.digitize(samples, boundaries[1:-1])
        for i in range(num_levels):
            mask = bin_indices == i
            if np.any(mask):
                centroids[i] = float(np.mean(samples[mask]))
            else:
                centroids[i] = (boundaries[i] + boundaries[i + 1]) / 2.0

        for i in range(1, num_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

        if np.max(np.abs(centroids - prev_centroids)) < 1e-7:
            break

    # Final assignment for distortion
    bin_indices = np.digitize(samples, boundaries[1:-1])
    nearest = centroids[bin_indices]
    distortion = float(np.mean((samples - nearest) ** 2))

    return CodebookData(
        centroids=centroids.tolist(),
        boundaries=boundaries.tolist(),
        dim=dim,
        bits=bits,
        distortion=distortion,
    )


class KVCollectorCache:
    """KVCache wrapper that intercepts raw K/V tensors for calibration.

    Wraps a real mlx_lm KVCache, records post-rotation coordinate samples
    (not raw tensors) to bound memory usage. Each step's K/V is immediately
    normalized, rotated, and converted to a small numpy array.
    """

    def __init__(self, inner: KVCache, transform: TransformState) -> None:
        self._inner = inner
        self._transform = transform
        self.key_coords: list[np.ndarray] = []
        self.value_coords: list[np.ndarray] = []

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        # Rotate and flatten immediately — don't hold raw MLX tensors
        self.key_coords.append(_rotate_and_flatten([keys], self._transform))
        self.value_coords.append(_rotate_and_flatten([values], self._transform))
        result: tuple[mx.array, mx.array] = self._inner.update_and_fetch(keys, values)  # type: ignore[no-untyped-call]
        return result

    @property
    def offset(self) -> int:
        return self._inner.offset

    @offset.setter
    def offset(self, v: int) -> None:
        self._inner.offset = v

    @property
    def state(self) -> Any:
        return self._inner.state

    @state.setter
    def state(self, v: Any) -> None:
        self._inner.state = v

    @property
    def meta_state(self) -> Any:
        return self._inner.meta_state

    @meta_state.setter
    def meta_state(self, v: Any) -> None:
        self._inner.meta_state = v

    def is_trimmable(self) -> bool:
        result: bool = self._inner.is_trimmable()  # type: ignore[no-untyped-call]
        return result

    def trim(self, n: int) -> int:
        result: int = self._inner.trim(n)  # type: ignore[no-untyped-call]
        return result

    def make_mask(self, N: int, **kwargs: Any) -> Any:
        return self._inner.make_mask(N, **kwargs)  # type: ignore[no-untyped-call]

    def empty(self) -> bool:
        result: bool = self._inner.empty()  # type: ignore[no-untyped-call]
        return result


def _rotate_and_flatten(
    tensors: list[mx.array],
    transform: TransformState,
) -> np.ndarray:
    """Normalize, rotate, and flatten KV tensors to 1-D coordinate samples.

    Batches all tensors into a single MLX graph for efficient dispatch.
    """
    if not tensors:
        return np.array([], dtype=np.float32)
    # Batch: concatenate all tensors, normalize+rotate in one graph
    stacked = mx.concatenate(
        [t.reshape(-1, t.shape[-1]).astype(mx.float32) for t in tensors], axis=0
    )
    norms = mx.linalg.norm(stacked, axis=-1, keepdims=True)
    normed = stacked / (norms + 1e-8)
    rotated = forward_transform(normed, transform)
    return np.array(rotated).ravel()


def _greedy_sampler(x: mx.array) -> mx.array:
    return mx.argmax(x, axis=-1)


def collect_kv_samples(
    model: Any,
    tokenizer: Any,
    prompts: dict[str, str],
    *,
    head_dim: int,
    seed: int = 42,
    max_tokens: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model forward passes and collect post-rotation KV coordinate samples.

    Returns (key_samples, value_samples) -- 1-D numpy arrays of coordinates
    from the post-rotation domain, suitable for fitting empirical codebooks.
    """
    transform = create_transform(head_dim, seed)
    key_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []

    for prompt_text in prompts.values():
        prompt_tokens = mx.array(tokenizer.encode(prompt_text))
        inner_cache = make_prompt_cache(model)
        collector_cache = [KVCollectorCache(layer, transform) for layer in inner_cache]

        for _token_id, _logprobs in generate_step(
            prompt=prompt_tokens,
            model=model,
            sampler=_greedy_sampler,
            prompt_cache=collector_cache,
            kv_bits=None,
            max_tokens=max_tokens,
        ):
            pass

        for layer in collector_cache:
            if layer.key_coords:
                key_parts.append(np.concatenate(layer.key_coords))
            if layer.value_coords:
                value_parts.append(np.concatenate(layer.value_coords))

    key_samples = np.concatenate(key_parts) if key_parts else np.array([], dtype=np.float32)
    value_samples = np.concatenate(value_parts) if value_parts else np.array([], dtype=np.float32)
    return key_samples, value_samples


def calibrate_codebooks(
    model: Any,
    tokenizer: Any,
    config: CalibrationConfig,
    prompts: dict[str, str] | None = None,
) -> dict[str, CodebookData]:
    """Full calibration pipeline: collect samples, fit codebooks.

    Returns dict mapping ``"{dim}_{bits}_{kv_type}"`` to CodebookData,
    e.g. ``"128_3_key"``, ``"128_3_value"``.
    """
    if prompts is None:
        prompts = BENCHMARK_PROMPTS

    key_samples, value_samples = collect_kv_samples(
        model,
        tokenizer,
        prompts,
        head_dim=config.head_dim,
        seed=config.seed,
        max_tokens=config.max_tokens,
    )

    codebooks: dict[str, CodebookData] = {}
    for bits in config.bits_list:
        for kv_type, samples in [("key", key_samples), ("value", value_samples)]:
            name = f"{config.head_dim}_{bits}_{kv_type}"
            codebooks[name] = build_empirical_codebook(
                samples,
                dim=config.head_dim,
                bits=bits,
                iterations=config.lloyd_max_iterations,
            )

    return codebooks
