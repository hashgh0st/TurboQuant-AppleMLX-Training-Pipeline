"""Lloyd-Max codebook generation and loading for Beta(d/2, d/2) distributions.

After a randomized Hadamard rotation, each coordinate of a unit-norm vector follows
Beta(d/2, d/2) on [-1, 1]. This module computes optimal Lloyd-Max quantizers for that
known distribution, stores them as JSON, and loads them at runtime.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Directory containing precomputed codebook JSON files
_DATA_DIR = Path(__file__).parent / "data"


@dataclass
class CodebookData:
    """Precomputed Lloyd-Max codebook for a specific (dim, bits) pair."""

    centroids: list[float]
    boundaries: list[float]
    dim: int
    bits: int
    distortion: float


def _beta_pdf_shifted(x: np.ndarray, d: int) -> np.ndarray:
    """Beta(d/2, d/2) PDF on [-1, 1].

    The standard Beta(a, b) PDF on [0, 1] is:
        f(t) = t^(a-1) * (1-t)^(b-1) / B(a, b)

    Shifted to [-1, 1] via t = (x+1)/2, with Jacobian 1/2:
        g(x) = f((x+1)/2) / 2

    For a = b = d/2 (symmetric), the PDF is symmetric around 0.
    """
    a = d / 2.0
    t = (x + 1.0) / 2.0
    log_beta = math.lgamma(a) + math.lgamma(a) - math.lgamma(2.0 * a)
    log_pdf = (a - 1.0) * np.log(np.clip(t, 1e-300, None)) + (a - 1.0) * np.log(
        np.clip(1.0 - t, 1e-300, None)
    )
    return np.exp(log_pdf - log_beta) / 2.0


def _bin_slice(
    grid: np.ndarray,
    pdf: np.ndarray,
    boundaries: np.ndarray,
    i: int,
    num_levels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice grid and pdf for bin i, including the right endpoint on the last bin."""
    if i < num_levels - 1:
        mask = (grid >= boundaries[i]) & (grid < boundaries[i + 1])
    else:
        mask = (grid >= boundaries[i]) & (grid <= boundaries[i + 1])
    return grid[mask], pdf[mask]


def build_lloyd_max_codebook(
    dim: int, bits: int, iterations: int = 300, grid_size: int = 10000
) -> CodebookData:
    """Compute optimal Lloyd-Max quantizer for Beta(dim/2, dim/2) on [-1, 1].

    The Lloyd-Max algorithm alternates:
      1. Update centroids: c_i = E[x | x in bin_i] = integral(x*pdf) / integral(pdf)
      2. Update boundaries: b_i = (c_{i-1} + c_i) / 2 (nearest-neighbor rule)

    Returns a CodebookData with centroids, boundaries, and achieved distortion.
    """
    num_levels = 1 << bits
    grid = np.linspace(-1.0, 1.0, grid_size)
    pdf = _beta_pdf_shifted(grid, dim)

    # Initialize boundaries uniformly
    boundaries = np.linspace(-1.0, 1.0, num_levels + 1)
    centroids = np.zeros(num_levels)

    for _ in range(iterations):
        # Update centroids: conditional expectation within each bin
        for i in range(num_levels):
            g, p = _bin_slice(grid, pdf, boundaries, i, num_levels)
            denom = float(np.trapezoid(p, g))
            if denom > 1e-15:
                centroids[i] = float(np.trapezoid(g * p, g)) / denom
            else:
                centroids[i] = (boundaries[i] + boundaries[i + 1]) / 2.0

        # Update boundaries: midpoints between adjacent centroids
        for i in range(1, num_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

    # Compute distortion: E[(x - Q(x))^2]
    distortion = 0.0
    for i in range(num_levels):
        g, p = _bin_slice(grid, pdf, boundaries, i, num_levels)
        distortion += float(np.trapezoid((g - centroids[i]) ** 2 * p, g))

    return CodebookData(
        centroids=centroids.tolist(),
        boundaries=boundaries.tolist(),
        dim=dim,
        bits=bits,
        distortion=float(distortion),
    )


def save_codebook(cb: CodebookData, path: Path) -> None:
    """Save codebook to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(cb), f, indent=2)


def load_codebook(dim: int, bits: int) -> CodebookData:
    """Load precomputed codebook from package data directory."""
    path = _DATA_DIR / f"{dim}_{bits}.json"
    if not path.exists():
        msg = (
            f"No precomputed codebook for dim={dim}, bits={bits}. "
            f"Expected file: {path}. Run `python -m mlx_turboquant.codec.codebooks` to generate."
        )
        raise FileNotFoundError(msg)
    with open(path) as f:
        data = json.load(f)
    return CodebookData(**data)


def verify_codebook(cb: CodebookData) -> bool:
    """Verify codebook integrity: monotonicity, symmetry, coverage, level count."""
    c = cb.centroids
    b = cb.boundaries
    expected_levels = 1 << cb.bits

    if len(c) != expected_levels:
        return False
    if len(b) != expected_levels + 1:
        return False

    # Monotonicity
    if not all(c[i] < c[i + 1] for i in range(len(c) - 1)):
        return False
    if not all(b[i] < b[i + 1] for i in range(len(b) - 1)):
        return False

    # Boundary coverage
    if abs(b[0] - (-1.0)) > 1e-6 or abs(b[-1] - 1.0) > 1e-6:
        return False

    # Symmetry: centroids[i] ≈ -centroids[n-1-i] for symmetric Beta
    n = len(c)
    return all(abs(c[i] + c[n - 1 - i]) <= 1e-4 for i in range(n // 2))


# Supported configurations: {dim} x {bits}
SUPPORTED_DIMS = (64, 128)
SUPPORTED_BITS = (2, 3, 4)


if __name__ == "__main__":
    # Generate all codebooks and save to data directory
    for dim in SUPPORTED_DIMS:
        for bits in SUPPORTED_BITS:
            print(f"Generating codebook: dim={dim}, bits={bits}...", end=" ", flush=True)
            cb = build_lloyd_max_codebook(dim, bits)
            path = _DATA_DIR / f"{dim}_{bits}.json"
            save_codebook(cb, path)
            ok = verify_codebook(cb)
            print(f"distortion={cb.distortion:.6f}, verified={ok}")
    print("Done. All codebooks saved to", _DATA_DIR)
