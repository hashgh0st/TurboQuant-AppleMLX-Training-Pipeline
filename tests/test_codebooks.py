"""Tests for Lloyd-Max codebook generation and loading."""

from __future__ import annotations

import numpy as np
import pytest

from mlx_turboquant.codec.codebooks import (
    SUPPORTED_BITS,
    SUPPORTED_DIMS,
    CodebookData,
    _beta_pdf_shifted,
    build_lloyd_max_codebook,
    load_codebook,
    verify_codebook,
)


class TestBetaPdf:
    def test_integrates_to_one(self) -> None:
        """The Beta PDF on [-1, 1] should integrate to ~1.0."""
        for dim in (64, 128):
            grid = np.linspace(-1.0, 1.0, 10000)
            pdf = _beta_pdf_shifted(grid, dim)
            integral = float(np.trapezoid(pdf, grid))
            assert abs(integral - 1.0) < 1e-3, f"dim={dim}: integral={integral}"

    def test_symmetric(self) -> None:
        """Beta(d/2, d/2) is symmetric around 0."""
        grid = np.linspace(-1.0, 1.0, 10001)
        pdf = _beta_pdf_shifted(grid, 128)
        mid = len(grid) // 2
        np.testing.assert_allclose(pdf[:mid], pdf[-1 : mid : -1], atol=1e-10)

    def test_positive(self) -> None:
        """PDF should be non-negative everywhere."""
        grid = np.linspace(-1.0, 1.0, 10000)
        pdf = _beta_pdf_shifted(grid, 64)
        assert np.all(pdf >= 0)


class TestLloydMax:
    def test_converges(self) -> None:
        """Centroids should stabilize well before 300 iterations."""
        cb_100 = build_lloyd_max_codebook(128, 3, iterations=100)
        cb_300 = build_lloyd_max_codebook(128, 3, iterations=300)
        diff = max(
            abs(a - b) for a, b in zip(cb_100.centroids, cb_300.centroids, strict=True)
        )
        assert diff < 1e-8, f"Centroids still changing: max diff={diff}"

    def test_codebook_monotonicity(self) -> None:
        cb = build_lloyd_max_codebook(128, 3)
        for i in range(len(cb.centroids) - 1):
            assert cb.centroids[i] < cb.centroids[i + 1]
        for i in range(len(cb.boundaries) - 1):
            assert cb.boundaries[i] < cb.boundaries[i + 1]

    def test_codebook_symmetry(self) -> None:
        cb = build_lloyd_max_codebook(128, 3)
        n = len(cb.centroids)
        for i in range(n // 2):
            assert abs(cb.centroids[i] + cb.centroids[n - 1 - i]) < 1e-4

    def test_codebook_coverage(self) -> None:
        cb = build_lloyd_max_codebook(128, 3)
        assert abs(cb.boundaries[0] - (-1.0)) < 1e-6
        assert abs(cb.boundaries[-1] - 1.0) < 1e-6

    def test_correct_level_count(self) -> None:
        for bits in (2, 3, 4):
            cb = build_lloyd_max_codebook(64, bits, iterations=50)
            assert len(cb.centroids) == 2**bits
            assert len(cb.boundaries) == 2**bits + 1

    def test_distortion_decreases_with_bits(self) -> None:
        d2 = build_lloyd_max_codebook(128, 2, iterations=100)
        d3 = build_lloyd_max_codebook(128, 3, iterations=100)
        d4 = build_lloyd_max_codebook(128, 4, iterations=100)
        assert d4.distortion < d3.distortion < d2.distortion


class TestPrecomputed:
    @pytest.mark.parametrize("dim", SUPPORTED_DIMS)
    @pytest.mark.parametrize("bits", SUPPORTED_BITS)
    def test_load(self, dim: int, bits: int) -> None:
        cb = load_codebook(dim, bits)
        assert cb.dim == dim
        assert cb.bits == bits
        assert len(cb.centroids) == 2**bits

    @pytest.mark.parametrize("dim", SUPPORTED_DIMS)
    @pytest.mark.parametrize("bits", SUPPORTED_BITS)
    def test_verify(self, dim: int, bits: int) -> None:
        cb = load_codebook(dim, bits)
        assert verify_codebook(cb)

    def test_load_missing_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_codebook(999, 3)


class TestVerify:
    def test_rejects_wrong_level_count(self) -> None:
        cb = CodebookData(
            centroids=[0.1, 0.2, 0.3],  # should be 4 for 2-bit
            boundaries=[-1.0, -0.5, 0.0, 0.5, 1.0],
            dim=128,
            bits=2,
            distortion=0.01,
        )
        assert not verify_codebook(cb)

    def test_rejects_non_monotone(self) -> None:
        cb = CodebookData(
            centroids=[-0.5, -0.7, 0.5, 0.7],  # not monotone
            boundaries=[-1.0, -0.6, 0.0, 0.6, 1.0],
            dim=128,
            bits=2,
            distortion=0.01,
        )
        assert not verify_codebook(cb)
