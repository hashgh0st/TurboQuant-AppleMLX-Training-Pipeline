"""Tests for codebook calibration: empirical fitting, storage, loading, codec integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlx_turboquant.codec.calibrate import build_empirical_codebook
from mlx_turboquant.codec.codebooks import (
    CodebookData,
    _model_slug,
    calibrated_codebook_dir,
    load_calibrated_codebook,
    load_codebook,
    load_codebook_with_fallback,
    save_codebook,
    verify_codebook,
)
from mlx_turboquant.codec.stage1_codec import CodecConfig, Stage1Codec


class TestBuildEmpiricalCodebook:
    def test_uniform_samples(self) -> None:
        """Uniform samples should produce approximately uniform centroids."""
        rng = np.random.default_rng(42)
        samples = rng.uniform(-1.0, 1.0, size=100_000)
        cb = build_empirical_codebook(samples, dim=64, bits=2, iterations=100)
        assert len(cb.centroids) == 4
        assert len(cb.boundaries) == 5
        # Centroids should be roughly at -0.5, -0.17, 0.17, 0.5
        for c in cb.centroids:
            assert -1.0 < c < 1.0

    def test_gaussian_concentrates_centroids(self) -> None:
        """Gaussian samples should place more centroids near zero."""
        rng = np.random.default_rng(42)
        samples = np.clip(rng.normal(0, 0.3, size=100_000), -1.0, 1.0)
        cb = build_empirical_codebook(samples, dim=128, bits=3, iterations=100)
        # Middle centroids should be closer together than outer ones
        gaps = [cb.centroids[i + 1] - cb.centroids[i] for i in range(len(cb.centroids) - 1)]
        assert gaps[3] < gaps[0], "Middle gap should be smaller than outer gap"

    def test_correct_level_count(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.uniform(-1.0, 1.0, size=10_000)
        for bits in [2, 3, 4]:
            cb = build_empirical_codebook(samples, dim=64, bits=bits)
            assert len(cb.centroids) == (1 << bits)
            assert len(cb.boundaries) == (1 << bits) + 1

    def test_monotonicity(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.uniform(-1.0, 1.0, size=10_000)
        cb = build_empirical_codebook(samples, dim=64, bits=3)
        for i in range(len(cb.centroids) - 1):
            assert cb.centroids[i] < cb.centroids[i + 1]
        for i in range(len(cb.boundaries) - 1):
            assert cb.boundaries[i] < cb.boundaries[i + 1]

    def test_boundary_coverage(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.uniform(-1.0, 1.0, size=10_000)
        cb = build_empirical_codebook(samples, dim=64, bits=3)
        assert abs(cb.boundaries[0] - (-1.0)) < 1e-6
        assert abs(cb.boundaries[-1] - 1.0) < 1e-6

    def test_distortion_decreases_with_bits(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.uniform(-1.0, 1.0, size=50_000)
        d2 = build_empirical_codebook(samples, dim=64, bits=2).distortion
        d3 = build_empirical_codebook(samples, dim=64, bits=3).distortion
        d4 = build_empirical_codebook(samples, dim=64, bits=4).distortion
        assert d4 < d3 < d2

    def test_convergence(self) -> None:
        """100 and 300 iterations should give nearly identical centroids."""
        rng = np.random.default_rng(42)
        samples = rng.uniform(-1.0, 1.0, size=50_000)
        cb100 = build_empirical_codebook(samples, dim=64, bits=3, iterations=100)
        cb300 = build_empirical_codebook(samples, dim=64, bits=3, iterations=300)
        for c1, c2 in zip(cb100.centroids, cb300.centroids, strict=True):
            assert abs(c1 - c2) < 1e-4

    def test_empty_bin_handling(self) -> None:
        """Concentrated samples should not crash even if some bins are initially empty."""
        samples = np.full(1000, 0.5)  # all samples at 0.5
        cb = build_empirical_codebook(samples, dim=64, bits=2)
        assert len(cb.centroids) == 4

    def test_passes_verify_without_symmetry(self) -> None:
        rng = np.random.default_rng(42)
        # Asymmetric: shifted distribution
        samples = np.clip(rng.normal(0.2, 0.3, size=100_000), -1.0, 1.0)
        cb = build_empirical_codebook(samples, dim=64, bits=3)
        assert verify_codebook(cb, symmetric=False)


class TestCalibratedCodebookStorage:
    def test_model_slug(self) -> None:
        assert _model_slug("mlx-community/Qwen2.5-0.5B") == "mlx-community__Qwen2.5-0.5B"
        assert _model_slug("local-model") == "local-model"

    def test_calibrated_codebook_dir(self) -> None:
        d = calibrated_codebook_dir("mlx-community/Qwen2.5-0.5B")
        assert "mlx-community__Qwen2.5-0.5B" in str(d)

    def test_calibrated_codebook_dir_custom_base(self, tmp_path: Path) -> None:
        d = calibrated_codebook_dir("my/model", base_dir=tmp_path)
        assert d == tmp_path / "my__model"

    def test_load_calibrated_returns_none_when_missing(self, tmp_path: Path) -> None:
        result = load_calibrated_codebook(64, 3, "key", "missing/model", base_dir=tmp_path)
        assert result is None

    def test_load_calibrated_reads_json(self, tmp_path: Path) -> None:
        cb = CodebookData(
            centroids=[-0.5, -0.1, 0.1, 0.5],
            boundaries=[-1.0, -0.3, 0.0, 0.3, 1.0],
            dim=64,
            bits=2,
            distortion=0.01,
        )
        model_dir = tmp_path / "test__model"
        save_codebook(cb, model_dir / "64_2_key.json")
        loaded = load_calibrated_codebook(64, 2, "key", "test/model", base_dir=tmp_path)
        assert loaded is not None
        assert loaded.centroids == cb.centroids

    def test_fallback_prefers_calibrated(self, tmp_path: Path) -> None:
        cb = CodebookData(
            centroids=[-0.6, -0.2, 0.2, 0.6],
            boundaries=[-1.0, -0.4, 0.0, 0.4, 1.0],
            dim=64,
            bits=2,
            distortion=0.005,
        )
        model_dir = tmp_path / "test__model"
        save_codebook(cb, model_dir / "64_2_key.json")
        loaded = load_codebook_with_fallback(
            64, 2, "key", model_name="test/model", calibrated_dir=tmp_path
        )
        assert loaded.centroids == cb.centroids

    def test_fallback_returns_precomputed(self) -> None:
        loaded = load_codebook_with_fallback(64, 3, "key", model_name=None)
        precomputed = load_codebook(64, 3)
        assert loaded.centroids == precomputed.centroids

    def test_fallback_when_no_calibrated(self) -> None:
        loaded = load_codebook_with_fallback(
            64, 3, "key", model_name="nonexistent/model"
        )
        precomputed = load_codebook(64, 3)
        assert loaded.centroids == precomputed.centroids


class TestCodecWithCalibration:
    def test_codec_uses_calibrated_when_available(self, tmp_path: Path) -> None:
        """Codec with model_name should load calibrated codebook."""
        # Create a calibrated codebook with distinctive centroids
        precomputed = load_codebook(64, 2)
        custom = CodebookData(
            centroids=[-0.7, -0.3, 0.3, 0.7],
            boundaries=[-1.0, -0.5, 0.0, 0.5, 1.0],
            dim=64,
            bits=2,
            distortion=0.01,
        )
        model_dir = tmp_path / "test__model"
        save_codebook(custom, model_dir / "64_2_key.json")

        config = CodecConfig(
            head_dim=64, bits=2, model_name="test/model",
            kv_type="key", calibrated_dir=tmp_path,
        )
        codec = Stage1Codec(config)
        # Centroids should match our custom codebook, not precomputed
        centroids_list = codec.centroids.tolist()
        assert abs(centroids_list[0] - (-0.7)) < 1e-5
        assert abs(centroids_list[0] - precomputed.centroids[0]) > 0.01

    def test_codec_falls_back_without_model_name(self) -> None:
        """Codec without model_name should use theoretical codebook."""
        config = CodecConfig(head_dim=64, bits=3)
        codec = Stage1Codec(config)
        precomputed = load_codebook(64, 3)
        assert codec.centroids.tolist() == pytest.approx(precomputed.centroids, abs=1e-6)


class TestVerifyCodebookSymmetry:
    def test_precomputed_passes_with_symmetry(self) -> None:
        cb = load_codebook(64, 3)
        assert verify_codebook(cb, symmetric=True)

    def test_asymmetric_fails_with_symmetry(self) -> None:
        cb = CodebookData(
            centroids=[-0.6, -0.1, 0.2, 0.7],
            boundaries=[-1.0, -0.35, 0.05, 0.45, 1.0],
            dim=64,
            bits=2,
            distortion=0.01,
        )
        assert not verify_codebook(cb, symmetric=True)
        assert verify_codebook(cb, symmetric=False)
