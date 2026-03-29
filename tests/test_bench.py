"""Tests for benchmark modules."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import mlx_turboquant.bench.quality as quality_module
from mlx_turboquant.bench.memory import benchmark_memory
from mlx_turboquant.bench.quality import QualityResult, benchmark_quality
from mlx_turboquant.bench.report import generate_report
from mlx_turboquant.cache.memory_accounting import MemoryReport
from mlx_turboquant.integration.generate_wrapper import GenerationResult


class TestMemoryBench:
    def test_returns_results(self) -> None:
        """Memory benchmark is pure calculation — no model needed."""
        results = benchmark_memory(
            num_layers=24,
            num_kv_heads=2,
            head_dim=64,
            seq_lengths=[512, 1024],
            kv_bits_list=[3, 4],
        )
        assert len(results) == 4  # 2 seq_lengths * 2 bit_widths

    def test_compression_ratios_positive(self) -> None:
        results = benchmark_memory(
            num_layers=24,
            num_kv_heads=2,
            head_dim=64,
            seq_lengths=[1024],
            kv_bits_list=[2, 3, 4],
        )
        for r in results:
            assert r.compression_ratio > 1.0

    def test_3bit_achieves_4x(self) -> None:
        results = benchmark_memory(
            num_layers=36,
            num_kv_heads=2,
            head_dim=128,
            seq_lengths=[4096],
            kv_bits_list=[3],
        )
        assert results[0].compression_ratio > 4.0


class TestQualityResult:
    def test_dataclass_fields(self) -> None:
        r = QualityResult(
            prompt_id="test",
            kv_bits=3,
            token_match_ratio=0.85,
            first_divergence_position=3,
            baseline_tokens=50,
            compressed_tokens=50,
        )
        assert r.token_match_ratio == 0.85
        assert r.first_divergence_position == 3


class TestBenchmarkQuality:
    @staticmethod
    def _result(tokens: list[int]) -> GenerationResult:
        return GenerationResult(
            text="",
            tokens=tokens,
            tokens_generated=len(tokens),
            ttft_ms=0.0,
            decode_tokens_per_sec=0.0,
            cache_bytes=0,
            cache_mode="baseline",
        )

    def test_identical_sequences_report_no_divergence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        baseline = self._result([1, 2, 3])
        compressed = self._result([1, 2, 3])
        monkeypatch.setattr(quality_module, "generate_baseline", lambda *args, **kwargs: baseline)
        monkeypatch.setattr(
            quality_module,
            "generate_with_compressed_cache",
            lambda *args, **kwargs: compressed,
        )

        result = benchmark_quality(None, None, {"prompt": "hello"}, kv_bits_list=[3])[0]
        assert result.first_divergence_position == -1

    def test_prefix_mismatch_reports_divergence_at_shared_prefix_length(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        baseline = self._result([1, 2, 3])
        compressed = self._result([1, 2])
        monkeypatch.setattr(quality_module, "generate_baseline", lambda *args, **kwargs: baseline)
        monkeypatch.setattr(
            quality_module,
            "generate_with_compressed_cache",
            lambda *args, **kwargs: compressed,
        )

        result = benchmark_quality(None, None, {"prompt": "hello"}, kv_bits_list=[3])[0]
        assert result.first_divergence_position == 2

    def test_middle_token_mismatch_reports_exact_index(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        baseline = self._result([1, 2, 3])
        compressed = self._result([1, 9, 3])
        monkeypatch.setattr(quality_module, "generate_baseline", lambda *args, **kwargs: baseline)
        monkeypatch.setattr(
            quality_module,
            "generate_with_compressed_cache",
            lambda *args, **kwargs: compressed,
        )

        result = benchmark_quality(None, None, {"prompt": "hello"}, kv_bits_list=[3])[0]
        assert result.first_divergence_position == 1

    def test_zero_length_sequences_report_zero_divergence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        empty = self._result([])
        monkeypatch.setattr(quality_module, "generate_baseline", lambda *args, **kwargs: empty)
        monkeypatch.setattr(
            quality_module,
            "generate_with_compressed_cache",
            lambda *args, **kwargs: empty,
        )

        result = benchmark_quality(None, None, {"prompt": "hello"}, kv_bits_list=[3])[0]
        assert result.first_divergence_position == 0


class TestReport:
    def test_generates_files(self) -> None:
        """Report generation works with mock data."""
        mem = [
            MemoryReport(
                baseline_bytes=1024,
                compressed_bytes=256,
                compression_ratio=4.0,
                bytes_per_token_baseline=100,
                bytes_per_token_compressed=25,
                num_layers=24,
                num_kv_heads=2,
                head_dim=64,
                kv_bits=3,
                seq_len=1024,
            )
        ]
        lat = [
            GenerationResult(
                text="hello",
                tokens=[1, 2, 3],
                tokens_generated=10,
                ttft_ms=50.0,
                decode_tokens_per_sec=100.0,
                cache_bytes=256,
                cache_mode="compressed-3bit",
            )
        ]
        qual = [
            QualityResult(
                prompt_id="test",
                kv_bits=3,
                token_match_ratio=0.9,
                first_divergence_position=5,
                baseline_tokens=50,
                compressed_tokens=48,
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            generate_report(mem, lat, qual, tmpdir, model_name="test-model")
            assert (Path(tmpdir) / "results.json").exists()
            assert (Path(tmpdir) / "BENCHMARKS.md").exists()

            # Verify JSON is valid
            with open(Path(tmpdir) / "results.json") as f:
                data = json.load(f)
            assert data["model"] == "test-model"
            assert len(data["memory"]) == 1
            assert len(data["latency"]) == 1
            assert len(data["quality"]) == 1

            # Verify Markdown has tables
            md = (Path(tmpdir) / "BENCHMARKS.md").read_text()
            assert "## Memory" in md
            assert "## Latency" in md
            assert "## Quality" in md
            assert "4.0x" in md
