"""Tests for benchmark modules."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import mlx_turboquant.bench.quality as quality_module
from mlx_turboquant.bench.memory import benchmark_memory
from mlx_turboquant.bench.promotion import (
    _NO_DIVERGE,
    ProfileVerdict,
    evaluate_profiles,
)
from mlx_turboquant.bench.quality import QualityResult, benchmark_quality
from mlx_turboquant.bench.report import generate_report
from mlx_turboquant.cache.memory_accounting import MemoryReport
from mlx_turboquant.integration.compression_profile import CompressionProfile
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
            cache_mode="compressed-3bit",
            kv_bits=3,
            value_kv_bits=3,
            backend="reference",
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

    def test_profiles_propagate_variant_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        baseline = self._result([1, 2, 3])
        compressed = GenerationResult(
            text="",
            tokens=[1, 2, 3],
            tokens_generated=3,
            ttft_ms=0.0,
            decode_tokens_per_sec=0.0,
            cache_bytes=0,
            cache_mode="compressed-k3v4bit-metal",
        )
        monkeypatch.setattr(quality_module, "generate_baseline", lambda *args, **kwargs: baseline)
        monkeypatch.setattr(
            quality_module,
            "generate_with_compressed_cache",
            lambda *args, **kwargs: compressed,
        )

        result = benchmark_quality(
            None,
            None,
            {"prompt": "hello"},
            profiles=[CompressionProfile(3, value_bits=4, backend="metal")],
        )[0]
        assert result.cache_mode == "compressed-k3v4bit-metal"
        assert result.value_kv_bits == 4
        assert result.backend == "metal"


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
                cache_mode="compressed-3bit",
                kv_bits=3,
                value_kv_bits=3,
                backend="reference",
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


def _qr(
    cache_mode: str = "compressed-3bit",
    token_match: float = 0.90,
    first_div: int = 15,
) -> QualityResult:
    return QualityResult(
        prompt_id="test",
        cache_mode=cache_mode,
        kv_bits=3,
        value_kv_bits=3,
        backend="reference",
        token_match_ratio=token_match,
        first_divergence_position=first_div,
        baseline_tokens=50,
        compressed_tokens=50,
    )


def _lr(cache_mode: str = "baseline", tok_s: float = 200.0) -> GenerationResult:
    return GenerationResult(
        text="",
        tokens=[1, 2, 3],
        tokens_generated=3,
        ttft_ms=10.0,
        decode_tokens_per_sec=tok_s,
        cache_bytes=0,
        cache_mode=cache_mode,
    )


class TestPromotion:
    def test_all_pass(self) -> None:
        qrs = [_qr(token_match=0.90, first_div=20)]
        lrs = [_lr("baseline", 200.0), _lr("compressed-3bit", 150.0)]
        verdicts = evaluate_profiles(qrs, lrs)
        assert len(verdicts) == 1
        assert verdicts[0].passes
        assert verdicts[0].failures == []

    def test_token_match_fails(self) -> None:
        qrs = [_qr(token_match=0.50)]
        lrs = [_lr("baseline", 200.0), _lr("compressed-3bit", 150.0)]
        verdicts = evaluate_profiles(qrs, lrs)
        assert not verdicts[0].passes
        assert any("token_match" in f for f in verdicts[0].failures)

    def test_first_diverge_fails(self) -> None:
        qrs = [_qr(first_div=3)]
        lrs = [_lr("baseline", 200.0), _lr("compressed-3bit", 150.0)]
        verdicts = evaluate_profiles(qrs, lrs)
        assert not verdicts[0].passes
        assert any("first_diverge" in f for f in verdicts[0].failures)

    def test_decode_slowdown_fails(self) -> None:
        qrs = [_qr()]
        lrs = [_lr("baseline", 300.0), _lr("compressed-3bit", 50.0)]  # 6x slowdown
        verdicts = evaluate_profiles(qrs, lrs)
        assert not verdicts[0].passes
        assert any("decode_slowdown" in f for f in verdicts[0].failures)

    def test_no_baseline_latency_skips_slowdown(self) -> None:
        qrs = [_qr()]
        lrs = [_lr("compressed-3bit", 100.0)]  # no baseline
        verdicts = evaluate_profiles(qrs, lrs)
        # Should still pass — slowdown can't be evaluated without baseline
        assert verdicts[0].passes

    def test_multiple_prompts_averaged(self) -> None:
        qrs = [
            _qr(token_match=0.95, first_div=20),
            _qr(token_match=0.65, first_div=5),  # bad prompt
        ]
        lrs = [_lr("baseline", 200.0), _lr("compressed-3bit", 150.0)]
        verdicts = evaluate_profiles(qrs, lrs)
        assert verdicts[0].avg_token_match == pytest.approx(0.80)
        assert verdicts[0].min_first_diverge == 5

    def test_identical_tokens_treated_as_passing(self) -> None:
        """first_divergence_position=-1 means perfect match."""
        qrs = [_qr(token_match=1.0, first_div=-1)]
        lrs = [_lr("baseline", 200.0), _lr("compressed-3bit", 150.0)]
        verdicts = evaluate_profiles(qrs, lrs)
        assert verdicts[0].passes
        assert verdicts[0].min_first_diverge == _NO_DIVERGE

    def test_uses_median_latency_not_best_case(self) -> None:
        """A profile with one lucky fast run should not pass on that alone."""
        qrs = [_qr(token_match=0.90, first_div=20)]
        lrs = [
            _lr("baseline", 200.0),
            _lr("baseline", 200.0),
            _lr("baseline", 200.0),
            # Compressed: two slow runs + one lucky fast run
            _lr("compressed-3bit", 40.0),  # 5x slowdown
            _lr("compressed-3bit", 45.0),  # 4.4x slowdown
            _lr("compressed-3bit", 180.0),  # lucky outlier — 1.1x
        ]
        verdicts = evaluate_profiles(qrs, lrs)
        # Median compressed = 45.0 tok/s. Median baseline = 200.0. Slowdown = 4.4x > 3x.
        assert not verdicts[0].passes
        assert any("decode_slowdown" in f for f in verdicts[0].failures)

    def test_multiple_profiles_separate_verdicts(self) -> None:
        qrs = [
            _qr(cache_mode="compressed-3bit", token_match=0.90, first_div=20),
            _qr(cache_mode="compressed-2bit", token_match=0.30, first_div=2),
        ]
        lrs = [
            _lr("baseline", 200.0),
            _lr("compressed-3bit", 150.0),
            _lr("compressed-2bit", 100.0),
        ]
        verdicts = evaluate_profiles(qrs, lrs)
        assert len(verdicts) == 2
        by_mode = {v.cache_mode: v for v in verdicts}
        assert by_mode["compressed-3bit"].passes
        assert not by_mode["compressed-2bit"].passes


class TestReportWithVerdicts:
    def test_promotion_section_in_markdown(self) -> None:
        verdicts = [
            ProfileVerdict(
                cache_mode="compressed-3bit",
                avg_token_match=0.90,
                min_first_diverge=20,
                decode_slowdown=1.5,
                passes=True,
                failures=[],
            ),
            ProfileVerdict(
                cache_mode="compressed-2bit",
                avg_token_match=0.30,
                min_first_diverge=2,
                decode_slowdown=2.0,
                passes=False,
                failures=["token_match 30% < 80%", "first_diverge 2 < 10"],
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_report([], [], [], tmpdir, model_name="test", verdicts=verdicts)
            md = (Path(tmpdir) / "BENCHMARKS.md").read_text()
            assert "## Promotion Status" in md
            assert "PASS" in md
            assert "FAIL" in md
            assert "compressed-3bit" in md
            assert "compressed-2bit" in md
