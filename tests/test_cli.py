"""Tests for CLI entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
import types

import pytest

import mlx_turboquant.cli as cli
import mlx_turboquant.integration.generate_wrapper as generate_wrapper
from mlx_turboquant.constants import CANONICAL_SAMPLE_MODEL
from mlx_turboquant.integration.generate_wrapper import GenerationResult

EXTERNAL_MODEL_FAILURE_MARKERS = (
    "repository not found",
    "repositorynotfounderror",
    "401 unauthorized",
    "403 forbidden",
    "invalid username or password",
    "connecterror",
    "timed out",
    "temporary failure",
)


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mlx_turboquant", *args],
        capture_output=True,
        text=True,
    )


class TestHelp:
    def test_main_help(self) -> None:
        result = _run_cli("--help")
        assert result.returncode == 0
        assert "mlx-tq" in result.stdout or "mlx-turboquant" in result.stdout

    def test_generate_help(self) -> None:
        result = _run_cli("generate", "--help")
        assert result.returncode == 0
        assert "--model" in result.stdout
        assert "--prompt" in result.stdout
        assert "--cache-mode" in result.stdout
        assert "default: baseline" in result.stdout
        assert "experimental" in result.stdout

    def test_compare_help(self) -> None:
        result = _run_cli("compare", "--help")
        assert result.returncode == 0
        assert "--model" in result.stdout
        assert "--kv-bits" in result.stdout

    def test_info_help(self) -> None:
        result = _run_cli("info", "--help")
        assert result.returncode == 0
        assert "--model" in result.stdout

    def test_no_subcommand_exits_nonzero(self) -> None:
        result = _run_cli()
        assert result.returncode != 0


@pytest.mark.parametrize(
    ("command", "args"),
    [
        ("generate", ("--model", "dummy/model", "--prompt", "hello", "--kv-bits", "5")),
        ("compare", ("--model", "dummy/model", "--prompt", "hello", "--kv-bits", "1")),
        ("bench", ("--model", "dummy/model", "--kv-bits", "0")),
    ],
)
def test_invalid_kv_bits_rejected(command: str, args: tuple[str, ...]) -> None:
    result = _run_cli(command, *args)
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert "{2,3,4}" in result.stderr


class TestLoadModel:
    def test_wraps_repo_errors_without_traceback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_mlx_lm = types.ModuleType("mlx_lm")

        def _raise_repo_error(_model_name: str) -> tuple[object, object]:
            raise RuntimeError(
                "401 Unauthorized\nRepository Not Found\nInvalid username or password."
            )

        fake_mlx_lm.load = _raise_repo_error  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

        with pytest.raises(SystemExit) as exc_info:
            cli._load_model("bad/model")

        assert exc_info.value.code == (
            "Failed to load model 'bad/model': repo not found or access denied. "
            "Verify the model ID and Hugging Face authentication."
        )


def test_generate_defaults_to_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def _capture(args: argparse.Namespace) -> None:
        captured["cache_mode"] = args.cache_mode

    monkeypatch.setattr(cli, "_cmd_generate", _capture)
    monkeypatch.setattr(
        sys, "argv", ["mlx-tq", "generate", "--model", "dummy/model", "--prompt", "hi"]
    )

    cli.main()

    assert captured["cache_mode"] == "baseline"


def test_compare_uses_logical_cache_bytes_for_headline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = GenerationResult(
        text="baseline",
        tokens=[1, 2],
        tokens_generated=2,
        ttft_ms=1.0,
        decode_tokens_per_sec=2.0,
        cache_bytes=4096,
        cache_mode="baseline",
        cache_allocated_bytes=32768,
    )
    compressed = GenerationResult(
        text="compressed",
        tokens=[1, 2],
        tokens_generated=2,
        ttft_ms=1.0,
        decode_tokens_per_sec=2.0,
        cache_bytes=1024,
        cache_mode="compressed-3bit",
        cache_allocated_bytes=1024,
    )

    monkeypatch.setattr(cli, "_load_model", lambda _model: (object(), object()))
    monkeypatch.setattr(
        generate_wrapper,
        "generate_baseline",
        lambda *args, **kwargs: baseline,
    )
    monkeypatch.setattr(
        generate_wrapper,
        "generate_with_compressed_cache",
        lambda *args, **kwargs: compressed,
    )

    cli._cmd_compare(
        argparse.Namespace(
            model="dummy/model",
            prompt="hello",
            kv_bits=3,
            max_tokens=2,
            temp=0.0,
        )
    )
    output = capsys.readouterr().out
    assert "Compression: 4.0x" in output
    assert "Allocated buffers: 32.0 KB vs 1.0 KB (32.0x)" in output


@pytest.mark.slow()
class TestSmoke:
    """End-to-end tests that require a model download."""

    def test_info_runs(self) -> None:
        result = _run_cli("info", "--model", CANONICAL_SAMPLE_MODEL)
        if result.returncode != 0:
            stderr_lower = result.stderr.lower()
            if any(marker in stderr_lower for marker in EXTERNAL_MODEL_FAILURE_MARKERS):
                pytest.skip(f"external model access unavailable: {result.stderr.splitlines()[-1]}")
        assert result.returncode == 0, result.stderr
        assert "Layers:" in result.stdout
        assert "KV heads:" in result.stdout
