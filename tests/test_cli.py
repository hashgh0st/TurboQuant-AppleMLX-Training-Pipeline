"""Tests for CLI entry point."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

import mlx_turboquant.cli as cli
from mlx_turboquant.constants import CANONICAL_SAMPLE_MODEL

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
