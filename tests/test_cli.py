"""Tests for CLI entry point."""

from __future__ import annotations

import subprocess
import sys

import pytest


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


@pytest.mark.slow()
class TestSmoke:
    """End-to-end tests that require a model download."""

    def test_info_runs(self) -> None:
        result = _run_cli("info", "--model", "Qwen/Qwen2.5-0.5B-Instruct-4bit")
        assert result.returncode == 0
        assert "Layers:" in result.stdout
        assert "KV heads:" in result.stdout
