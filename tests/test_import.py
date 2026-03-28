"""Smoke test: verify the package imports and has a version string."""

from __future__ import annotations


def test_import() -> None:
    import mlx_turboquant

    assert hasattr(mlx_turboquant, "__version__")
    assert isinstance(mlx_turboquant.__version__, str)
    assert len(mlx_turboquant.__version__) > 0
