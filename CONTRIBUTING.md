# Contributing to mlx-turboquant

## Development Setup

```bash
git clone https://github.com/dak/mlx-turboquant.git
cd mlx-turboquant
uv sync --all-extras
uv run pre-commit install
```

## Code Quality

All code must pass before merging:

```bash
uv run ruff check .        # lint
uv run ruff format --check . # format check
uv run mypy mlx_turboquant/ # type check
uv run pytest               # tests
```

## Guidelines

- Type-annotate all public functions
- Write tests for new functionality
- Keep the codec hot path free of Python loops (use vectorized MLX ops)
- Do not expose a `bits` attribute on cache classes (see TDD for explanation)
- Label all benchmark claims as estimates unless validated locally

## Running Slow Tests

Integration tests that require model downloads are marked with `@pytest.mark.slow`:

```bash
uv run pytest -m slow
```

These tests depend on external model availability and Hugging Face access. The CLI smoke test will skip automatically when the example model cannot be reached or authenticated.
