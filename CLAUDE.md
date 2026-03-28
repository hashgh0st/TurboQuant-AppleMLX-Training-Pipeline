# CLAUDE.md — mlx-turboquant

## Project Overview

Apple-Silicon KV-cache compression for MLX/MLX-LM, inspired by TurboQuant research. Stage-1-only prototype targeting Qwen 2.5/3 models on M4 Mini 16 GB.

## Tech Stack

- Python 3.11+, MLX, MLX-LM, numpy
- uv for dependency management
- ruff for linting/formatting, mypy for type checking, pytest for tests

## Critical Architectural Rules

1. **CompressedKVCache must NOT expose a `bits` attribute.** MLX-LM's `scaled_dot_product_attention` dispatches to `quantized_matmul` (affine group quant) when `hasattr(cache, 'bits')` is True. Our rotation + Lloyd-Max approach is incompatible.

2. **No Python loops in codec hot paths.** Encode/decode must use vectorized MLX ops only.

3. **Pass `kv_bits=None` to MLX-LM's `generate_step()`** to disable its built-in affine KV quantization. Our CompressedKVCache handles compression internally.

## Commands

```bash
uv sync                          # install deps
uv run pytest                    # run fast tests
uv run pytest -m slow            # run integration tests (needs model download)
uv run ruff check .              # lint
uv run ruff format .             # format
uv run mypy mlx_turboquant/      # type check
uv run mlx-tq --help             # CLI
```

## Code Conventions

- Line length: 100 characters
- Type annotations on all public functions (mypy strict mode)
- Tests in `tests/` mirroring source structure
- Slow tests (model downloads) marked with `@pytest.mark.slow`
- Codebook data stored as JSON in `mlx_turboquant/codec/data/`

## Documentation

- `docs/PRDv2.md` — Product requirements
- `docs/RESEARCHv2.md` — Research brief
- `docs/TDDv2.md` — Technical design
- `docs/IMPLEMENTATION_PLAN.md` — Phased build plan
