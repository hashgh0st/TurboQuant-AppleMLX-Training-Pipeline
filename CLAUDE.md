# CLAUDE.md — mlx-turboquant

## Project Overview

Apple-Silicon KV-cache compression for MLX/MLX-LM, inspired by TurboQuant research. Stage-1-only prototype targeting Qwen 2.5/3 models on M4 Mini 16 GB.

**Current status:** Phase 1 complete. Phase 2 (Compressed Cache Layer) is next.

## Tech Stack

- Python 3.11, MLX 0.31.1, MLX-LM 0.31.1, numpy
- uv for dependency management (hatchling build backend)
- ruff for linting/formatting, mypy (strict) for type checking, pytest for tests
- GitHub Actions CI on macOS (Python 3.11 + 3.12)

## Critical Architectural Rules

1. **CompressedKVCache must NOT expose a `bits` attribute.** MLX-LM's `scaled_dot_product_attention` dispatches to `quantized_matmul` (affine group quant) when `hasattr(cache, 'bits')` is True. Our rotation + Lloyd-Max approach is incompatible with that path.

2. **No Python loops in codec hot paths.** Encode/decode must use vectorized MLX ops only. Pattern: `mx.sum` of shifted values == bitwise OR when bits don't overlap (used in `pack()`).

3. **Pass `kv_bits=None` to MLX-LM's `generate_step()`** to disable its built-in affine KV quantization. Our CompressedKVCache handles compression internally.

4. **Use `Literal["reference", "metal"]` for backend selection**, not bare strings.

5. **`memory_accounting.py` owns calculations, `bench/memory.py` owns iteration and reporting.** Don't duplicate the formula or `cache.nbytes` reading in the bench module.

## Key MLX APIs

- `mx.hadamard_transform(a, scale=None)` — native WHT, works for dim = m * 2^k (covers head_dim 64 and 128)
- `mx.fast.scaled_dot_product_attention` — fused attention (used when cache has no `bits` attr)
- `mx.fast.metal_kernel` — custom Metal shader API for Phase 5 optimization
- `mx.quantize` / `mx.dequantize` — MLX's built-in affine quant (we do NOT use these; our codec is different)

## Commands

```bash
uv sync                          # install deps
uv run pytest                    # run fast tests
uv run pytest -m slow            # run integration tests (needs model download)
uv run ruff check .              # lint
uv run ruff format .             # format
uv run mypy mlx_turboquant/      # type check
```

## Code Conventions

- Line length: 100 characters
- Type annotations on all public functions (mypy strict mode)
- Tests in `tests/` mirroring source structure
- Slow tests (model downloads) marked with `@pytest.mark.slow`
- Codebook data stored as JSON in `mlx_turboquant/codec/data/`
- Version sourced from `importlib.metadata` (single source of truth in pyproject.toml)

## Package Structure

```
mlx_turboquant/
├── codec/       # codebooks, transforms, packbits, stage1_codec
├── cache/       # compressed_cache, cache_layout, memory_accounting
├── integration/ # mlx_lm_adapter, generate_wrapper
├── kernels/     # metal_pack, metal_attention (Phase 5)
├── bench/       # latency, quality, memory, prompts (Phase 4)
└── cli.py       # mlx-tq entry point (Phase 3)
```

## Documentation

- `docs/PRDv2.md` — Product requirements
- `docs/RESEARCHv2.md` — Research brief
- `docs/TDDv2.md` — Technical design
- `docs/IMPLEMENTATION_PLAN.md` — Phased build plan (Phase 1 complete)
