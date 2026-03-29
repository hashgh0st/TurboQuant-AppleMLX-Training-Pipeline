# mlx-turboquant

Apple-Silicon KV-cache compression for MLX/MLX-LM, targeting Qwen 2.5/3 models on M4 Mini 16 GB.

## What is this?

`mlx-turboquant` compresses the KV cache during LLM inference on Apple Silicon, reducing memory pressure and enabling longer context windows.

**Status: pivoting quantization approach.** The original TurboQuant-inspired rotation + Lloyd-Max codec (Stages 1 & 2) does not produce usable output at 2-4 bits on real models. Testing on Qwen2.5-7B showed that per-element quantization error scales with vector norm, destroying attention logit differences. See [`docs/ROTATION_APPROACH_POSTMORTEM.md`](docs/ROTATION_APPROACH_POSTMORTEM.md) for the full analysis. The project is pivoting to per-channel group quantization (KIVI-style), which preserves per-element scale.

The infrastructure (duck-type KVCache protocol, attention sinks, incremental decode, Metal kernels, benchmark suite, promotion gates) carries forward.

## Memory Geometry (Qwen 2.5-0.5B, rotation codec)

| Metric | Baseline | 3-bit Compressed |
|--------|----------|-----------------|
| Occupied cache at 4K tokens | 48.0 MB | 11.2 MB (**4.3x**) |

Note: memory savings are real but the rotation codec's output quality is not usable. The per-channel codec (in development) will deliver both compression and quality.

## How it works

```
Input KV vectors (float16)
    |
    v
1. Normalize ── compute L2 norm, store as float16
    |
    v
2. Rotate ──── randomized Hadamard transform (data-oblivious)
    |
    v
3. Quantize ── Lloyd-Max optimal codebook for Beta(d/2, d/2)
    |
    v
4. Pack ────── 2/3/4-bit indices into uint32 words
    |
    v
Compressed storage (4-10x smaller)
```

At attention time, the process reverses: unpack, dequantize, inverse rotate, rescale. An optional Metal kernel fuses unpack+dequant for 1.6x decode speedup.

## Requirements

- Python 3.11+
- Apple Silicon Mac (M1/M2/M3/M4)
- [MLX](https://github.com/ml-explore/mlx) and [MLX-LM](https://github.com/ml-explore/mlx-lm)

## Getting started

```bash
git clone https://github.com/hashgh0st/TurboQuant-AppleMLX-Training-Pipeline.git
cd TurboQuant-AppleMLX-Training-Pipeline
uv sync
```

The GitHub repository is currently named `TurboQuant-AppleMLX-Training-Pipeline`, while the Python package and CLI remain `mlx-turboquant` and `mlx-tq`.

## Usage

### CLI

```bash
# Generate with the baseline cache (default)
mlx-tq generate --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --prompt "Explain KV cache compression"

# Opt into the experimental compressed cache
mlx-tq generate --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --prompt "Explain KV cache compression" \
    --cache-mode compressed --kv-bits 3

# Compare baseline vs compressed side-by-side
mlx-tq compare --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --prompt "What is the meaning of life?" --kv-bits 3

# Experimental mixed-precision compare: 3-bit keys, 4-bit values
mlx-tq compare --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --prompt "Explain attention" --kv-bits 3 --value-kv-bits 4

# Experimental Metal-backed compressed path
mlx-tq compare --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --prompt "Explain attention" --kv-bits 4 --backend metal

# Attention sink: keep first 4 tokens uncompressed for quality
mlx-tq generate --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --prompt "Explain attention" --cache-mode compressed --kv-bits 3 --sink-tokens 4

# Show model architecture and memory estimates
mlx-tq info --model mlx-community/Qwen2.5-0.5B-Instruct-4bit

# Run benchmarks and generate reports
mlx-tq bench --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --suite quick

# Run benchmarks with promotion gate (exits non-zero if any profile fails)
mlx-tq bench --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --suite quick --gate

# Calibrate codebooks to a specific model (fits to real activations)
mlx-tq calibrate --model mlx-community/Qwen2.5-0.5B-Instruct-4bit

# Use calibrated codebooks during generation
mlx-tq compare --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --prompt "What is the capital of France?" --kv-bits 3 \
    --calibrated-dir ~/.cache/mlx-turboquant/calibrated/mlx-community__Qwen2.5-0.5B-Instruct-4bit
```

`--kv-bits` is validated at the CLI boundary and currently supports `2`, `3`, or `4`.
`generate` defaults to `baseline`; `--cache-mode compressed` is an explicit experimental opt-in.
Compressed generation also supports experimental `--value-kv-bits`, `--backend metal`, and `--sink-tokens` tuning knobs for mixed-precision, Metal-backed, and attention-sink evaluation.
The `bench` subcommand supports `--gate` to enforce promotion thresholds (min 80% token match, min 10 first-diverge, max 3x slowdown) and exit non-zero on failure.
The `calibrate` subcommand fits Lloyd-Max codebooks to actual model activations, producing separate key/value codebooks that capture model-specific distribution patterns. Pass `--calibrated-dir` to `generate` or `compare` to use them.

### Python API

```python
from mlx_lm import load
from mlx_turboquant.integration.generate_wrapper import (
    generate_with_compressed_cache,
    generate_baseline,
)

model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

# Safe default
baseline = generate_baseline(
    model, tokenizer, "Hello world", max_tokens=100,
)
print(baseline.text)
print(f"Logical cache: {baseline.cache_bytes / 1024:.0f} KB")

# Experimental compressed generation
result = generate_with_compressed_cache(
    model, tokenizer, "Hello world", kv_bits=3, max_tokens=100,
)
print(result.text)
print(f"Logical cache: {result.cache_bytes / 1024:.0f} KB")
if result.cache_allocated_bytes is not None:
    print(f"Allocated cache: {result.cache_allocated_bytes / 1024:.0f} KB")
```

See [`examples/`](examples/) for more.
The new [`examples/evaluate_compression_profiles.py`](examples/evaluate_compression_profiles.py) script runs the recommended mixed-precision and Metal-backed experimental profiles end to end.

## Troubleshooting

- If model loading fails with a "repo not found or access denied" message, verify the model ID and your Hugging Face authentication.
- `compare` and benchmark reports now headline logical occupied cache bytes. Short runs can have much larger allocated backing buffers, which are shown separately when relevant.
- Slow CLI smoke tests depend on external model access and may skip automatically when the example model is unavailable.

## Project structure

```
mlx_turboquant/
  codec/        Phase 1: codebooks, transforms, packbits, stage1_codec, calibrate
  cache/        Phase 2: compressed_cache, memory_accounting, cache_layout
  integration/  Phase 3: mlx_lm_adapter, generate_wrapper
  bench/        Phase 4+: memory, latency, quality, prompts, report, promotion
  kernels/      Phase 5: metal_pack (fused unpack+dequant Metal kernels)
  cli.py        CLI entry point (generate, compare, info, bench, calibrate)
```

## Development

```bash
uv sync --all-extras          # install all deps
uv run pytest                 # 234 tests
uv run ruff check .           # lint
uv run mypy mlx_turboquant/   # type check (strict)
```

## References

1. [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — Google Research, March 2026
2. [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) — arXiv, April 2025
3. [MLX](https://github.com/ml-explore/mlx) / [MLX-LM](https://github.com/ml-explore/mlx-lm) — Apple's ML framework for Apple Silicon

## License

MIT
