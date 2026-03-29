# mlx-turboquant

Apple-Silicon KV-cache compression for MLX/MLX-LM, inspired by the [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) research direction.

## What is this?

`mlx-turboquant` compresses the KV cache during LLM inference on Apple Silicon, reducing memory pressure and enabling longer context windows. It uses randomized Hadamard rotation with Lloyd-Max optimal codebooks to achieve near-Shannon-limit compression at 2-4 bits per coordinate.

> **Status:** Phase 2 (Compressed Cache Layer) complete -- `CompressedKVCache` with KVCache protocol compatibility, memory accounting, and GQA support, built on the Phase 1 codec (randomized Hadamard rotation + Lloyd-Max codebooks at 2-4 bits). This is a stage-1 prototype, not a full reproduction of Google's two-stage TurboQuant system. See the [implementation plan](docs/IMPLEMENTATION_PLAN.md) for details.

## How it works

1. **Random rotation** — Hadamard transform makes coordinate distributions data-oblivious
2. **Optimal quantization** — precomputed Lloyd-Max codebooks for the known Beta(d/2, d/2) distribution
3. **Bit-packing** — store quantized indices at 2-4 bits per coordinate
4. **On-demand decompression** — reconstruct KV vectors at attention time

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

## Development

```bash
uv run pytest                    # tests
uv run ruff check .              # lint
uv run mypy mlx_turboquant/      # type check
```

## References

1. [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — Google Research, March 2026
2. [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) — arXiv, April 2025
3. [MLX](https://github.com/ml-explore/mlx) / [MLX-LM](https://github.com/ml-explore/mlx-lm) — Apple's ML framework for Apple Silicon

## License

MIT
