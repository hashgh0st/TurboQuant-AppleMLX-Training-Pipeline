# Per-Channel KV Cache Compression: New Project Brief

## Purpose

This document is the foundation for a **separate project** implementing KIVI-style
per-channel group quantization for MLX/MLX-LM on Apple Silicon. It captures
everything learned from the mlx-turboquant rotation approach and provides a
concrete starting point.

## Why a Separate Project

The mlx-turboquant project is committed to the TurboQuant paper's rotation-based
approach (with pre-RoPE interception as the next research direction). Per-channel
quantization is a fundamentally different algorithm from a different paper (KIVI,
ICML 2024) and deserves its own identity.

## What We Learned From mlx-turboquant

### What Failed (Rotation Approach)

- Randomized Hadamard rotation + Lloyd-Max scalar quantization at 2-4 bits produces
  **unusable output** on real LLMs (Qwen2.5-7B, Qwen2.5-0.5B)
- Root cause: per-element error scales with vector NORM (~400-1000 for RoPE'd keys),
  not element magnitude. 70/128 dimensions have |error| > |original value|
- Attention logit error range (119) is 3x the meaningful signal range (38)
- QJL (Stage 2) residual correction makes it WORSE — correction variance (14.5 RMS)
  exceeds the MSE error (7.1 RMS) it tries to correct
- Calibrated codebooks don't help (same structural problem)
- Full analysis: `mlx-turboquant/docs/ROTATION_APPROACH_POSTMORTEM.md`

### What Works (Infrastructure to Reuse)

All of these transfer directly — copy from mlx-turboquant and adapt:

| Component | File | What It Does |
|-----------|------|-------------|
| `CompressedKVCache` | `cache/compressed_cache.py` | Duck-types MLX-LM's KVCache protocol. No `bits` attribute (avoids quantized_matmul dispatch). Handles allocation, growth, trim, state serialization. |
| Attention Sink | Same file | First N tokens in FP16. StreamingLLM insight — preserves quality on high-attention initial tokens. |
| Incremental Decode | Same file | O(1) per autoregressive step. Caches decoded output, only decodes new tokens. |
| Metal Kernel Framework | `kernels/metal_pack.py` | `mx.fast.metal_kernel` API. Per-word processing, shared memory codebook, threadgroup sizing. |
| Benchmark Suite | `bench/{quality,latency,memory,report,promotion}.py` | Token match ratio, first divergence, decode speed. Promotion gates with configurable thresholds. |
| Compression Profiles | `integration/compression_profile.py` | `CompressionProfile` dataclass, `resolve_profiles()` helper. Labels profiles as "research candidates" until they pass `--gate`. |
| CLI Structure | `cli.py` | argparse subcommands (generate, compare, info, bench, calibrate). Model loading, error formatting, result printing. |
| Calibration Pipeline | `codec/calibrate.py` | `KVCollectorCache` wraps real cache to intercept K/V. `collect_kv_samples()` runs prompts through model. Can be adapted to collect per-channel statistics instead of per-coordinate. |

### Critical Architectural Rules to Keep

1. **No `bits` attribute on the cache class.** MLX-LM dispatches to `quantized_matmul`
   when `hasattr(cache, 'bits')`. Our approach is incompatible with that path.

2. **Pass `kv_bits=None` to `generate_step()`** to disable MLX-LM's built-in quantization.

3. **Compute norms in float32.** Float16 sum-of-squares overflows for RoPE'd keys
   with elements ~170.

4. **No Python loops on the hot path.** Encode/decode must be fully vectorized MLX ops.

## Per-Channel Group Quantization Design

### Algorithm (KIVI, ICML 2024)

**Key insight**: Keys have channel-wise outliers (from RoPE), values have
token-wise patterns. Quantize keys per-channel and values per-token.

**Encode (per group of `group_size` elements)**:
```
1. Compute group min and max (or scale and zero-point)
2. scale = (max - min) / (2^bits - 1)
3. zero = min
4. quantized = round((x - zero) / scale)
5. Pack quantized values into uint8/uint16/uint32
6. Store scale + zero per group (2 x float16 = 4 bytes)
```

**Decode**:
```
1. Unpack quantized values
2. x_approx = quantized * scale + zero
```

**Storage per group at b bits, group_size=128**:
- Quantized: 128 * b / 8 bytes
- Parameters: 4 bytes (scale + zero in float16)
- Total: 128b/8 + 4 bytes
- vs FP16 baseline: 256 bytes

| Bits | Bytes/group | Compression vs FP16 |
|------|-------------|---------------------|
| 2 | 36 | 7.1x |
| 3 | 52 | 4.9x |
| 4 | 68 | 3.8x |
| 8 | 132 | 1.9x |

### Why Per-Channel Works Where Rotation Fails

The rotation approach gives every element the same absolute error regardless of
its magnitude. Per-channel quantization gives each group its own scale, so:

- Group with elements in [-170, 170] (RoPE high-freq): scale = 340/15 = 22.7, error ~ 11
- Group with elements in [-1, 1] (RoPE low-freq): scale = 2/15 = 0.13, error ~ 0.07

The low-magnitude group that carries discriminative attention signal gets
**160x smaller error** than the high-magnitude group. This is exactly what
the rotation approach destroys by distributing error uniformly.

### Asymmetric K/V Strategy (from KIVI)

- **Keys**: Quantize per-CHANNEL (each of the 128 dimensions independently across
  the token dimension). This handles RoPE's per-dimension magnitude variation.
- **Values**: Quantize per-TOKEN (each token's 128-dim vector independently).
  Values don't have RoPE and have more uniform per-element scales, but different
  tokens can have different magnitudes.

At inference time:
- During prefill: quantize all K/V at once (efficient batch operation)
- During decode: quantize the single new K/V token and append

### Key vs Value Quantization Axis

```
Keys (B, n_kv_heads, seq_len, head_dim):
  Quantize along seq_len axis with groups along head_dim
  → per-channel: group elements at same head_dim position across tokens
  → scale/zero shape: (n_kv_heads, 1, head_dim // group_size) per-channel-group

Values (B, n_kv_heads, seq_len, head_dim):
  Quantize along head_dim axis with groups within each token
  → per-token: group elements within each token's head_dim
  → scale/zero shape: (n_kv_heads, seq_len, head_dim // group_size) per-token-group
```

## Implementation Plan

### Phase 1: Core Codec

New file: `codec/group_quantizer.py`

```python
@dataclass
class GroupQuantConfig:
    head_dim: int
    bits: int  # 2, 3, 4, or 8
    group_size: int = 128  # elements per quantization group
    symmetric: bool = False  # True: scale only, False: scale + zero

@dataclass
class GroupQuantTensor:
    packed: mx.array      # (..., packed_dim) uint8/uint16/uint32
    scales: mx.array      # (..., num_groups) float16
    zeros: mx.array       # (..., num_groups) float16 (None if symmetric)
    config: GroupQuantConfig

class GroupQuantizer:
    def encode(self, x: mx.array) -> GroupQuantTensor: ...
    def decode(self, qt: GroupQuantTensor) -> mx.array: ...
```

### Phase 2: Asymmetric KV Cache

Replace `CompressedKVCache` internals with group-quantized storage:
- Keys: per-channel quantization (groups along head_dim, shared across tokens)
- Values: per-token quantization (groups within each token)
- Attention sinks and incremental decode carry forward unchanged

### Phase 3: Metal Kernel

Fused unpack + scale + zero-point kernel. Simpler than the rotation codec's
codebook lookup — just multiply-add:
```metal
float val = float(packed_value) * scale + zero;
```

### Phase 4: Calibration

Adapt the existing calibration pipeline:
- Instead of collecting post-rotation coordinate distributions, collect
  per-channel min/max statistics
- Use calibration data to determine optimal group boundaries or clipping ranges
- Optionally: per-layer adaptive bit-width based on channel sensitivity

### Phase 5: Benchmarks and Promotion

Run through the existing promotion gate pipeline:
- Token match >= 80%, first divergence >= 10, decode slowdown <= 3x
- At 4-bit per-channel: expect to PASS (KIVI reports <0.1 PPL degradation)
- At 2-bit: expect marginal pass or research-only status

## Expected Results

Based on KIVI (ICML 2024) published numbers on LLaMA-2-7B:

| Config | PPL Delta | Token Match (est.) | Compression |
|--------|-----------|-------------------|-------------|
| 4-bit K + 4-bit V | +0.1 | ~95%+ | 3.8x |
| 4-bit K + 2-bit V | +0.2 | ~90% | 5.3x |
| 2-bit K + 2-bit V | +0.5 | ~80% | 7.1x |

These are dramatically better than the rotation approach (which produces
0% usable output at any bit-width).

## Performance Expectations

**Encode**: Faster than rotation — no Hadamard transform. Just min/max + scale + round.

**Decode**: Faster than rotation — no inverse Hadamard. Just multiply + add.

**Memory**: Similar compression ratios. The per-group scale/zero overhead (4 bytes
per 128 elements) is comparable to the per-vector norm (2 bytes per vector) in
the rotation approach.

## Tech Stack

Same as mlx-turboquant:
- Python 3.11, MLX 0.31.1+, MLX-LM 0.31.1+, numpy
- uv + pyproject.toml + hatchling
- ruff + mypy (strict) + pytest
- GitHub Actions CI on macOS

## References

- **KIVI**: Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache", ICML 2024. https://arxiv.org/abs/2402.02750
- **KVQuant**: Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization", NeurIPS 2024. https://arxiv.org/abs/2401.18079
- **Rotation Postmortem**: `mlx-turboquant/docs/ROTATION_APPROACH_POSTMORTEM.md`
