# Rotation-Based KV Cache Compression: Investigation Log

## Summary

The TurboQuant rotation + Lloyd-Max approach initially produced unusable output
at 2-4 bits on real LLMs. After investigating QJL (Stage 2), pre-RoPE
interception, and calibrated codebooks — all of which failed — we discovered
that working implementations (llama.cpp, turboquant_plus) use a **norm
correction** technique we had missed. Applying norm correction improved
attention logit error range by **25x**, producing coherent output for the
first time.

**Current state**: 4-bit produces coherent output that degrades over long
sequences. Further improvements documented in `QUALITY_IMPROVEMENT_ROADMAP.md`.

## What We Built

- **Stage 1**: Normalize to unit vector, randomized Hadamard rotation, per-coordinate
  Lloyd-Max quantization (2-4 bit codebooks precomputed for Beta(d/2,d/2)), bit-packing.
  Decode reverses: unpack, codebook lookup, inverse rotation, rescale by stored norm.

- **Stage 2 (QJL)**: After Stage 1 MSE quantization at (b-1) bits, compute residual,
  project through random matrix S, store 1-bit signs + residual norm. During decode,
  add correction: `sqrt(pi/2)/d * gamma * S^T * signs`. Uses b total bits.

- **Infrastructure**: Metal kernel fusion, attention sinks, incremental decode,
  calibrated codebooks, promotion gates, duck-type KVCache protocol.

## What We Observed

### Baseline vs Compressed Output (Qwen2.5-7B, 4-bit, no QJL)

**Baseline**: "KV-cache compression is a technique used in transformer models to
reduce the memory footprint of storing key-value pairs during inference..."

**Compressed**: "perdida perdida de en modelosos de perdida perdida perdida..."

### Root Cause: Per-Element Error Scales With Vector Norm

Diagnostic on real model keys (head 0, layer 0, Qwen 7B):

| Metric | Value |
|--------|-------|
| Key vector norm | 466.5 |
| Max element magnitude | 170.2 |
| Mean element magnitude | 16.2 |
| Max absolute error | 10.9 |
| Mean absolute error | 2.5 |
| Elements where |error| > |original| | **70 / 128** |

The rotation approach normalizes to unit vectors, quantizes, then rescales by
the original norm. This distributes quantization error UNIFORMLY across all
128 dimensions, regardless of their original magnitude. Dimensions with small
values (which carry discriminative signal for attention) are destroyed.

### Attention Logit Analysis

For 6 consecutive tokens:

| Token | Original Logit | Compressed Logit | Error |
|-------|---------------|------------------|-------|
| 0 | 19,199 | 19,351 | +152 |
| 1 | 19,214 | 19,447 | +233 |
| 2 | 19,197 | 19,459 | +262 |
| 3 | 19,176 | 19,406 | +230 |
| 4 | 19,190 | 19,437 | +247 |
| 5 | 19,180 | 19,450 | +271 |

- **Meaningful signal range**: 38 (19,214 - 19,176)
- **Error range**: 119 (271 - 152)
- **Error range / signal range**: 3.1x

Softmax depends on RELATIVE logit differences. The error range (119) is 3x
the actual signal (38), so the attention distribution is completely wrong.

### QJL Made It Worse

At 4-bit total (3-bit MSE + 1-bit QJL):

| Metric | MSE Only (4-bit) | With QJL (3+1 bit) |
|--------|-----------------|---------------------|
| Logit error range | 121.9 | **1,173.7** |
| Per-element correction RMS | n/a | 14.5 |
| Per-element MSE error RMS | 7.1 | 7.1 (MSE part) |

The QJL correction vector has 35% of the unit-vector magnitude. After scaling by
the key norm (466), the per-element correction noise (14.5) is 2x the MSE error
(7.1). The correction is mathematically unbiased (E[<q, correction>] = <q, r>)
but its variance overwhelms the signal.

## Why This Approach Fails

The core assumption is that randomized Hadamard rotation makes all coordinates
"equally important" (Beta(d/2,d/2) distributed). This is true. But after
dequantization and inverse rotation, the error is distributed proportional to
the VECTOR NORM, not the per-element magnitude.

For RoPE'd keys:
- High-frequency RoPE components produce elements with magnitude ~170
- Low-frequency components produce elements with magnitude ~1
- Both get the same absolute error (~8.6 per element at 4-bit)
- The low-magnitude elements (which carry the discriminative signal) are wiped out

## What Production Systems Do Differently

| System | Approach | Why It Works |
|--------|----------|--------------|
| **KIVI** (ICML 2024) | Per-channel quantization (keys), per-token (values) | Each dimension gets its own scale; small channels get small errors |
| **KVQuant** (NeurIPS 2024) | Pre-RoPE quantization, per-channel, non-uniform | Avoids RoPE magnitude explosion entirely |
| **QServe** (MLSys 2025) | W4A8KV4 system co-design | Integrated weight + activation + KV quantization |
| **GEAR** (NeurIPS 2024) | Low-precision majority + outlier handling | Separates outliers from bulk |

All of these handle heterogeneous element magnitudes. The rotation approach
does not.

## What We're Keeping

The infrastructure built for the rotation approach transfers directly:
- `CompressedKVCache` duck-type protocol (no `bits` attribute)
- Attention sinks (FP16 first N tokens)
- Incremental decode (O(1) per step)
- Metal kernel framework
- Promotion gates and benchmark suite
- Calibration pipeline (for distribution analysis)

## The Fix: Norm Correction (from llama.cpp community)

Research into how atomic.chat (a macOS wrapper around llama.cpp) and other
working implementations handle TurboQuant revealed three critical fixes we
were missing. The most important:

### Fix 1: Norm Correction (25x improvement)

**Before**: Store `original_norm`, multiply during decode.
**After**: Store `original_norm / ||reconstruction_unit||`, multiply during decode.

This ensures `||reconstructed|| == ||original||` exactly. The reconstruction
unit vector has norm != 1.0 due to quantization (centroids don't perfectly
reconstruct the unit sphere). Without correction, the norm mismatch creates
non-uniform scaling that destroys attention logit differences.

**Results on Qwen2.5-7B-Instruct-4bit at 4-bit**:

| Metric | Without Norm Correction | With Norm Correction |
|--------|------------------------|---------------------|
| Attention logit error range | 121.9 | **4.9** |
| Signal range | 38 | 38 |
| Error/signal ratio | 3.1x (broken) | **0.13x (works)** |
| Mean error | +216 (non-uniform) | -61.9 (uniform bias, softmax-safe) |
| Output quality | "perdida perdida..." | Coherent English |

### Fix 2: Float32 Norms (precision)

Changed norm storage from float16 to float32. With key norms of 400-1000+,
float16 loses precision needed for the corrected norm ratio.

### Fix 3: Drop QJL (already validated)

Every working implementation uses ALL bits for Lloyd-Max MSE, not (b-1) MSE
+ 1-bit QJL. QJL removes bias but explodes variance — softmax tolerates
uniform bias but not variance. Our earlier QJL testing confirmed this
(error range 121.9 -> 1173.7 with QJL).

### What We Investigated That Didn't Help

| Approach | Result | Why |
|----------|--------|-----|
| QJL (Stage 2) | 10x worse | Variance explosion overwhelms bias correction |
| Pre-RoPE interception | 30% better | k_proj creates heterogeneous magnitudes before RoPE |
| Calibrated codebooks | No change | Same structural problem, not a codebook issue |
| Attention sinks | Mixed | Helps first tokens but FP16/compressed boundary hurts |

## Remaining Quality Gap

Even with norm correction, output degrades over long sequences (~30+ tokens).
The 70/128 "bad elements" remain — per-element error is still proportional
to vector norm, not element magnitude. But the uniform bias is now harmless
to softmax, and the error RANGE (4.9) is below the signal range (38).

Further improvements are documented in `QUALITY_IMPROVEMENT_ROADMAP.md`.
