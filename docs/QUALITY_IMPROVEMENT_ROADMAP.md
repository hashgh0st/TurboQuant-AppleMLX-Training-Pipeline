# Quality Improvement Roadmap

After applying norm correction (25x attention logit improvement), the rotation
codec produces coherent 4-bit output on Qwen2.5-7B. Quality degrades over long
sequences. This document outlines remaining improvements ranked by expected impact.

## Current State (post norm correction)

| Metric | 4-bit | Notes |
|--------|-------|-------|
| Compression | 3.8x | vs FP16 baseline |
| First ~30 tokens | Coherent | On-topic, grammatically correct |
| 30+ tokens | Degrades | Repetition, drift, occasional garbage |
| Attention logit error range | 4.9 | Below signal range of 38 |
| Attention logit bias | -61.9 | Uniform, softmax-safe |
| Bad elements (|error|>|original|) | 70/128 | Structural, from rotation |

## Improvements Ranked by Expected Impact

### 1. Higher Bit-Width (Immediate)

**Expected impact: HIGH**

The llama.cpp community converged on TQ3 (3.25 effective bits) as the quality
sweet spot. Our 4-bit (4.0 effective bits) should be better. Consider:

- **TQ4.25**: Use 4-bit codebook + 0.25 extra bits for norm precision. This is
  what working implementations use — slightly more than 4 bits per coordinate
  for the quantized values, plus float32 norm overhead.
- **5-bit**: 32 levels. Would reduce per-coordinate distortion from 0.000058 to
  ~0.000015 (4x improvement). Not standard but could be generated.

### 2. Residual Quantization / Multi-Stage MSE (Medium-term)

**Expected impact: HIGH**

Instead of QJL (which adds variance), use a second round of MSE quantization on
the residual. This is standard in product quantization:

```
Stage 1: 3-bit MSE quantize -> residual r = x - reconstruct
Stage 2: 2-bit MSE quantize the residual r
Total: 5 bits, but distortion of 3+2 split < 5-bit single-stage
```

Unlike QJL, residual MSE doesn't add variance — it strictly reduces error.
The key insight is that residual vectors have a DIFFERENT distribution from
the original vectors, so they need their own codebook.

### 3. Group-Size Tuning for Norm Storage (Immediate)

**Expected impact: MEDIUM**

Currently we store one float32 norm per vector (head_dim elements). If we
group vectors and share norms across a group, the per-element overhead drops.
Conversely, storing per-CHANNEL norms (one per dimension) would give each
channel its own scale — approaching per-channel quantization quality within
the rotation framework.

Consider: after rotation, compute norms per group of 32 coordinates instead
of per vector. This captures coordinate-level variance that the single
per-vector norm misses.

### 4. Outlier-Aware Rotation (from RotateKV, IJCAI 2025)

**Expected impact: MEDIUM-HIGH**

RotateKV achieves <0.3 PPL degradation at 2-bit by:
1. Channel-reordering before Hadamard (groups similar-magnitude channels)
2. Grouped-head rotation (rotates across KV heads, not just within)
3. Attention-sink-aware quantization

Our Hadamard rotation is "blind" — it doesn't know which channels have
outliers. RotateKV's channel reordering would cluster similar-magnitude
channels so that rotation equalizes them more effectively.

### 5. Longer Codebook Training (Quick Win)

**Expected impact: LOW-MEDIUM**

Our Lloyd-Max codebooks use 300 iterations on the Beta(d/2,d/2) PDF. The
convergence check shows they stabilize by ~30 iterations. But the grid_size
(10,000 points) may limit precision. Increasing to 100,000 points could
tighten the centroid placement for marginal gains.

Also: non-uniform codebooks (more levels near the distribution peak, fewer
in the tails) can be explored via entropy-constrained optimization.

### 6. Qwen3.5 Model Testing (Validation)

**Expected impact: VALIDATION**

Multiple implementations report that Qwen2.5 is a particularly difficult
model for TurboQuant (helgklaizar explicitly lists it as failing). Qwen3.5
is reported to work well across implementations. Testing on Qwen3.5 would
separate model-specific issues from algorithmic limitations.

### 7. Per-Channel Quantization (Separate Project)

**Expected impact: VERY HIGH but different approach**

Per-channel group quantization (KIVI-style) is fundamentally more suited to
KV cache compression because it preserves per-element scale. This is being
developed as a separate project — see `docs/KIVI_PIVOT_BRIEF.md`.

The rotation approach can coexist: use rotation for models where it works
(e.g., Qwen3.5) and per-channel for models where it doesn't (e.g., Qwen2.5).

## Implementation Priority

1. **Test on Qwen3.5** — validate that norm correction + 4-bit works on a known-good model
2. **Residual MSE (multi-stage)** — reduce distortion without QJL's variance penalty
3. **Outlier-aware rotation** — adopt RotateKV's channel reordering
4. **Profile sweep** — systematic comparison of 3/4/5-bit across Qwen 2.5, 3, 3.5

## References

- llama.cpp TurboQuant discussion: https://github.com/ggml-org/llama.cpp/discussions/20969
- ik_llama.cpp implementation: https://github.com/ikawrakow/ik_llama.cpp/issues/1509
- RotateKV: https://arxiv.org/abs/2501.16383
- KIVI: https://arxiv.org/abs/2402.02750
- KVQuant: https://arxiv.org/abs/2401.18079
