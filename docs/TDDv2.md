# TDD — MLX TurboQuant-Inspired KV Cache Compression

**Status:** Rewritten technical design draft  
**Date:** 2026-03-28  
**Target stack:** Python 3.11+, MLX, MLX-LM, Apple Silicon

---

## 1. Scope note

This TDD is intentionally narrower and more accurate than the original.

It defines a technical design for a **stage-1-only KV-cache compression prototype for MLX/MLX-LM**. It does **not** claim to be a complete implementation of Google’s current public two-stage TurboQuant framing.

Where this TDD says “stage 1,” read that as:

- a practical first compressor for MLX,
- suitable for Apple-Silicon benchmarking,
- leaving full PolarQuant / QJL parity to a later branch.

---

## 2. Design goals

1. Build a correct reference codec first.
2. Integrate with real MLX-LM generation workflows.
3. Make memory savings measurable.
4. Keep the implementation simple enough to debug.
5. Provide a clean migration path toward optimized Metal kernels.

---

## 3. Non-goals

- full-faithful implementation of Google’s full March 2026 system,
- training-time KV compression,
- 1-bit / ternary weight quantization,
- QAT,
- broad production hardening across every MLX-LM model family in v1.

---

## 4. High-level architecture

```text
User / Bench CLI
        |
        v
 Generation wrapper
        |
        +-----------------------------+
        |                             |
        v                             v
 Baseline MLX-LM cache        Compressed cache adapter
                                      |
                                      v
                           Stage-1 KV codec (reference)
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
              Pure MLX ops path                Optional Metal path
```

---

## 5. Package layout

```text
mlx_turboquant/
├── __init__.py
├── cache/
│   ├── __init__.py
│   ├── compressed_cache.py
│   ├── cache_layout.py
│   └── memory_accounting.py
├── codec/
│   ├── __init__.py
│   ├── stage1_codec.py
│   ├── packbits.py
│   ├── codebooks.py
│   └── transforms.py
├── integration/
│   ├── __init__.py
│   ├── mlx_lm_adapter.py
│   └── generate_wrapper.py
├── kernels/
│   ├── __init__.py
│   ├── metal_pack.py
│   └── metal_attention.py
├── bench/
│   ├── __init__.py
│   ├── latency.py
│   ├── quality.py
│   ├── memory.py
│   └── prompts.py
├── tests/
│   ├── test_packbits.py
│   ├── test_codebooks.py
│   ├── test_codec_roundtrip.py
│   ├── test_cache_accounting.py
│   ├── test_generate_integration.py
│   └── test_perf_smoke.py
└── cli.py
```

---

## 6. Integration strategy

## 6.1 Baseline assumption

MLX-LM already supports generation, streaming generation, prompt caching, and a rotating fixed-size KV cache. This project adds a **parallel cache implementation**, not a replacement of all MLX-LM internals.

## 6.2 Integration principle

The initial design should avoid invasive changes. The adapter layer should:

1. intercept writes to the KV cache,
2. compress stored vectors,
3. reconstruct or consume compressed values during attention,
4. allow easy fallback to the baseline cache.

## 6.3 Compatibility stance

The first working target should be a **small number of known-good model families**, not every model that MLX-LM can load.

---

## 7. Data model

## 7.1 Core cache structure

A compressed cache instance stores the following per layer and per K/V branch:

- packed low-bit coordinate data,
- optional norm or auxiliary scalar metadata,
- sequence length,
- codec configuration,
- shape metadata needed for reconstruction.

### Suggested Python structure

```python
from dataclasses import dataclass

@dataclass
class CacheConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    max_seq_len: int
    kv_bits: int
    backend: str  # "reference" | "metal"

@dataclass
class QuantizedTensorStore:
    packed: object
    aux: object
    seq_len: int
```

## 7.2 Memory accounting formula

### Baseline FP16 KV cache bytes per token

```text
bytes_per_token_fp16 =
    2 (K and V)
  * num_layers
  * num_heads
  * head_dim
  * 2 bytes
```

### Compressed-cache bytes per token

For the early prototype:

```text
bytes_per_token_compressed ≈
    2
  * num_layers
  * num_heads
  * (ceil(head_dim * kv_bits / 8) + aux_bytes)
```

Where `aux_bytes` depends on what extra metadata the codec stores.

The benchmark harness must report both the formula estimate and the observed storage usage.

### Important reporting rule

The project should track two different memory numbers:

- **logical occupied bytes**: bytes implied by the current token count and storage formula,
- **allocated bytes**: bytes reserved in backing buffers because of cache growth granularity.

User-facing compare ratios and benchmark headlines should use logical occupied bytes. Allocated bytes are still useful, but only as diagnostic context for short runs and allocator behavior.

---

## 8. Codec design

## 8.1 Important naming rule

The stage-1 codec should be described as a **project codec**, not as “the full TurboQuant algorithm.”

## 8.2 Stage-1 reference codec

The reference codec should follow a straightforward low-bit vector-compression path suitable for MLX:

1. optionally compute and store a norm or scale term,
2. apply a deterministic sign pattern if the chosen transform requires it,
3. apply a fast orthogonalizing transform surrogate suitable for MLX,
4. scalar-quantize transformed coordinates using precomputed tables,
5. pack indices into byte arrays.

### Read path

1. unpack indices,
2. dequantize through the lookup table,
3. apply the inverse transform,
4. restore any stored norm / scale term,
5. produce the vector representation needed by attention.

### Why this design

This is the simplest useful path for MLX because it allows:

- debugging in Python,
- unit testing of every stage,
- later substitution of optimized kernels.

---

## 9. Codebook handling

The project should treat codebooks as versioned assets derived offline.

### Requirements

- deterministic generation,
- saved metadata describing dimension and bit-width,
- easy loading at runtime,
- unit tests verifying monotonicity and shape.

### Initial recommendation

Support the minimum set of dimensions required by the first target models and expand only after integration is stable.

---

## 10. Pure-MLX reference backend

The first backend should be implemented entirely with MLX array ops where practical.

### Candidate operations

- reshape-based transform stages,
- lookup-table decode,
- bit packing / unpacking through integer ops,
- explicit reconstruction for correctness mode.

### Why it matters

A pure-MLX path provides:

- a correctness oracle for later Metal kernels,
- easier debugging,
- deterministic unit testing.

---

## 11. Metal optimization plan

MLX exposes custom kernels through `mlx.core.fast.metal_kernel`, so the design should reserve a clean backend boundary.

## 11.1 Kernel candidates

### Kernel A — pack / unpack hot path

Use a Metal kernel for:

- packing quantized coordinates,
- unpacking packed cache blocks,
- possibly fusing lookup + unpack for decode.

### Kernel B — fused attention-logit path

A later optimization can attempt to avoid fully materializing reconstructed K/V tensors by fusing:

- unpack,
- lookup,
- partial reconstruction,
- dot-product work.

This is the highest-risk and highest-value optimization.

## 11.2 Development order

1. Reference MLX path first.
2. Packing / unpacking kernel second.
3. Fused attention kernel last.

---

## 12. MLX-LM adapter design

### Suggested interface

```python
class CacheAdapter:
    def create_cache(self, model_config, kv_bits: int, backend: str):
        ...

    def prefill_write(self, layer_idx, key, value, positions):
        ...

    def read_for_attention(self, layer_idx, positions):
        ...
```

### Generate wrapper

```python
def generate_with_compressed_cache(model, tokenizer, prompt, *, kv_bits=3, backend="reference"):
    ...
```

### Design rule

The wrapper must make it trivial to compare baseline and compressed cache on the same prompt and sampler settings.

The wrapper should also expose both logical occupied cache bytes and allocated cache bytes so benchmark/report code does not confuse short-run allocation effects with true compression ratio.

---

## 13. Execution flow

## 13.1 Prefill

1. Load model and tokenizer.
2. Tokenize prompt.
3. During prefill, route K/V writes to the compressed cache adapter.
4. Record memory and latency metrics.

## 13.2 Decode step

1. For each generated token, read the cache through the selected backend.
2. Compute attention using either:
   - reconstructed tensors from the reference path, or
   - a later fused kernel path.
3. Continue generation while recording metrics.

---

## 14. Benchmark design

## 14.1 Benchmark classes

### A. Functional smoke

- tiny model,
- short prompt,
- baseline vs compressed output sanity.

### B. Practical chat benchmark

- one instruct model in the 7B / 8B class,
- typical user prompt,
- prefill + decode timing,
- peak memory comparison.

### C. Long-context benchmark

- long prompt,
- baseline cache vs rotating cache vs compressed cache,
- memory and quality comparison.

## 14.2 Metrics

Required metrics:

- peak memory,
- logical bytes per token of cache,
- allocated cache bytes as a diagnostic metric,
- prefill seconds,
- decode tok/s,
- output drift versus baseline,
- any task-specific accuracy metric used by the project.

---

## 15. Testing strategy

## 15.1 Unit tests

- pack / unpack round trips,
- codebook integrity,
- transform inverse consistency,
- codec round-trip error bounds,
- memory accounting correctness.

## 15.2 Integration tests

- one baseline generation test,
- one compressed-cache generation test,
- identical prompt + seed comparisons,
- fallback path test.

## 15.3 Performance smoke tests

- reference backend benchmark,
- Metal backend benchmark when enabled,
- regression thresholds captured in CI where practical.

---

## 16. Failure modes to design for

1. **Shape mismatch** across model families.
2. **Non-contiguous tensor assumptions** when using Metal kernels.
3. **Quality collapse** at aggressive bit-widths.
4. **Throughput regression** where memory savings are outweighed by decode cost.
5. **Silent correctness drift** between the reference backend and optimized backend.

Every one of these should have at least one explicit test or runtime assertion.

---

## 17. Optional fine-tuning integration

MLX-LM already supports LoRA / QLoRA / fuse workflows. This project should treat that as a **downstream integration point**, not a first-class v1 responsibility.

Recommended rule:

- fine-tune or fuse with MLX-LM first,
- then run inference through the compressed cache path.

Do not make 16 GB QLoRA guarantees in the TDD.

---

## 18. Phased implementation plan

## Phase 1 — correctness

- package skeleton,
- codebooks,
- pack / unpack,
- reference codec,
- baseline benchmark harness.

## Phase 2 — integration

- MLX-LM wrapper,
- prompt and generation benchmarks,
- model compatibility for first target family.

## Phase 3 — optimization

- Metal pack / unpack,
- improved decode path,
- memory / speed tuning.

## Phase 4 — research extension

- evaluate closer PolarQuant-oriented implementation,
- evaluate QJL-style residual branch,
- decide whether full parity is worth pursuing.

---

## 19. Technical decisions locked by this rewrite

1. v1 is **stage-1-only**.
2. v1 must be labeled **TurboQuant-inspired**, not full parity.
3. pure-MLX reference path comes before Metal optimization.
4. baseline comparison is mandatory.
5. weight-quantization research stays out of the critical path.

---

## 20. References

1. Google Research Blog — *TurboQuant: Redefining AI efficiency with extreme compression*  
   https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

2. arXiv — *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*  
   https://arxiv.org/abs/2504.19874

3. MLX docs — `mlx.core.fast.metal_kernel`  
   https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html

4. MLX docs — Custom Metal Kernels  
   https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html

5. MLX-LM README  
   https://github.com/ml-explore/mlx-lm

6. MLX-LM LoRA / QLoRA docs  
   https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
