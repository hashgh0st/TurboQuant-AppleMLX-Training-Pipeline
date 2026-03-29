# PRD — MLX TurboQuant-Inspired KV Cache Compression

**Status:** Rewritten research-grounded draft  
**Date:** 2026-03-28  
**Working project name:** `mlx-turboquant`  
**Primary environment:** Apple Silicon + MLX + MLX-LM

---

## 1. Executive summary

This project aims to build an Apple-Silicon-first KV-cache compression layer for MLX/MLX-LM so larger models can sustain longer context windows on machines with limited unified memory.

This document intentionally corrects the framing from the original plans:

- **TurboQuant is not a weight quantizer.** It is a vector / KV-cache compression method.
- **Google’s current public framing is two-stage.** The March 24, 2026 Google Research blog presents TurboQuant as a combination of **PolarQuant** for the main compression stage plus a **1-bit QJL residual** stage.
- **This PRD scopes v1 more narrowly.** The first shipping milestone is a **stage-1-only MLX KV-cache compressor prototype**. It should be described as **TurboQuant-inspired** or **TurboQuant-stage-1-oriented**, not as a claim of full parity with Google’s complete system.

The product goal is practical: integrate low-bit KV-cache compression into MLX inference workflows without rewriting the entire MLX stack or promising unsupported hardware outcomes.

---

## 2. Source-of-truth framing

### 2.1 What the official sources say

Google’s March 24, 2026 blog introduces TurboQuant as a compression method for KV cache and vector search, describing it as a two-step system built from **PolarQuant** and a **1-bit QJL residual** stage. The same post reports benchmarked KV-cache compression to **3 bits**, at least **6x** KV-memory reduction, and up to **8x** attention-logit speedup on H100 in Google’s measurements.

The arXiv paper was first submitted on **April 28, 2025**. Its abstract describes TurboQuant as a **two-stage** method and reports **absolute quality neutrality at 3.5 bits per channel** and **marginal quality degradation at 2.5 bits per channel**.

MLX currently exposes custom Metal kernels through `mlx.core.fast.metal_kernel`, and MLX-LM already supports generation, streaming generation, prompt caching, a rotating fixed-size KV cache, and LoRA / QLoRA / fuse workflows.

### 2.2 What this PRD will not claim

This document will **not** claim the following as established facts unless validated by project benchmarks:

- that a stage-1-only implementation is equivalent to full TurboQuant,
- that 13B / 70B / 405B models will fit specific Apple Silicon memory tiers,
- that 7B QLoRA on 16 GB is guaranteed,
- that Apple-Silicon performance will match Google’s H100 speedup figures.

---

## 3. Problem statement

MLX-LM provides mechanisms to manage long prompts, including a rotating KV cache and prompt caching, but it does **not** currently provide a general-purpose low-bit KV-cache compression layer comparable to the newly public TurboQuant direction. That leaves long-context inference on Apple Silicon heavily memory-bound.

The project problem is therefore:

> Build a practical MLX-compatible KV-cache compression system that materially reduces KV memory usage on Apple Silicon, integrates into real generation workflows, and leaves a clear path toward more faithful research reproduction later.

---

## 4. Product goals

### 4.1 Primary goals

1. **Reduce KV-cache memory usage** during MLX inference.
2. **Increase feasible context length** on Apple Silicon at a fixed memory budget.
3. **Preserve output quality well enough** for practical chat, summarization, and retrieval-style workloads.
4. **Integrate cleanly with MLX-LM** rather than requiring a bespoke runtime.
5. **Provide a measured benchmark harness** so claims are backed by local evidence.

### 4.2 Secondary goals

1. Expose a clean Python API and CLI.
2. Support multiple Apple Silicon tiers, from an M4 Mini 16 GB up to higher-memory Ultra-class machines.
3. Leave room for future research tracks, including PolarQuant- and QJL-oriented follow-on work.

---

## 5. Non-goals for v1

The following are explicitly out of scope for the first release:

- full-faithful reproduction of Google’s complete March 2026 TurboQuant stack,
- broad model-family support across every MLX-LM model on day one,
- training-time compression,
- quantization-aware training,
- 1-bit or ternary weight-quantization work,
- claims of exact parity with Google benchmark numbers,
- shipping a production-optimized Metal kernel before the reference MLX path works.

---

## 6. Target users

### Primary user

A power user / developer running MLX models locally on Apple Silicon who wants materially longer context or lower memory pressure.

### Secondary users

- MLX community contributors,
- local-inference hobbyists,
- researchers exploring KV-cache compression on Apple hardware,
- developers who already use MLX-LM and want a more memory-efficient cache path.

---

## 7. Product scope by phase

## Phase 0 — Research-aligned rewrite and benchmark plan

Deliverables:

- corrected PRD / research brief / TDD,
- source-of-truth references,
- benchmark plan,
- explicit separation between official claims and project estimates.

## Phase 1 — Stage-1-only reference implementation

Deliverables:

- a pure-MLX reference codec for KV vectors,
- configurable low-bit cache storage,
- integration with at least one MLX-LM generation path,
- benchmark harness for memory, latency, and quality,
- fallback to normal MLX-LM cache for comparison.

This phase is the minimum viable product.

## Phase 2 — Apple Silicon optimization pass

Deliverables:

- custom Metal kernels through MLX for hot-path operations,
- lower overhead packing / unpacking,
- faster attention-logit path,
- model compatibility expansion.

## Phase 3 — Research extension branch

Deliverables:

- exploration of a more faithful PolarQuant-oriented main stage,
- evaluation of QJL-style residual correction,
- comparison against the stage-1-only baseline,
- decision on whether full-parity reproduction is realistic in MLX.

---

## 8. Functional requirements

### 8.1 Core inference requirements

1. The system must support **drop-in or near-drop-in generation** for at least one MLX-LM compatible model.
2. The user must be able to choose between:
   - baseline cache,
   - compressed cache reference path,
   - optimized compressed cache path when available.
3. The system must report:
   - total cache bytes,
   - bytes per token,
   - peak process memory observed during a run,
   - selected compression settings.
4. The system must expose an **easy benchmark mode** comparing baseline vs compressed cache on the same prompt.

### 8.2 Configuration requirements

The system should expose, at minimum:

- bit-width selection,
- max context / cache length,
- reference vs optimized backend,
- quality / benchmark mode,
- model and prompt selection.

### 8.3 Benchmark requirements

The benchmark harness must capture:

- prefill latency,
- decode latency or tokens/sec,
- memory usage,
- output-quality deltas relative to baseline.

---

## 9. Quality and acceptance criteria

These are **engineering acceptance criteria**, not claims about the paper.

### P0 acceptance criteria

1. **Reference path works end-to-end** on Apple Silicon for at least one small debug model and one practical instruct model.
2. **Cache memory accounting is correct** within a small tolerance versus the implemented storage formula, and clearly distinguishes logical occupied bytes from allocated backing-buffer bytes.
3. **Benchmark mode is reproducible** on the same machine and model.
4. **Quality regressions are visible and measurable**, not hidden.
5. **Users can switch back to baseline cache** without changing the rest of their workflow.

### P1 acceptance criteria

1. Optimized path outperforms the pure-MLX reference path on the same hardware.
2. No silent correctness failures relative to the reference codec.
3. The optimized path is optional and feature-gated.

### Suggested internal targets

These are project targets only:

- meaningful KV-memory reduction at 3-bit and 4-bit operating points,
- materially longer context than the baseline cache on the same machine,
- limited quality degradation on the project’s selected eval set.

---

## 10. Hardware strategy

The hardware plan should be described as **tiers**, not promises.

| Tier | Purpose | Notes |
|---|---|---|
| M4 Mini 16 GB | Functional baseline | First target for correctness and usability |
| M4 Pro / 48–64 GB class | Performance tuning | Better headroom for larger models and longer contexts |
| M4 Max / Ultra class | Research exploration | Large-model experiments and kernel tuning |

### Important rule

Any model-fit tables in the codebase or docs must be labeled **estimate** unless verified by local benchmarks.

---

## 11. User experience

### 11.1 CLI

Suggested CLI surface:

```bash
mlx-tq generate --model <model> --prompt "..." --cache-mode baseline
mlx-tq generate --model <model> --prompt "..." --cache-mode stage1 --kv-bits 3
mlx-tq bench --model <model> --prompt-file prompt.txt --kv-bits 3
mlx-tq info --model <model>
```

Current product posture:

- baseline generation should remain the safe default,
- compressed generation should be an explicit experimental opt-in until local benchmarks support a stronger claim,
- compare and benchmark views should headline logical occupied cache bytes and may show allocated bytes only as diagnostic context.

### 11.2 Python API

Suggested API surface:

```python
from mlx_turboquant import load_compressed_cache, generate, benchmark
```

The Python API matters more than the CLI in the early phases.

---

## 12. Risks

### Technical risks

1. **Integration risk:** MLX-LM cache internals may differ across model families.
2. **Performance risk:** a correct reference codec may still be too slow without Metal optimization.
3. **Research risk:** a stage-1-only implementation may deliver weaker quality than hoped on real long-context tasks.
4. **Parity risk:** Google’s current public framing emphasizes PolarQuant + QJL, which may not map directly to the simplest MLX prototype.
5. **Maintenance risk:** MLX and MLX-LM evolve quickly.

### Product risks

1. Overclaiming equivalence to TurboQuant.
2. Publishing unsupported hardware-fit claims.
3. Mixing weight quantization work into the initial scope and diluting the main goal.

---

## 13. Dependencies

- MLX
- MLX-LM
- Apple Silicon hardware
- benchmark prompts and evaluation harness
- optional custom Metal-kernel work through `mlx.core.fast.metal_kernel`

---

## 14. Milestones

### M1 — Correct framing and design lock

- docs rewritten,
- benchmark plan defined,
- scope frozen for phase 1.

### M2 — Reference codec

- pure-MLX codec implemented,
- bit-packing and dequantization working,
- memory accounting validated.

### M3 — MLX-LM integration

- generation path working with compressed cache,
- baseline vs compressed comparisons available.

### M4 — Optimized backend

- custom Metal path for one or more hot operations,
- perf gains validated against reference.

### M5 — Research extension

- investigate PolarQuant-oriented and QJL-oriented paths,
- decide whether “full TurboQuant parity” remains a realistic roadmap item.

---

## 15. Naming guidance

Until the project implements more than the stage-1 prototype, the safest public description is:

> **Apple Silicon KV-cache compression for MLX, inspired by the recent TurboQuant research direction.**

Avoid calling the early version a full TurboQuant port.

---

## 16. References

Primary sources used for this rewrite:

1. Google Research Blog — *TurboQuant: Redefining AI efficiency with extreme compression*  
   https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

2. arXiv — *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*  
   https://arxiv.org/abs/2504.19874

3. MLX docs — `mlx.core.fast.metal_kernel`  
   https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html

4. MLX docs — Custom Metal Kernels  
   https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html

5. MLX-LM repository README  
   https://github.com/ml-explore/mlx-lm

6. MLX-LM LoRA / QLoRA docs  
   https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
