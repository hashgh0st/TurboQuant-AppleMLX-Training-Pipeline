# Research Brief — TurboQuant, MLX, and what is actually supported today

**Status:** Rewritten source-of-truth brief  
**Date:** 2026-03-28

---

## 1. Purpose

This brief replaces the original research notes with a version that is aligned to the official public sources and explicit about what is known, what is inferred, and what remains an implementation choice.

---

## 2. Executive summary

The key correction is simple:

> **TurboQuant should currently be treated as a two-stage compression direction, not as “Algorithm 1 only.”**

Google’s March 24, 2026 blog presents TurboQuant as the combination of:

1. **PolarQuant** for the main compression stage, and
2. **1-bit QJL** on the residual to remove inner-product bias.

The arXiv paper, first submitted on April 28, 2025, also describes TurboQuant as a **two-stage** approach and reports **absolute quality neutrality at 3.5 bits per channel** with **marginal degradation at 2.5 bits per channel**.

That means the earlier project docs were directionally interesting but too aggressive in three places:

- they treated a stage-1-only design as if it were full TurboQuant,
- they promoted hardware-fit estimates into near-facts,
- they mixed KV-cache compression and low-bit weight-quantization work too tightly.

The correct practical move is to split the work:

- **build a stage-1-only MLX prototype first**,
- **benchmark it honestly as a TurboQuant-inspired cache compressor**,
- **keep full PolarQuant / QJL parity as a later research branch**.

---

## 3. What the official sources say

## 3.1 Google Research blog

The Google Research post dated **March 24, 2026** states that TurboQuant is meant for **KV-cache compression** and **vector search**. It says TurboQuant uses:

- a **main compression stage** built from **PolarQuant**, and
- a **small 1-bit QJL residual** to eliminate bias in attention-score estimation.

The same blog reports:

- KV-cache compression down to **3 bits** in the benchmarked setting,
- at least **6x** KV-memory reduction,
- **no observed compromise in model accuracy** in the reported tests,
- up to **8x** attention-logit speedup on H100 relative to an unquantized-key baseline in Google’s setup.

## 3.2 arXiv paper

The arXiv paper was **submitted on April 28, 2025**. Its abstract states that TurboQuant:

- targets both **MSE distortion** and **inner-product distortion**,
- uses a **two-stage** design,
- applies an **MSE quantizer** first,
- then applies a **1-bit QJL transform on the residual**,
- achieves **quality neutrality at 3.5 bits per channel**,
- shows **marginal degradation at 2.5 bits per channel**.

## 3.3 MLX / MLX-LM baseline capabilities

From the current MLX and MLX-LM documentation:

- MLX supports **custom Metal kernels** via `mlx.core.fast.metal_kernel`.
- MLX-LM already supports:
  - `generate` and `stream_generate`,
  - prompt caching,
  - a rotating fixed-size KV cache,
  - LoRA / QLoRA / fuse workflows.

This makes MLX a plausible place to build a KV-cache-compression prototype even though MLX-LM does not currently expose a first-class TurboQuant-style cache compressor.

---

## 4. What the original project plans got right

Several ideas in the original docs were good and should be preserved:

1. **The center of gravity should be the KV cache, not the model weights.**
2. **MLX + MLX-LM is the right Apple stack** for a local prototype.
3. **A reference implementation should come before heavy kernel optimization.**
4. **Custom Metal kernels are likely needed** for the best Apple-Silicon performance.
5. **The benchmark harness is as important as the codec itself.**

---

## 5. What needed correction

## 5.1 “Algorithm 1 only” was overstated

The biggest correction is scope labeling.

A stage-1-only compressor may still be the right first engineering move, but it should be labeled as:

- a **prototype**,
- a **baseline**,
- or a **TurboQuant-inspired stage-1 implementation**.

It should **not** be presented as identical to the full Google framing.

## 5.2 Some quality claims were too broad

The original notes leaned too hard on simplified language like “3-bit zero-loss.” The official public record is narrower:

- the **blog** highlights a 3-bit benchmarked result,
- the **paper abstract** highlights **3.5 bits per channel** as quality-neutral.

Those statements are related, but they are not interchangeable.

## 5.3 Hardware-fit claims were treated as facts

Claims like:

- “13B on 16 GB,”
- “70B on M4 Pro,”
- “405B on M4 Max,”
- or hard guarantees around QLoRA on 16 GB,

should be treated as **planning estimates only** until they are demonstrated locally.

## 5.4 Weight quantization was too entangled with KV-cache work

The earlier docs bundled three different projects together:

1. KV-cache compression,
2. LoRA / QLoRA fine-tuning integration,
3. 1-bit / ternary weight quantization.

Those are related, but they are not the same project. The lowest-risk path is to finish KV-cache compression first.

---

## 6. Practical interpretation for Apple Silicon

## 6.1 Why MLX is still a strong target

MLX is attractive because:

- it is native to Apple Silicon,
- it already has a local LLM ecosystem through MLX-LM,
- it exposes a path to custom kernels,
- it already has long-prompt and fine-tuning utilities.

## 6.2 What is realistically buildable first

The best first deliverable is not “full TurboQuant on MLX.”

The best first deliverable is:

> **a stage-1-only MLX KV-cache compression prototype with honest benchmarks**.

That can validate whether Apple Silicon benefits are large enough to justify deeper parity work.

## 6.3 What should be benchmarked against

The project should compare against:

- baseline MLX-LM cache,
- MLX-LM rotating KV cache for memory-constrained runs,
- the project’s compressed cache reference path,
- later, the project’s optimized Metal path.

---

## 7. Recommended project framing

Use this wording in the repo and docs:

> `mlx-turboquant` is an Apple-Silicon KV-cache compression project for MLX/MLX-LM inspired by the recent TurboQuant research direction. v1 focuses on a practical stage-1-only prototype and benchmark harness rather than claiming complete parity with Google’s full two-stage system.

This wording is accurate, useful, and avoids overclaiming.

For current UX and reporting, the framing should also stay conservative:

- baseline generation is the default path,
- compressed generation is an experimental opt-in,
- benchmark headlines should use logical occupied cache bytes, not short-run backing-buffer allocation artifacts.

---

## 8. Recommended build order

### Step 1 — lock the scope

Freeze v1 as:

- reference codec,
- MLX-LM integration,
- benchmark harness,
- clear memory accounting.

### Step 2 — validate memory wins first

Before chasing fused attention kernels, verify that the cache actually produces worthwhile memory savings on Apple Silicon.

### Step 3 — then optimize

Once the reference path is correct, move hot-path work to MLX custom Metal kernels.

### Step 4 — revisit parity

Only after a working benchmarked prototype exists should the project decide whether to chase:

- PolarQuant-oriented main-stage parity,
- QJL residual parity,
- or a different Apple-friendly approximation.

---

## 9. Open questions

These are the main research questions still open for the project:

1. Can a stage-1-only MLX implementation preserve enough quality to justify the project on Apple Silicon?
2. Which MLX-LM integration point is the least fragile across model families?
3. How much of the runtime cost can be removed with `metal_kernel` before maintenance complexity becomes too high?
4. Is a faithful PolarQuant main stage more attractive than a simpler rotated-coordinate baseline for the first optimized path?
5. Is a QJL-style residual path practical in MLX without fully custom attention plumbing?

---

## 10. Decision summary

### Decisions that should be considered locked

- Treat the March 2026 Google blog and the April 2025 arXiv paper as the main source of truth.
- Treat TurboQuant as a **two-stage** research direction.
- Label v1 as **TurboQuant-inspired**, not full parity.
- Keep low-bit weight work out of the critical path.
- Build a benchmark harness before making hardware claims.

### Decisions intentionally left open

- exact stage-1 codec details,
- exact integration hook inside MLX-LM,
- whether to pursue full PolarQuant / QJL parity later.

---

## 11. References

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
