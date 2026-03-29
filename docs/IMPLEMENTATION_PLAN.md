# Implementation Plan — mlx-turboquant

## Context

This project builds an Apple-Silicon-first KV-cache compression library for MLX/MLX-LM, inspired by Google's TurboQuant research (March 2026 blog + April 2025 arXiv paper). The goal is a practical, well-engineered prototype that:

- Materially reduces KV-cache memory on M4 Mini 16 GB
- Integrates cleanly with MLX-LM's Qwen 2.5/3 model family
- Provides honest benchmarks comparing baseline vs compressed cache
- Demonstrates technical excellence as a portfolio piece

**Problem**: MLX-LM has no general-purpose low-bit KV-cache compression beyond simple affine quantization. Long-context inference on Apple Silicon is memory-bound. A 7B model's KV cache at 4K context consumes ~512 MB in fp16 — a significant fraction of 16 GB unified memory.

**Solution**: A stage-1-only TurboQuant-inspired codec that uses random Hadamard rotation + Lloyd-Max optimal codebooks to compress KV vectors to 2-4 bits per coordinate, achieving near-Shannon-limit distortion rates with data-oblivious codebooks.

### Critical Architectural Discoveries

1. **`mx.hadamard_transform(a, scale=None)` exists natively** in MLX for dimensions `m * 2^k` where m in {1, 12, 20, 28}. Both Qwen head_dim values (64=2^6, 128=2^7) satisfy this. No custom WHT implementation needed.

2. **`CompressedKVCache` must NOT expose a `bits` attribute.** MLX-LM's `scaled_dot_product_attention` in `models/base.py` dispatches to `quantized_scaled_dot_product_attention` (which uses `mx.quantized_matmul` with affine group quant) when `hasattr(cache, "bits")` is true. Our rotation + Lloyd-Max approach is incompatible with this path. Our cache must return decompressed fp16 tensors so the standard `mx.fast.scaled_dot_product_attention` path is used.

3. **The "query-rotation trick"** for Phase 5: instead of inverse-Hadamard-transforming every key vector at attention time (O(seq_len * head_dim)), forward-Hadamard the query once (O(head_dim)), then dot-product in the rotated domain. This turns per-token work into per-query work.

4. **MLX lacks `searchsorted`** — quantization must use broadcast comparison against codebook boundaries, which is efficient for small codebooks (8-16 levels).

### User Decisions

- **Hardware**: M4 Mini 16 GB (tight memory — compression benefits obvious)
- **Target model**: Qwen 2.5/3 family (GQA architecture, 0.5B for testing, 3B/7B for benchmarks)
- **Tooling**: uv + pyproject.toml
- **Polish**: Full open-source ready (CI, pre-commit, ruff/mypy, pytest, README, LICENSE)

---

## Phase 0 — Project Foundation [COMPLETE]

**Completed:** 2026-03-28 | **Commit:** `165007f`

### Goal
Establish a professional, fully-tooled Python package skeleton that builds, imports, and passes CI from the first commit.

### Deliverables
- Package skeleton with all directories and typed `__init__.py` stubs
- `pyproject.toml` with uv, ruff, mypy, pytest configuration
- GitHub Actions CI workflow (lint + type-check + test on macOS)
- Pre-commit hooks (ruff format, ruff check)
- `README.md`, `LICENSE` (MIT), `CONTRIBUTING.md`
- `CLAUDE.md` with project conventions
- `.gitignore` for Python/MLX/macOS/Claude Code

### Files to Create

```
mlx-turboquant/                     # repo root (rename from appleturbo)
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions: lint + typecheck + test
├── .pre-commit-config.yaml
├── .gitignore
├── .python-version                 # 3.11
├── pyproject.toml                  # uv + ruff + mypy + pytest config
├── README.md
├── LICENSE                         # MIT
├── CONTRIBUTING.md
├── CLAUDE.md
├── mlx_turboquant/
│   ├── __init__.py                 # version, public API re-exports
│   ├── py.typed                    # PEP 561 marker
│   ├── codec/
│   │   ├── __init__.py
│   │   ├── stage1_codec.py         # stub
│   │   ├── packbits.py             # stub
│   │   ├── codebooks.py            # stub
│   │   └── transforms.py           # stub
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── compressed_cache.py     # stub
│   │   ├── cache_layout.py         # stub
│   │   └── memory_accounting.py    # stub
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── mlx_lm_adapter.py       # stub
│   │   └── generate_wrapper.py     # stub
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── metal_pack.py           # stub
│   │   └── metal_attention.py      # stub
│   ├── bench/
│   │   ├── __init__.py
│   │   ├── latency.py              # stub
│   │   ├── quality.py              # stub
│   │   ├── memory.py               # stub
│   │   └── prompts.py              # stub
│   └── cli.py                      # stub
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # shared fixtures
│   └── test_import.py              # smoke test: import mlx_turboquant
└── docs/
    ├── PRDv2.md                    # move existing docs
    ├── RESEARCHv2.md
    └── TDDv2.md
```

### Key Configuration

**pyproject.toml** dependencies:
```toml
[project]
requires-python = ">=3.11"
dependencies = [
    "mlx>=0.22.0",
    "mlx-lm>=0.22.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.8", "mypy>=1.13", "pre-commit>=4.0"]
bench = ["rich>=13.0", "tabulate>=0.9"]

[project.scripts]
mlx-tq = "mlx_turboquant.cli:main"
```

**ruff** config: line-length 100, target Python 3.11, select E/W/F/I/UP/B/SIM/RUF rules.
**mypy**: strict mode, disallow untyped defs.

### Tests
- `test_import.py`: `import mlx_turboquant` succeeds, version string exists

### Exit Criteria
- [x] `uv sync` installs all dependencies (54 packages including mlx 0.31.1, mlx-lm 0.31.1)
- [x] `import mlx_turboquant` works without error
- [x] `uv run pytest` passes with 1 test
- [x] `uv run ruff check .` reports 0 errors
- [x] `uv run mypy mlx_turboquant/` reports 0 errors (22 source files)
- [ ] GitHub Actions CI is green (not yet triggered — needs first push to main with workflow)
- [x] Pre-commit hooks configured (ruff format + ruff check)

### Post-Review Fixes Applied
- Version string single-sourced via `importlib.metadata` (was hardcoded in 3 places)
- Removed Python 3.13 classifier (no CI coverage, MLX wheel availability unconfirmed)
- CI changed from `--all-extras` to `--extra dev` (bench deps not needed for CI)
- Added `.claude/` to `.gitignore`

---

## Phase 1 — Core Codec [COMPLETE]

**Completed:** 2026-03-28

### Goal
Build the mathematical heart of the system: a correct, tested, pure-MLX reference codec that compresses and decompresses KV vectors using randomized Hadamard rotation + Lloyd-Max optimal scalar quantization.

### Deliverables
- Lloyd-Max codebook generator for Beta(d/2, d/2) distributions
- 6 precomputed codebook files: {d=64, d=128} x {bits=2, bits=3, bits=4}
- Randomized Hadamard transform using `mx.hadamard_transform` + Rademacher signs
- Per-coordinate scalar quantization via broadcast comparison
- Bit-packing/unpacking for 2, 3, and 4-bit values
- Full encode/decode pipeline with measured distortion bounds

### Files to Implement

#### `mlx_turboquant/codec/codebooks.py`

The codebook generator and loader. Codebooks are precomputed once and stored as JSON.

```python
# Key functions:
def build_lloyd_max_codebook(dim: int, bits: int, iterations: int = 300) -> CodebookData
    """Compute optimal codebook for Beta(dim/2, dim/2) distribution.

    Returns CodebookData with:
      - centroids: mx.array of shape (2**bits,) — reconstruction levels
      - boundaries: mx.array of shape (2**bits + 1,) — decision boundaries
      - metadata: dim, bits, distortion achieved
    """

def load_codebook(dim: int, bits: int) -> CodebookData
    """Load precomputed codebook from package data."""

def verify_codebook(cb: CodebookData) -> bool
    """Check monotonicity, symmetry, boundary containment."""
```

**Implementation detail**: The Beta(d/2, d/2) PDF is symmetric around 0.5, so we compute on normalized [0,1] then shift to [-1, 1]. For dim=128, bits=3: 8 centroids, 9 boundaries. Lloyd-Max iterates: (1) update centroids as conditional expectations E[x | x in bin_i] under Beta PDF, (2) update boundaries as midpoints between adjacent centroids.

**Codebook storage**: `mlx_turboquant/codec/data/{dim}_{bits}.json` — small files (~200 bytes each).

#### `mlx_turboquant/codec/transforms.py`

Random rotation using native MLX Hadamard transform.

```python
@dataclass
class TransformState:
    signs_1: mx.array  # Rademacher signs, shape (head_dim,)
    signs_2: mx.array  # Rademacher signs, shape (head_dim,)
    scale: float        # 1/sqrt(head_dim) normalization

def create_transform(head_dim: int, seed: int = 42) -> TransformState
    """Create deterministic random rotation state."""

def forward_transform(x: mx.array, state: TransformState) -> mx.array
    """Apply: x → diag(s2) @ H @ diag(s1) @ x, where H is Hadamard.

    Uses mx.hadamard_transform on last axis.
    Operates on batched input: (..., head_dim) → (..., head_dim)
    """

def inverse_transform(x: mx.array, state: TransformState) -> mx.array
    """Inverse rotation: exact inverse since Hadamard is self-inverse
    and sign flips are self-inverse.

    x → diag(s1) @ H @ diag(s2) @ x
    """
```

**Key insight**: `mx.hadamard_transform(a)` operates on the last axis and returns `H @ a` where H is the Hadamard matrix. The transform is self-inverse (H^2 = I up to scaling), so inverse is the same operation with signs swapped.

#### `mlx_turboquant/codec/packbits.py`

Bit-packing for memory-efficient storage.

```python
def pack(indices: mx.array, bits: int) -> mx.array
    """Pack low-bit indices into uint32 array.

    Packing schemes:
      - 2-bit: 16 values per uint32 (32 bits used)
      - 3-bit: 10 values per uint32 (30 bits used, 2 wasted)
      - 4-bit: 8 values per uint32 (32 bits used)

    Input: (..., head_dim) uint8 with values in [0, 2**bits)
    Output: (..., packed_dim) uint32
    """

def unpack(packed: mx.array, bits: int, head_dim: int) -> mx.array
    """Unpack uint32 array back to per-coordinate indices.

    Output: (..., head_dim) uint8
    """
```

**Implementation**: Use vectorized bitwise operations (`mx.left_shift`, `mx.bitwise_or`, `mx.right_shift`, `mx.bitwise_and`). The 3-bit case packs 10 values into 30 bits of a uint32 — the last 2 bits are unused. For head_dim=128 at 3-bit: 128/10 = 13 uint32s = 52 bytes per vector (vs 256 bytes at fp16).

#### `mlx_turboquant/codec/stage1_codec.py`

The main codec that ties everything together.

```python
@dataclass
class CompressedTensor:
    packed: mx.array      # (..., packed_dim) uint32 — quantized indices
    norms: mx.array       # (...,) float16 — per-vector L2 norms
    config: CodecConfig   # bits, head_dim, seed

@dataclass
class CodecConfig:
    head_dim: int
    bits: int
    seed: int = 42

class Stage1Codec:
    def __init__(self, config: CodecConfig):
        self.config = config
        self.codebook = load_codebook(config.head_dim, config.bits)
        self.transform = create_transform(config.head_dim, config.seed)

    def encode(self, x: mx.array) -> CompressedTensor:
        """Compress KV vectors.

        Input: (..., head_dim) float16/float32
        Pipeline: normalize → rotate → quantize → pack
        """

    def decode(self, compressed: CompressedTensor) -> mx.array:
        """Decompress to float16.

        Pipeline: unpack → dequantize (codebook lookup) → inverse rotate → rescale
        """

    def encode_decode(self, x: mx.array) -> mx.array:
        """Round-trip for quality measurement."""
```

**Encode pipeline (vectorized, no loops)**:
1. Compute per-vector L2 norms: `norms = mx.linalg.norm(x, axis=-1)`
2. Normalize: `x_normed = x / (norms[..., None] + 1e-8)`
3. Forward Hadamard: `x_rot = forward_transform(x_normed, self.transform)`
4. Quantize via broadcast: `diffs = mx.abs(x_rot[..., None] - boundaries[None, :])` → `indices = mx.argmin(diffs, axis=-1)` (clamp to [0, 2^bits - 1])
5. Pack: `packed = pack(indices, bits)`

**Decode pipeline (vectorized)**:
1. Unpack: `indices = unpack(packed, bits, head_dim)`
2. Dequantize: `x_rot = centroids[indices]` (fancy indexing)
3. Inverse Hadamard: `x_normed = inverse_transform(x_rot, self.transform)`
4. Rescale: `x = x_normed * norms[..., None]`

### Tests

#### `tests/test_codebooks.py`
- `test_lloyd_max_converges`: 300 iterations produces stable centroids
- `test_codebook_monotonicity`: centroids and boundaries are strictly monotone
- `test_codebook_symmetry`: centroids symmetric around 0 for symmetric Beta
- `test_codebook_coverage`: boundaries span [-1, 1]
- `test_precomputed_codebooks_load`: all 6 codebook files load correctly
- `test_codebook_verify`: `verify_codebook()` passes for all precomputed

#### `tests/test_transforms.py`
- `test_forward_inverse_roundtrip`: `inverse(forward(x)) ≈ x` within float32 tolerance
- `test_transform_preserves_norm`: `||forward(x)|| ≈ ||x||`
- `test_transform_deterministic`: same seed → same transform state
- `test_transform_batched`: works on (batch, seq, head_dim) inputs

#### `tests/test_packbits.py`
- `test_pack_unpack_roundtrip_2bit`: pack then unpack recovers exact indices
- `test_pack_unpack_roundtrip_3bit`: same for 3-bit
- `test_pack_unpack_roundtrip_4bit`: same for 4-bit
- `test_packed_size_reduction`: verify expected compression ratios
- `test_pack_batched`: works on multi-dimensional inputs

#### `tests/test_codec.py`
- `test_encode_decode_roundtrip_quality`: MSE < threshold for random Gaussian vectors
- `test_encode_decode_batched`: works on (batch, seq, head_dim)
- `test_compressed_tensor_sizes`: packed tensor has expected shape
- `test_codec_deterministic`: same input → same compressed output
- `test_higher_bits_lower_error`: 4-bit MSE < 3-bit MSE < 2-bit MSE
- `test_inner_product_preservation`: `<x, y> ≈ <decode(encode(x)), y>` — the property attention needs

### Exit Criteria
- [x] All 6 precomputed codebooks generated and stored
- [x] `encode()` → `decode()` round-trip NMSE within theoretical bounds:
  - 4-bit: NMSE < 0.015 (theoretical ~0.009)
  - 3-bit: NMSE < 0.05 (theoretical ~0.034)
  - 2-bit: NMSE < 0.15 (theoretical ~0.115)
- [x] Bit-packing achieves expected compression: 3-bit head_dim=128 → 52 bytes + 2 bytes norm = 54 bytes (vs 256 bytes fp16 = **4.7x compression**)
- [x] All tests pass (`pytest tests/test_codebooks.py tests/test_transforms.py tests/test_packbits.py tests/test_codec.py`)
- [x] Inner product preservation error < 5% at 3-bit for random vectors
- [x] No Python loops in encode/decode hot path — fully vectorized MLX ops

### Post-Review Fixes Applied
- Replaced Python loop in `pack()` with vectorized `mx.sum` (CLAUDE.md rule 2 compliance — sum == OR when bits don't overlap)
- Extracted `_bin_slice()` helper to eliminate duplicated bin-mask logic in `codebooks.py`
- Removed `_integrate()` wrapper, inlined `np.trapezoid` with standard arg order
- Removed no-op `mx.clip` in quantization
- Added config mismatch guard in `decode()`
- Updated NMSE thresholds to match theoretical values (d * per_coordinate_distortion)

---

## Phase 2 — Compressed Cache Layer [COMPLETE]

**Completed:** 2026-03-28

### Goal
Build a `CompressedKVCache` that is duck-type compatible with MLX-LM's `KVCache`, stores KV vectors in compressed form using the Phase 1 codec, and provides accurate memory accounting.

### Deliverables
- `CompressedKVCache` class matching MLX-LM's cache protocol
- Memory accounting with formula estimates and actual measurement
- Cache configuration and layout management
- Support for Qwen's GQA (fewer KV heads than query heads)

### Files to Implement

#### `mlx_turboquant/cache/compressed_cache.py`

The core cache adapter.

```python
class CompressedKVCache:
    """KV cache that stores vectors in compressed form.

    Duck-typed to match mlx_lm.models.cache.KVCache protocol:
      - update_and_fetch(keys, values) -> (all_keys, all_values)
      - state property (for serialization)
      - offset property
      - meta_state property (for safetensors serialization)
      - is_trimable() / trim(n)

    CRITICAL: Does NOT expose a `bits` attribute.
    MLX-LM checks hasattr(cache, 'bits') to dispatch to
    quantized_scaled_dot_product_attention, which uses affine
    group quantization — incompatible with our rotation+codebook approach.
    """

    def __init__(
        self,
        codec: Stage1Codec,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 4096,
    ):
        ...

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Store new KV vectors (compressed), return all KV vectors (decompressed).

        Input keys/values: (1, num_kv_heads, new_tokens, head_dim)
        Returns: decompressed (1, num_kv_heads, total_seq_len, head_dim) float16

        On write: encode new vectors through codec, append packed data
        On read: decode all stored vectors back to float16
        """

    @property
    def offset(self) -> int:
        """Current sequence length stored in cache."""

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        """Return decompressed state for compatibility."""

    @property
    def meta_state(self) -> dict:
        """Serialization metadata for safetensors."""

    def trim(self, n: int) -> int:
        """Trim first n tokens. For prompt caching compatibility."""

    @property
    def nbytes(self) -> int:
        """Actual compressed storage in bytes."""
```

**Storage layout**: Pre-allocate in 256-step chunks (matching MLX-LM's `KVCache` strategy).
Each layer stores:
- `packed_keys`: `(1, num_kv_heads, max_chunks, packed_dim)` uint32
- `packed_values`: `(1, num_kv_heads, max_chunks, packed_dim)` uint32
- `key_norms`: `(1, num_kv_heads, max_chunks)` float16
- `value_norms`: `(1, num_kv_heads, max_chunks)` float16
- `_offset`: int tracking current position

#### `mlx_turboquant/cache/memory_accounting.py`

```python
@dataclass
class MemoryReport:
    baseline_bytes: int
    compressed_bytes: int
    compression_ratio: float
    bytes_per_token_baseline: int
    bytes_per_token_compressed: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    kv_bits: int
    seq_len: int

class MemoryAccountant:
    def estimate(
        self, num_layers: int, num_kv_heads: int, head_dim: int,
        kv_bits: int, seq_len: int
    ) -> MemoryReport:
        """Formula-based memory estimate."""

    def measure(self, cache_layers: list[CompressedKVCache]) -> MemoryReport:
        """Actual measurement from live cache objects."""
```

**Baseline formula**: `2 * num_layers * num_kv_heads * head_dim * 2 * seq_len` bytes (K+V, fp16)
**Compressed formula**: `2 * num_layers * num_kv_heads * (ceil(head_dim * bits / 32) * 4 + 2) * seq_len` bytes

For Qwen2.5-3B (36 layers, 2 KV heads, head_dim=128) at 4096 tokens:
- Baseline fp16: 36 * 2 * 128 * 2 * 2 * 4096 = **150 MB**
- Compressed 3-bit: 36 * 2 * (54) * 2 * 4096 = **32 MB** (~4.7x reduction)

#### `mlx_turboquant/cache/cache_layout.py`

```python
@dataclass
class CacheConfig:
    num_layers: int
    num_kv_heads: int  # GQA: may be < num_attention_heads
    head_dim: int
    max_seq_len: int
    kv_bits: int
    backend: str = "reference"  # "reference" | "metal"

def create_cache_layers(config: CacheConfig, codec: Stage1Codec) -> list[CompressedKVCache]:
    """Create per-layer cache instances."""
```

### Tests

#### `tests/test_compressed_cache.py`
- `test_update_and_fetch_single_token`: store 1 token, retrieve 1 token, check shape
- `test_update_and_fetch_prefill`: store 100 tokens in one call, verify shapes
- `test_update_and_fetch_incremental`: prefill then add tokens one at a time
- `test_gqa_head_counts`: num_kv_heads=2 works correctly (Qwen 2.5 GQA)
- `test_no_bits_attribute`: `hasattr(cache, 'bits')` is False
- `test_offset_tracking`: offset matches number of tokens stored
- `test_trim`: trim first n tokens, verify remaining
- `test_nbytes_accuracy`: `nbytes` matches formula estimate within 5%

#### `tests/test_memory_accounting.py`
- `test_estimate_matches_formula`: known model params → expected bytes
- `test_measure_matches_estimate`: live cache measurement ≈ formula estimate
- `test_compression_ratio`: 3-bit compression > 4x for typical models

### Exit Criteria
- [x] `CompressedKVCache` satisfies MLX-LM cache protocol (duck-typing)
- [x] `hasattr(cache, 'bits')` returns `False` (critical for correct SDPA dispatch)
- [x] GQA with num_kv_heads < num_attention_heads works correctly
- [x] Memory accounting formula matches measured usage within 5%
- [x] Incremental token-by-token updates produce same output as batch prefill
- [x] All cache tests pass

### Post-Review Fixes Applied
- Replaced array-slicing nbytes with pure arithmetic (avoids graph node creation)
- Fixed state setter type to match getter (tuple[mx.array | None, mx.array | None])
- Removed 3 type: ignore[index] suppressions
- Noted O(seq_len) full-decode-per-step as known optimization for Phase 5

---

## Phase 3 — MLX-LM Integration + CLI [COMPLETE]

**Completed:** 2026-03-28

### Goal
Wire the compressed cache into MLX-LM's generation pipeline for Qwen models, expose a CLI, and enable side-by-side baseline vs compressed generation.

### Deliverables
- Model introspection to auto-detect Qwen layer/head/dim parameters
- Generate wrapper that threads compressed cache through `generate_step()`
- CLI with `generate`, `compare`, and `info` commands
- Fallback to baseline cache for A/B comparison

### Files to Implement

#### `mlx_turboquant/integration/mlx_lm_adapter.py`

```python
def introspect_model(model) -> CacheConfig:
    """Auto-detect model architecture parameters.

    Inspects model.layers[0].self_attn to find:
      - num_layers: len(model.layers)
      - num_kv_heads: from model config or attention module
      - head_dim: from model config or weight shapes

    Returns CacheConfig populated from the model.
    """

def make_compressed_cache(
    model, *, kv_bits: int = 3, max_seq_len: int = 4096
) -> list[CompressedKVCache]:
    """Create compressed cache layers for a loaded MLX-LM model.

    Analogous to mlx_lm.models.cache.make_prompt_cache().
    """
```

#### `mlx_turboquant/integration/generate_wrapper.py`

```python
def generate_with_compressed_cache(
    model,
    tokenizer,
    prompt: str,
    *,
    kv_bits: int = 3,
    max_tokens: int = 256,
    max_seq_len: int = 4096,
    temp: float = 0.0,
    verbose: bool = False,
) -> GenerationResult:
    """Generate text using compressed KV cache.

    CRITICAL: Passes kv_bits=None to generate_step() to disable
    MLX-LM's built-in affine KV quantization. Our CompressedKVCache
    handles compression internally.

    Returns GenerationResult with text, timing, and memory stats.
    """

def generate_baseline(
    model, tokenizer, prompt: str, *, max_tokens: int = 256, temp: float = 0.0
) -> GenerationResult:
    """Generate with standard MLX-LM cache for comparison."""

@dataclass
class GenerationResult:
    text: str
    tokens: list[int]
    tokens_generated: int
    ttft_ms: float
    decode_tokens_per_sec: float
    cache_bytes: int  # logical occupied cache bytes
    cache_mode: str  # "baseline" | "compressed-Xbit"
    cache_allocated_bytes: int | None
```

#### `mlx_turboquant/cli.py`

```python
def main():
    """CLI entry point: mlx-tq"""
    # Subcommands:
    #   generate  — generate text, baseline by default
    #   compare   — run baseline and compressed side-by-side
    #   info      — show model info and memory estimates

    # mlx-tq generate --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    #                  --prompt "Explain quantum computing" \
    #                  --cache-mode baseline

    # mlx-tq generate --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    #                  --prompt "Explain quantum computing" \
    #                  --cache-mode compressed --kv-bits 3

    # mlx-tq compare --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    #                --prompt "Explain quantum computing" \
    #                --kv-bits 3

    # mlx-tq info --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
```

### Tests

#### `tests/test_integration.py`
- `test_introspect_qwen_0_5b`: correctly detects 14 layers, 2 KV heads, head_dim=64
- `test_generate_compressed_produces_text`: generates non-empty output
- `test_generate_baseline_produces_text`: baseline generation works
- `test_compare_produces_both_results`: both modes return valid results
- `test_compressed_output_drift_is_measurable`: compressed output quality drift is surfaced instead of hidden
- `test_generate_result_has_metrics`: timing plus logical/allocated cache fields populated

#### `tests/test_cli.py`
- `test_cli_info_runs`: `mlx-tq info` exits cleanly
- `test_cli_generate_help`: `mlx-tq generate --help` shows usage
- `test_cli_compare_smoke`: end-to-end comparison on tiny model

**Note**: Integration tests require downloading a model. They are marked with `@pytest.mark.slow`, and the CLI smoke coverage should skip automatically when external model access or Hugging Face authentication is unavailable.

### Exit Criteria
- [x] `mlx-tq generate --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --prompt "Hello"` runs baseline generation with no extra flags
- [x] `mlx-tq generate --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --prompt "Hello" --cache-mode compressed --kv-bits 3` runs the experimental compressed path end to end
- [x] `mlx-tq compare` shows side-by-side baseline vs compressed output with metrics
- [x] `mlx-tq info` displays model architecture and memory estimates
- [x] No modifications to MLX-LM source code — pure wrapper/adapter pattern
- [x] Compressed generation runs without OOM on M4 Mini 16 GB
- [x] All integration tests pass with Qwen 0.5B model

### Post-Review Fixes Applied
- Fixed float16 norm overflow bug (RoPE'd keys overflow float16 sum-of-squares, causing NaN — now computes in float32)
- Renamed prefill_time_ms to ttft_ms (time-to-first-token) for accurate labeling
- Pass max_tokens to generate_step (was relying on generate_step's internal default)
- Separated ModelInfo from CacheConfig (introspect_model no longer returns CacheConfig with hardcoded kv_bits)
- Extracted _print_result CLI helper to eliminate 3x copy-paste
- Added zero-guard on compression ratio division
- Standardized docs, tests, and examples on the `mlx-community/Qwen2.5-*` model IDs
- Corrected README onboarding to clone this repository instead of an unrelated project
- Added CLI validation for `--kv-bits` so unsupported values fail at argument parsing
- Added concise user-facing model load errors for repo/auth/network failures
- Hardened the slow CLI smoke test to skip on external model-access failures instead of failing the suite

---

## Phase 4 — Benchmark Harness [COMPLETE]

**Completed:** 2026-03-28

### Goal
Build a reproducible, rigorous benchmark suite that measures memory reduction, latency impact, and output quality, producing publication-quality reports.

### Deliverables
- Memory benchmark across sequence lengths (512, 1024, 2048, 4096, 8192)
- Latency benchmark with proper `mx.synchronize()` timing
- Quality benchmark: token-match ratio, perplexity delta, attention fidelity
- JSON + Markdown report generation
- Comparison framework: baseline vs 2-bit vs 3-bit vs 4-bit

### Files to Implement

#### `mlx_turboquant/bench/memory.py`

```python
def benchmark_memory(
    model_path: str,
    seq_lengths: list[int] = [512, 1024, 2048, 4096, 8192],
    kv_bits_list: list[int] = [2, 3, 4],
) -> list[MemoryBenchResult]:
    """Measure actual memory usage across configurations.

    For each (seq_len, kv_bits):
      - Create compressed cache
      - Fill with seq_len tokens of random KV data
      - Measure cache.nbytes
      - Compare to baseline fp16 cache
    """

@dataclass
class MemoryBenchResult:
    model: str
    seq_len: int
    kv_bits: int
    baseline_mb: float
    compressed_mb: float
    compression_ratio: float
    overhead_bytes: int  # norm storage + padding
```

#### `mlx_turboquant/bench/latency.py`

```python
def benchmark_latency(
    model_path: str,
    prompt: str,
    max_tokens: int = 100,
    kv_bits_list: list[int] = [3, 4],
    warmup_runs: int = 2,
    bench_runs: int = 5,
) -> list[LatencyBenchResult]:
    """Measure prefill and decode latency with proper synchronization.

    CRITICAL: Call mx.synchronize() before timing to ensure
    lazy evaluation is complete. MLX's lazy eval means wall-clock
    timing without sync is meaningless.
    """

@dataclass
class LatencyBenchResult:
    model: str
    cache_mode: str
    kv_bits: int | None
    prefill_ms: float
    decode_tok_per_sec: float
    total_time_ms: float
    tokens_generated: int
```

#### `mlx_turboquant/bench/quality.py`

```python
def benchmark_quality(
    model_path: str,
    prompts: list[str],
    kv_bits_list: list[int] = [2, 3, 4],
    max_tokens: int = 200,
) -> list[QualityBenchResult]:
    """Measure output quality degradation.

    Metrics:
      - token_match_ratio: fraction of tokens identical to baseline (temp=0)
      - first_divergence_position: where compressed output first differs
      - perplexity_delta: difference in per-token perplexity
    """

@dataclass
class QualityBenchResult:
    model: str
    prompt_id: str
    kv_bits: int
    token_match_ratio: float
    first_divergence_position: int
    baseline_perplexity: float
    compressed_perplexity: float
    perplexity_delta: float
```

#### `mlx_turboquant/bench/prompts.py`

```python
BENCHMARK_PROMPTS: dict[str, str] = {
    "short_qa": "What is the capital of France?",
    "medium_instruct": "Explain how a transformer attention mechanism works...",
    "long_context": "<loaded from file: a 2000+ token passage>",
    "code_gen": "Write a Python function that implements binary search...",
}
```

#### Report generation

```python
def generate_report(
    memory_results: list[MemoryBenchResult],
    latency_results: list[LatencyBenchResult],
    quality_results: list[QualityBenchResult],
    output_dir: str,
) -> None:
    """Generate JSON + Markdown benchmark report.

    Produces:
      - results.json (machine-readable)
      - BENCHMARKS.md (human-readable with tables)
    """
```

### CLI Extension

```bash
mlx-tq bench --model mlx-community/Qwen2.5-3B-Instruct-4bit --suite full
mlx-tq bench --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --suite quick
```

### Tests

#### `tests/test_bench.py`
- `test_memory_bench_runs`: produces results with positive compression ratios
- `test_latency_bench_format`: results have all required fields
- `test_quality_bench_token_match`: 4-bit has higher match ratio than 2-bit
- `test_report_generates_files`: JSON and Markdown files created

### Exit Criteria
- [x] `mlx-tq bench --suite quick` completes on Qwen 0.5B in < 5 minutes
- [x] Memory benchmark shows 3-bit achieves > 4x compression ratio
- [ ] Current quick benchmark on Qwen 0.5B does not meet the decode-overhead target; recent runs measured roughly 36-38% slowdown vs baseline
- [ ] Current quick benchmark on Qwen 0.5B does not meet the 3-bit quality target; recent runs measured 0-6% token match on the sample prompts
- [x] Reports render correctly as Markdown tables
- [x] All benchmark results are reproducible (< 5% variance across runs)

### Post-Review Fixes Applied
- Bench modules compose existing code (GenerationResult, estimate_memory, generate_with_compressed_cache) instead of reimplementing
- benchmark_latency/quality take loaded model+tokenizer, not model path (avoids expensive reloading)
- Memory benchmark is pure calculation — no model loading needed
- `GenerationResult.cache_bytes` now records logical occupied cache bytes for both baseline and compressed modes
- `GenerationResult.cache_allocated_bytes` now carries raw backing-buffer allocation separately for short-run diagnostics
- `mlx-tq generate` now defaults to baseline mode; compressed mode is an explicit experimental opt-in
- Docs/examples now avoid unsupported latency and quality claims for the canonical sample model

---

## Phase 5 — Metal Optimization [COMPLETE]

**Completed:** 2026-03-28

### Goal
Replace the pure-MLX hot paths with custom Metal kernels for measurable speedup, while preserving bit-exact output matching the reference path.

### Deliverables
- **Kernel A**: Fused unpack + codebook lookup Metal shader
- **Kernel B** (stretch): Fused compressed attention using query-rotation trick
- Performance validation harness comparing Metal vs reference output
- Feature-gated: `backend="metal"` opt-in, `backend="reference"` default

### Files to Implement

#### `mlx_turboquant/kernels/metal_pack.py`

```python
def metal_unpack_dequantize(
    packed: mx.array,   # (..., packed_dim) uint32
    codebook: mx.array, # (2**bits,) float32 centroids
    bits: int,
    head_dim: int,
) -> mx.array:
    """Fused unpack + codebook lookup via Metal kernel.

    Instead of: unpack → index into codebook (2 passes over data),
    this kernel does both in a single pass:
      1. Extract bit-field from packed uint32
      2. Immediately look up centroid value
      3. Write float16 output

    Grid: one thread per output element.
    """

# Metal shader source (3-bit variant):
METAL_UNPACK_DEQUANT_3BIT = """
    uint elem = thread_position_in_grid.x;
    uint word_idx = elem / 10;       // 10 values per uint32
    uint bit_offset = (elem % 10) * 3;
    uint packed_word = packed[word_idx];
    uint index = (packed_word >> bit_offset) & 0x7;  // 3-bit mask
    out[elem] = codebook[index];
"""
```

#### `mlx_turboquant/kernels/metal_attention.py`

```python
def metal_compressed_attention(
    queries: mx.array,       # (1, num_heads, 1, head_dim)
    packed_keys: mx.array,   # (1, num_kv_heads, seq_len, packed_dim) uint32
    key_norms: mx.array,     # (1, num_kv_heads, seq_len)
    packed_values: mx.array, # (1, num_kv_heads, seq_len, packed_dim) uint32
    value_norms: mx.array,   # (1, num_kv_heads, seq_len)
    codebook: mx.array,      # centroids
    transform_signs: tuple[mx.array, mx.array],
    scale: float,
    bits: int,
) -> mx.array:
    """Fused compressed attention using query-rotation trick.

    KEY INSIGHT: Instead of decompressing all K vectors (O(seq_len * head_dim)):
      1. Forward-Hadamard the query: q_rot = H @ diag(s1) @ q  — O(head_dim)
      2. Dot-product in rotated domain: score_i = q_rot · (centroid[idx] * norm_i)
      3. This works because <q, H^-1 @ x_rot> = <H @ q, x_rot>

    Still need full V decompression for weighted sum (or approximate).

    This is the highest-value optimization — turns O(S*D) decompression
    into O(D) rotation + O(S*D) cheap centroid dot products.
    """
```

**Note**: Kernel B is a stretch goal. If attention fidelity suffers from operating in the rotated domain for values, fall back to decompressing V only.

### Tests

#### `tests/test_metal_kernels.py`
- `test_metal_unpack_matches_reference`: bit-exact match with pure-MLX unpack
- `test_metal_dequantize_matches_reference`: centroid lookup matches
- `test_metal_fused_matches_reference`: fused unpack+dequantize matches two-step
- `test_metal_attention_matches_reference`: compressed attention ≈ standard attention
- `test_metal_performance_improvement`: Metal path faster than reference at seq_len >= 256

#### `tests/test_backend_parity.py`
- `test_full_generation_metal_vs_reference`: same prompt + seed → same output
- `test_backend_fallback`: Metal not available → graceful fallback to reference

### Exit Criteria
- [x] Kernel A (unpack+dequantize) produces bit-exact output vs reference
- [x] Kernel A shows measurable speedup (>20%) at seq_len >= 512
- [x] Metal path is feature-gated behind `backend="metal"` flag
- [x] Full generation with Metal backend produces same text as reference backend
- [x] All parity tests pass
- [ ] (Stretch) Kernel B shows >2x attention speedup at seq_len >= 1024

### Post-Review Fixes Applied
- Templated Metal shader sources from VALUES_PER_WORD constants (eliminated 3x copy-paste)
- Renamed _VALUES_PER_WORD to VALUES_PER_WORD (public API, used across module boundaries)
- Moved lazy import to module level (was running on every decode() call)
- Replaced math.prod with packed.size for batch element count
- 1.62x speedup validated for fused unpack+dequant kernel

---

## Phase 6 — Polish and Ship [COMPLETE]

**Completed:** 2026-03-28

### Goal
Final quality pass: documentation, README with real benchmarks, example notebooks, and clean git history.

### Deliverables
- README with architecture diagram, benchmark results table, quickstart
- Example script: `examples/quickstart.py`
- `BENCHMARKS.md` with real numbers from M4 Mini 16 GB
- API documentation (docstrings, type hints complete)
- Clean commit history (squash/rebase if needed)
- Tag v0.1.0 release

### Files to Create/Update
- `README.md` — project overview, installation, quickstart, benchmarks, architecture
- `examples/quickstart.py` — minimal working example
- `examples/benchmark_qwen.py` — reproduce published benchmarks
- `BENCHMARKS.md` — real benchmark results with hardware info
- All docstrings reviewed and complete

### Exit Criteria
- [x] `pip install .` from clean clone works
- [x] README quickstart example runs successfully
- [x] All benchmarks reproducible on M4 Mini 16 GB
- [x] `ruff check .` clean, `mypy mlx_turboquant/` clean
- [x] Full test suite green (`pytest --slow` for integration tests)
- [x] v0.1.0 tagged

---

## Dependency Chain

```
Phase 0 (Foundation)
    ↓
Phase 1 (Core Codec)
    ↓
Phase 2 (Cache Layer)
    ↓
Phase 3 (Integration + CLI)
    ↓
    ├── Phase 4 (Benchmarks)      ← can run in parallel with Phase 5
    └── Phase 5 (Metal Kernels)   ← can run in parallel with Phase 4
         ↓
    Phase 6 (Polish & Ship)
```

Phases 0→3 are strictly sequential. Phases 4 and 5 can be developed in parallel. Phase 6 follows completion of both.

## Verification

### End-to-End Smoke Test (after Phase 3)
```bash
# Install
uv sync

# Generate with the baseline cache (default)
mlx-tq generate \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "Explain KV cache compression in transformers" \
  --max-tokens 200

# Opt into the experimental compressed cache
mlx-tq generate \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "Explain KV cache compression in transformers" \
  --cache-mode compressed \
  --kv-bits 3 \
  --max-tokens 200

# Compare baseline vs compressed
mlx-tq compare \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "What is the meaning of life?" \
  --kv-bits 3
```

### Full Benchmark Suite (after Phase 4)
```bash
# Quick suite (< 5 min on M4 Mini 16GB)
mlx-tq bench --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --suite quick

# Full suite with larger model
mlx-tq bench --model mlx-community/Qwen2.5-3B-Instruct-4bit --suite full
```

### Metal Kernel Validation (after Phase 5)
```bash
# Parity test
pytest tests/test_metal_kernels.py tests/test_backend_parity.py -v

# Performance comparison
mlx-tq bench --model mlx-community/Qwen2.5-3B-Instruct-4bit --backend metal --suite quick
```

### CI Pipeline
```bash
# What CI runs on every push
uv run ruff check .
uv run mypy mlx_turboquant/
uv run pytest tests/ -m "not slow" -v
```
