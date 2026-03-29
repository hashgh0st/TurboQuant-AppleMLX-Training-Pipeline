"""CLI entry point for mlx-tq: generate, compare, info commands."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from mlx_turboquant.constants import SUPPORTED_KV_BITS
from mlx_turboquant.integration.generate_wrapper import GenerationResult


def _print_result(result: GenerationResult, header: str | None = None) -> None:
    """Print a generation result with metrics."""
    label = header or result.cache_mode
    print(f"\n--- {label} ---")
    print(result.text)
    print(f"\nTokens: {result.tokens_generated}")
    print(f"TTFT: {result.ttft_ms:.1f} ms")
    print(f"Decode: {result.decode_tokens_per_sec:.1f} tok/s")
    print(f"Cache (logical): {result.cache_bytes / 1024:.1f} KB")
    if (
        result.cache_allocated_bytes is not None
        and result.cache_allocated_bytes != result.cache_bytes
    ):
        print(f"Allocated cache: {result.cache_allocated_bytes / 1024:.1f} KB")


def _format_model_load_error(model_name: str, exc: Exception) -> str:
    """Render a concise, user-facing model load error."""
    detail = str(exc)
    lowered = detail.lower()

    if (
        "repository not found" in lowered
        or "repositorynotfounderror" in lowered
        or "401" in lowered
        or "invalid username or password" in lowered
    ):
        reason = "repo not found or access denied"
        next_step = "Verify the model ID and Hugging Face authentication."
    elif "403" in lowered or "gated" in lowered:
        reason = "model access is gated"
        next_step = "Authenticate with Hugging Face and accept any required model terms."
    elif (
        "connect" in lowered
        or "network" in lowered
        or "timed out" in lowered
        or "name or service not known" in lowered
        or "temporary failure" in lowered
    ):
        reason = "network or download error"
        next_step = "Check connectivity and retry."
    else:
        reason = "unexpected loader error"
        next_step = detail.splitlines()[-1] if detail else exc.__class__.__name__

    return f"Failed to load model '{model_name}': {reason}. {next_step}"


def _load_model(model_name: str) -> tuple[Any, Any]:
    """Load an MLX-LM model and exit cleanly on expected user-facing failures."""
    from mlx_lm import load

    try:
        loaded = load(model_name)
    except Exception as exc:
        raise SystemExit(_format_model_load_error(model_name, exc)) from None
    return loaded[0], loaded[1]


def _add_kv_bits_argument(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    """Add a validated kv-bits argument shared by CLI subcommands."""
    parser.add_argument(
        "--kv-bits",
        type=int,
        choices=SUPPORTED_KV_BITS,
        default=3,
        help=help_text,
    )


def _add_value_kv_bits_argument(parser: argparse.ArgumentParser) -> None:
    """Add optional value-branch bit selection for experimental profiles."""
    parser.add_argument(
        "--value-kv-bits",
        type=int,
        choices=SUPPORTED_KV_BITS,
        default=None,
        help="Experimental value-branch compression bits (defaults to --kv-bits).",
    )


def _add_backend_argument(parser: argparse.ArgumentParser) -> None:
    """Add experimental backend selection for compressed generation."""
    parser.add_argument(
        "--backend",
        choices=["reference", "metal"],
        default="reference",
        help="Compressed-cache backend (default: reference).",
    )


def _add_sink_tokens_argument(parser: argparse.ArgumentParser) -> None:
    """Add attention-sink token count for quality preservation."""
    parser.add_argument(
        "--sink-tokens",
        type=int,
        default=0,
        help="Keep first N tokens in FP16 (attention sink, default: 0).",
    )


def _cmd_generate(args: argparse.Namespace) -> None:
    """Generate text with baseline or compressed KV cache."""
    from mlx_turboquant.integration.generate_wrapper import (
        generate_baseline,
        generate_with_compressed_cache,
    )

    print(f"Loading model: {args.model}...")
    model, tokenizer = _load_model(args.model)

    if args.cache_mode == "compressed":
        result = generate_with_compressed_cache(
            model,
            tokenizer,
            args.prompt,
            kv_bits=args.kv_bits,
            value_kv_bits=args.value_kv_bits,
            backend=args.backend,
            max_tokens=args.max_tokens,
            temp=args.temp,
            sink_tokens=args.sink_tokens,
        )
    else:
        result = generate_baseline(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            temp=args.temp,
        )

    _print_result(result)


def _cmd_compare(args: argparse.Namespace) -> None:
    """Run baseline and compressed generation side-by-side."""
    from mlx_turboquant.integration.generate_wrapper import (
        generate_baseline,
        generate_with_compressed_cache,
    )

    print(f"Loading model: {args.model}...")
    model, tokenizer = _load_model(args.model)

    print("Running baseline...")
    baseline = generate_baseline(
        model,
        tokenizer,
        args.prompt,
        max_tokens=args.max_tokens,
        temp=args.temp,
    )

    compressed_label = (
        f"{args.kv_bits}-bit"
        if args.value_kv_bits in (None, args.kv_bits) and args.backend == "reference"
        else (
            f"k{args.kv_bits}/v{args.value_kv_bits or args.kv_bits}-bit"
            + (f" ({args.backend})" if args.backend != "reference" else "")
        )
    )
    print(f"Running compressed ({compressed_label})...")
    compressed = generate_with_compressed_cache(
        model,
        tokenizer,
        args.prompt,
        kv_bits=args.kv_bits,
        value_kv_bits=args.value_kv_bits,
        backend=args.backend,
        max_tokens=args.max_tokens,
        temp=args.temp,
        sink_tokens=args.sink_tokens,
    )

    _print_result(baseline, header="BASELINE")
    _print_result(compressed, header=compressed.cache_mode)

    if baseline.cache_bytes > 0 and compressed.cache_bytes > 0:
        ratio = baseline.cache_bytes / compressed.cache_bytes
        savings = (1 - compressed.cache_bytes / baseline.cache_bytes) * 100
        print(f"\nCompression: {ratio:.1f}x ({savings:.0f}% memory saved)")

    baseline_allocated = baseline.cache_allocated_bytes or baseline.cache_bytes
    compressed_allocated = compressed.cache_allocated_bytes or compressed.cache_bytes
    if (
        baseline_allocated > 0
        and compressed_allocated > 0
        and (
            baseline_allocated != baseline.cache_bytes
            or compressed_allocated != compressed.cache_bytes
        )
    ):
        alloc_ratio = baseline_allocated / compressed_allocated
        print(
            "Allocated buffers: "
            f"{baseline_allocated / 1024:.1f} KB vs "
            f"{compressed_allocated / 1024:.1f} KB "
            f"({alloc_ratio:.1f}x)"
        )


def _cmd_info(args: argparse.Namespace) -> None:
    """Show model architecture and memory estimates."""
    from mlx_turboquant.cache.memory_accounting import estimate_memory
    from mlx_turboquant.integration.mlx_lm_adapter import introspect_model

    print(f"Loading model: {args.model}...")
    model, _tokenizer = _load_model(args.model)
    info = introspect_model(model)

    print(f"\nModel: {args.model}")
    print(f"Layers: {info.num_layers}")
    print(f"KV heads: {info.num_kv_heads}")
    print(f"Head dim: {info.head_dim}")
    print(f"Max seq len: {info.max_seq_len}")

    print("\nMemory estimates at 4096 tokens:")
    for bits in (2, 3, 4):
        report = estimate_memory(
            num_layers=info.num_layers,
            num_kv_heads=info.num_kv_heads,
            head_dim=info.head_dim,
            kv_bits=bits,
            seq_len=4096,
        )
        print(
            f"  {bits}-bit: {report.compressed_bytes / 1024 / 1024:.1f} MB "
            f"(baseline: {report.baseline_bytes / 1024 / 1024:.1f} MB, "
            f"{report.compression_ratio:.1f}x)"
        )


def _cmd_bench(args: argparse.Namespace) -> None:
    """Run benchmark suite and generate report."""
    from mlx_turboquant.bench.latency import benchmark_latency
    from mlx_turboquant.bench.memory import benchmark_memory
    from mlx_turboquant.bench.prompts import BENCHMARK_PROMPTS, QUICK_PROMPTS
    from mlx_turboquant.bench.quality import benchmark_quality
    from mlx_turboquant.bench.report import generate_report
    from mlx_turboquant.integration.mlx_lm_adapter import introspect_model

    print(f"Loading model: {args.model}...")
    model, tokenizer = _load_model(args.model)
    info = introspect_model(model)

    is_quick = args.suite == "quick"
    prompts = QUICK_PROMPTS if is_quick else BENCHMARK_PROMPTS
    max_tokens = 50 if is_quick else 200
    runs = 1 if is_quick else 3
    warmup = 0 if is_quick else 1
    kv_bits_list = [args.kv_bits] if is_quick else [2, 3, 4]
    seq_lengths = [1024, 4096] if is_quick else None

    # Memory benchmark (pure calculation)
    print("Running memory benchmarks...")
    mem_results = benchmark_memory(
        num_layers=info.num_layers,
        num_kv_heads=info.num_kv_heads,
        head_dim=info.head_dim,
        seq_lengths=seq_lengths,
        kv_bits_list=kv_bits_list,
    )

    # Latency: single representative prompt (multiple prompts add noise, not signal)
    latency_prompt = next(iter(prompts.values()))
    print(f"Running latency benchmarks ({runs} runs)...")
    lat_results = benchmark_latency(
        model,
        tokenizer,
        latency_prompt,
        max_tokens=max_tokens,
        kv_bits_list=kv_bits_list,
        runs=runs,
        warmup=warmup,
    )

    # Quality benchmark
    print(f"Running quality benchmarks ({len(prompts)} prompts)...")
    qual_results = benchmark_quality(
        model,
        tokenizer,
        prompts,
        kv_bits_list=kv_bits_list,
        max_tokens=max_tokens,
    )

    # Evaluate promotion gates
    from mlx_turboquant.bench.promotion import evaluate_profiles

    verdicts = evaluate_profiles(qual_results, lat_results)

    # Generate report
    output_dir = args.output_dir
    generate_report(
        mem_results, lat_results, qual_results, output_dir,
        model_name=args.model, verdicts=verdicts,
    )
    print(f"\nReport written to {output_dir}/results.json and {output_dir}/BENCHMARKS.md")

    # Enforce gate if requested
    if args.gate:
        failed = [v for v in verdicts if not v.passes]
        if failed:
            print("\nPromotion gate FAILED:")
            for v in failed:
                print(f"  {v.cache_mode}: {', '.join(v.failures)}")
            raise SystemExit(1)
        print(f"\nPromotion gate PASSED: {len(verdicts)} profile(s) meet thresholds.")


def main() -> None:
    """Entry point for the mlx-tq CLI."""
    parser = argparse.ArgumentParser(
        prog="mlx-tq",
        description="mlx-turboquant: Apple-Silicon KV-cache compression for MLX-LM",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate
    gen = subparsers.add_parser("generate", help="Generate text")
    gen.add_argument("--model", required=True, help="HuggingFace model path")
    gen.add_argument("--prompt", required=True, help="Input prompt")
    gen.add_argument(
        "--cache-mode",
        choices=["baseline", "compressed"],
        default="baseline",
        help="Cache mode (default: baseline; compressed is experimental)",
    )
    _add_kv_bits_argument(gen, help_text="Compression bits (choices: 2, 3, 4; default: 3)")
    _add_value_kv_bits_argument(gen)
    _add_backend_argument(gen)
    _add_sink_tokens_argument(gen)
    gen.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    gen.add_argument("--temp", type=float, default=0.0, help="Sampling temperature")
    gen.set_defaults(func=_cmd_generate)

    # compare
    cmp = subparsers.add_parser("compare", help="Compare baseline vs compressed")
    cmp.add_argument("--model", required=True, help="HuggingFace model path")
    cmp.add_argument("--prompt", required=True, help="Input prompt")
    _add_kv_bits_argument(cmp, help_text="Compression bits (choices: 2, 3, 4; default: 3)")
    _add_value_kv_bits_argument(cmp)
    _add_backend_argument(cmp)
    _add_sink_tokens_argument(cmp)
    cmp.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    cmp.add_argument("--temp", type=float, default=0.0, help="Sampling temperature")
    cmp.set_defaults(func=_cmd_compare)

    # info
    info = subparsers.add_parser("info", help="Show model info and memory estimates")
    info.add_argument("--model", required=True, help="HuggingFace model path")
    info.set_defaults(func=_cmd_info)

    # bench
    bench = subparsers.add_parser("bench", help="Run benchmark suite")
    bench.add_argument("--model", required=True, help="HuggingFace model path")
    bench.add_argument(
        "--suite",
        choices=["quick", "full"],
        default="quick",
        help="Benchmark suite (default: quick)",
    )
    _add_kv_bits_argument(
        bench,
        help_text="Compression bits for quick suite (choices: 2, 3, 4; default: 3)",
    )
    bench.add_argument("--output-dir", default=".", help="Output directory for reports")
    bench.add_argument(
        "--gate",
        action="store_true",
        help="Enforce promotion thresholds; exit non-zero if any profile fails.",
    )
    bench.set_defaults(func=_cmd_bench)

    parsed = parser.parse_args()
    if not hasattr(parsed, "func"):
        parser.print_help()
        sys.exit(1)
    parsed.func(parsed)
