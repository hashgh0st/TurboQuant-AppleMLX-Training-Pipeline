"""CLI entry point for mlx-tq: generate, compare, info commands."""

from __future__ import annotations

import argparse
import sys

from mlx_turboquant.integration.generate_wrapper import GenerationResult


def _print_result(result: GenerationResult, header: str | None = None) -> None:
    """Print a generation result with metrics."""
    label = header or result.cache_mode
    print(f"\n--- {label} ---")
    print(result.text)
    print(f"\nTokens: {result.tokens_generated}")
    print(f"TTFT: {result.ttft_ms:.1f} ms")
    print(f"Decode: {result.decode_tokens_per_sec:.1f} tok/s")
    print(f"Cache: {result.cache_bytes / 1024:.1f} KB")


def _cmd_generate(args: argparse.Namespace) -> None:
    """Generate text with baseline or compressed KV cache."""
    from mlx_lm import load

    from mlx_turboquant.integration.generate_wrapper import (
        generate_baseline,
        generate_with_compressed_cache,
    )

    print(f"Loading model: {args.model}...")
    loaded = load(args.model)
    model, tokenizer = loaded[0], loaded[1]

    if args.cache_mode == "compressed":
        result = generate_with_compressed_cache(
            model, tokenizer, args.prompt,
            kv_bits=args.kv_bits, max_tokens=args.max_tokens, temp=args.temp,
        )
    else:
        result = generate_baseline(
            model, tokenizer, args.prompt, max_tokens=args.max_tokens, temp=args.temp,
        )

    _print_result(result)


def _cmd_compare(args: argparse.Namespace) -> None:
    """Run baseline and compressed generation side-by-side."""
    from mlx_lm import load

    from mlx_turboquant.integration.generate_wrapper import (
        generate_baseline,
        generate_with_compressed_cache,
    )

    print(f"Loading model: {args.model}...")
    loaded = load(args.model)
    model, tokenizer = loaded[0], loaded[1]

    print("Running baseline...")
    baseline = generate_baseline(
        model, tokenizer, args.prompt, max_tokens=args.max_tokens, temp=args.temp,
    )

    print(f"Running compressed ({args.kv_bits}-bit)...")
    compressed = generate_with_compressed_cache(
        model, tokenizer, args.prompt,
        kv_bits=args.kv_bits, max_tokens=args.max_tokens, temp=args.temp,
    )

    _print_result(baseline, header="BASELINE")
    _print_result(compressed, header=f"COMPRESSED ({args.kv_bits}-bit)")

    if baseline.cache_bytes > 0 and compressed.cache_bytes > 0:
        ratio = baseline.cache_bytes / compressed.cache_bytes
        savings = (1 - compressed.cache_bytes / baseline.cache_bytes) * 100
        print(f"\nCompression: {ratio:.1f}x ({savings:.0f}% memory saved)")


def _cmd_info(args: argparse.Namespace) -> None:
    """Show model architecture and memory estimates."""
    from mlx_lm import load

    from mlx_turboquant.cache.memory_accounting import estimate_memory
    from mlx_turboquant.integration.mlx_lm_adapter import introspect_model

    print(f"Loading model: {args.model}...")
    model = load(args.model)[0]
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
    from mlx_lm import load

    from mlx_turboquant.bench.latency import benchmark_latency
    from mlx_turboquant.bench.memory import benchmark_memory
    from mlx_turboquant.bench.prompts import BENCHMARK_PROMPTS, QUICK_PROMPTS
    from mlx_turboquant.bench.quality import benchmark_quality
    from mlx_turboquant.bench.report import generate_report
    from mlx_turboquant.integration.mlx_lm_adapter import introspect_model

    print(f"Loading model: {args.model}...")
    loaded = load(args.model)
    model, tokenizer = loaded[0], loaded[1]
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
        model, tokenizer, latency_prompt,
        max_tokens=max_tokens, kv_bits_list=kv_bits_list, runs=runs, warmup=warmup,
    )

    # Quality benchmark
    print(f"Running quality benchmarks ({len(prompts)} prompts)...")
    qual_results = benchmark_quality(
        model, tokenizer, prompts,
        kv_bits_list=kv_bits_list, max_tokens=max_tokens,
    )

    # Generate report
    output_dir = args.output_dir
    generate_report(mem_results, lat_results, qual_results, output_dir, model_name=args.model)
    print(f"\nReport written to {output_dir}/results.json and {output_dir}/BENCHMARKS.md")


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
        "--cache-mode", choices=["compressed", "baseline"], default="compressed",
        help="Cache mode (default: compressed)",
    )
    gen.add_argument("--kv-bits", type=int, default=3, help="Compression bits (default: 3)")
    gen.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    gen.add_argument("--temp", type=float, default=0.0, help="Sampling temperature")
    gen.set_defaults(func=_cmd_generate)

    # compare
    cmp = subparsers.add_parser("compare", help="Compare baseline vs compressed")
    cmp.add_argument("--model", required=True, help="HuggingFace model path")
    cmp.add_argument("--prompt", required=True, help="Input prompt")
    cmp.add_argument("--kv-bits", type=int, default=3, help="Compression bits (default: 3)")
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
        "--suite", choices=["quick", "full"], default="quick",
        help="Benchmark suite (default: quick)",
    )
    bench.add_argument("--kv-bits", type=int, default=3, help="Compression bits for quick suite")
    bench.add_argument("--output-dir", default=".", help="Output directory for reports")
    bench.set_defaults(func=_cmd_bench)

    parsed = parser.parse_args()
    if not hasattr(parsed, "func"):
        parser.print_help()
        sys.exit(1)
    parsed.func(parsed)
