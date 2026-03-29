"""Report generation: JSON + Markdown output for benchmark results."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mlx_turboquant.bench.quality import QualityResult
from mlx_turboquant.cache.memory_accounting import MemoryReport
from mlx_turboquant.integration.generate_wrapper import GenerationResult


def generate_report(
    memory_results: list[MemoryReport],
    latency_results: list[GenerationResult],
    quality_results: list[QualityResult],
    output_dir: str,
    model_name: str = "",
) -> None:
    """Write results.json and BENCHMARKS.md to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON report
    data: dict[str, Any] = {
        "model": model_name,
        "memory": [asdict(r) for r in memory_results],
        "latency": [asdict(r) for r in latency_results],
        "quality": [asdict(r) for r in quality_results],
    }
    with open(out / "results.json", "w") as f:
        json.dump(data, f, indent=2, default=str)

    # Markdown report
    lines = [f"# Benchmark Results — {model_name}", ""]

    # Memory table
    if memory_results:
        lines.append("## Memory")
        lines.append("")
        lines.append("| Seq Len | Bits | Baseline (MB) | Compressed (MB) | Ratio |")
        lines.append("|---------|------|--------------|----------------|-------|")
        for mr in memory_results:
            lines.append(
                f"| {mr.seq_len:,} | {mr.kv_bits} | "
                f"{mr.baseline_bytes / 1024 / 1024:.1f} | "
                f"{mr.compressed_bytes / 1024 / 1024:.1f} | "
                f"{mr.compression_ratio:.1f}x |"
            )
        lines.append("")

    # Latency table
    if latency_results:
        lines.append("## Latency")
        lines.append("")
        lines.append("| Mode | TTFT (ms) | Decode (tok/s) | Tokens | Cache (KB) |")
        lines.append("|------|----------|----------------|--------|-----------|")
        for lr in latency_results:
            lines.append(
                f"| {lr.cache_mode} | {lr.ttft_ms:.1f} | "
                f"{lr.decode_tokens_per_sec:.1f} | "
                f"{lr.tokens_generated} | {lr.cache_bytes / 1024:.1f} |"
            )
        lines.append("")

    # Quality table
    if quality_results:
        lines.append("## Quality")
        lines.append("")
        lines.append("| Prompt | Bits | Token Match | First Diverge | Baseline Toks | Compressed Toks |")
        lines.append("|--------|------|-------------|---------------|--------------|----------------|")
        for qr in quality_results:
            lines.append(
                f"| {qr.prompt_id} | {qr.kv_bits} | "
                f"{qr.token_match_ratio:.1%} | "
                f"{qr.first_divergence_position} | "
                f"{qr.baseline_tokens} | {qr.compressed_tokens} |"
            )
        lines.append("")

    with open(out / "BENCHMARKS.md", "w") as f:
        f.write("\n".join(lines))
