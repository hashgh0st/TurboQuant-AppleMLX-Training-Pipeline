"""Evaluate experimental compressed-cache profiles on the canonical sample model.

Usage:
    python examples/evaluate_compression_profiles.py
"""

from __future__ import annotations

from mlx_lm import load

from mlx_turboquant.bench.latency import benchmark_latency
from mlx_turboquant.bench.memory import benchmark_memory
from mlx_turboquant.bench.prompts import DIAGNOSTIC_PROMPTS
from mlx_turboquant.bench.quality import benchmark_quality
from mlx_turboquant.bench.report import generate_report
from mlx_turboquant.constants import CANONICAL_SAMPLE_MODEL
from mlx_turboquant.integration.compression_profile import default_experimental_profiles
from mlx_turboquant.integration.mlx_lm_adapter import introspect_model

OUTPUT_DIR = "experimental_profile_results"


def main() -> None:
    print(f"Loading {CANONICAL_SAMPLE_MODEL}...")
    model, tokenizer = load(CANONICAL_SAMPLE_MODEL)  # type: ignore[misc]
    profiles = default_experimental_profiles()
    info = introspect_model(model)

    print("Running memory estimates...")
    memory = benchmark_memory(
        num_layers=info.num_layers,
        num_kv_heads=info.num_kv_heads,
        head_dim=info.head_dim,
        seq_lengths=[1024, 4096],
        profiles=profiles,
    )

    print("Running latency comparisons...")
    latency = benchmark_latency(
        model,
        tokenizer,
        DIAGNOSTIC_PROMPTS["medium_continue"],
        max_tokens=80,
        profiles=profiles,
        runs=1,
        warmup=0,
    )

    print("Running quality comparisons...")
    quality = benchmark_quality(
        model,
        tokenizer,
        DIAGNOSTIC_PROMPTS,
        profiles=profiles,
        max_tokens=80,
    )

    generate_report(memory, latency, quality, OUTPUT_DIR, model_name=CANONICAL_SAMPLE_MODEL)
    print(f"\nResults written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
