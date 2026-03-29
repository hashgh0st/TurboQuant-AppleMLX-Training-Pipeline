"""Promotion gates: evaluate compression profiles against quality thresholds.

A profile must pass all thresholds to be considered production-ready.
Profiles that fail remain in "research" status — usable for exploration
but not advertised as reliable.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from mlx_turboquant.bench.quality import QualityResult
from mlx_turboquant.integration.generate_wrapper import GenerationResult


@dataclass(frozen=True)
class PromotionThresholds:
    """Quality gates a profile must pass to be production-ready."""

    min_token_match: float = 0.80
    min_first_diverge: int = 10
    max_decode_slowdown: float = 3.0


@dataclass
class ProfileVerdict:
    """Result of evaluating one profile against promotion thresholds."""

    cache_mode: str
    avg_token_match: float
    min_first_diverge: int
    decode_slowdown: float  # ratio vs baseline; 0.0 if baseline latency unavailable
    passes: bool
    failures: list[str] = field(default_factory=list)


DEFAULT_THRESHOLDS = PromotionThresholds()

# Sentinel for "no divergence" (identical output). Must exceed any realistic max_tokens.
_NO_DIVERGE = 10_000


def evaluate_profiles(
    quality_results: list[QualityResult],
    latency_results: list[GenerationResult],
    thresholds: PromotionThresholds = DEFAULT_THRESHOLDS,
) -> list[ProfileVerdict]:
    """Evaluate each compressed profile against promotion thresholds.

    Groups quality results by ``cache_mode``, computes aggregate metrics,
    and checks each against thresholds.  Baseline results are excluded.
    """
    quality_by_mode: dict[str, list[QualityResult]] = defaultdict(list)
    for qr in quality_results:
        quality_by_mode[qr.cache_mode].append(qr)

    # Use median latency (not best-case) to avoid promoting on a lucky run.
    baseline_samples: list[float] = []
    latency_samples_by_mode: dict[str, list[float]] = defaultdict(list)
    for lr in latency_results:
        if lr.cache_mode == "baseline":
            baseline_samples.append(lr.decode_tokens_per_sec)
        else:
            latency_samples_by_mode[lr.cache_mode].append(lr.decode_tokens_per_sec)

    baseline_tok_s = statistics.median(baseline_samples) if baseline_samples else 0.0
    latency_by_mode: dict[str, float] = {
        mode: statistics.median(samples)
        for mode, samples in latency_samples_by_mode.items()
        if samples
    }

    verdicts: list[ProfileVerdict] = []
    # Sorted for deterministic report ordering.
    for mode, qrs in sorted(quality_by_mode.items()):
        avg_match = sum(q.token_match_ratio for q in qrs) / len(qrs)
        worst_diverge = min(
            d if d >= 0 else _NO_DIVERGE for d in (q.first_divergence_position for q in qrs)
        )

        profile_tok_s = latency_by_mode.get(mode, 0.0)
        slowdown = baseline_tok_s / profile_tok_s if profile_tok_s > 0 else 0.0

        failures: list[str] = []
        if avg_match < thresholds.min_token_match:
            failures.append(f"token_match {avg_match:.0%} < {thresholds.min_token_match:.0%}")
        if worst_diverge < thresholds.min_first_diverge:
            failures.append(
                f"first_diverge {worst_diverge} < {thresholds.min_first_diverge}"
            )
        if profile_tok_s > 0 and baseline_tok_s > 0 and slowdown > thresholds.max_decode_slowdown:
            failures.append(
                f"decode_slowdown {slowdown:.1f}x > {thresholds.max_decode_slowdown:.1f}x"
            )

        verdicts.append(
            ProfileVerdict(
                cache_mode=mode,
                avg_token_match=avg_match,
                min_first_diverge=worst_diverge,
                decode_slowdown=slowdown,
                passes=len(failures) == 0,
                failures=failures,
            )
        )

    return verdicts
