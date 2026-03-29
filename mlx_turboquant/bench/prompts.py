"""Benchmark prompt sets for quality and latency evaluation."""

from __future__ import annotations

BENCHMARK_PROMPTS: dict[str, str] = {
    "short_qa": "What is the capital of France?",
    "medium_instruct": (
        "Explain how the attention mechanism works in transformer neural networks. "
        "Include details about queries, keys, values, and the softmax operation."
    ),
    "medium_continue": (
        "Continue this explanation in a concise technical style: "
        "KV-cache compression trades extra compute during decode for lower memory use. "
        "The most important implementation detail is"
    ),
    "code_gen": "Write a Python function that implements binary search on a sorted list.",
    "reasoning": (
        "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? "
        "Think step by step."
    ),
}

# Subset for quick benchmarks
QUICK_PROMPTS: dict[str, str] = {
    "short_qa": BENCHMARK_PROMPTS["short_qa"],
    "code_gen": BENCHMARK_PROMPTS["code_gen"],
}

DIAGNOSTIC_PROMPTS: dict[str, str] = {
    "short_qa": BENCHMARK_PROMPTS["short_qa"],
    "code_gen": BENCHMARK_PROMPTS["code_gen"],
    "reasoning": BENCHMARK_PROMPTS["reasoning"],
    "medium_continue": BENCHMARK_PROMPTS["medium_continue"],
}
