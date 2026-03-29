"""Run a quick benchmark and generate a report.

Usage:
    python examples/run_benchmarks.py

Equivalent to:
    mlx-tq bench --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --suite quick --output-dir benchmark_results

Requires: mlx-community/Qwen2.5-0.5B-Instruct-4bit (downloaded automatically)
"""

import subprocess
import sys

OUTPUT_DIR = "benchmark_results"


def main() -> None:
    cmd = [
        sys.executable, "-m", "mlx_turboquant",
        "bench",
        "--model", "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "--suite", "quick",
        "--output-dir", OUTPUT_DIR,
    ]
    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)

    print(f"\nResults written to {OUTPUT_DIR}/")
    print(f"  {OUTPUT_DIR}/results.json   — machine-readable")
    print(f"  {OUTPUT_DIR}/BENCHMARKS.md  — human-readable tables")


if __name__ == "__main__":
    main()
