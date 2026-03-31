"""CLI for memory profiling and feasibility computation."""

from __future__ import annotations

import argparse
import sys

from unisplit.model.cnn import IoTCNN
from unisplit.profiler.feasibility import FeasibilityCalculator
from unisplit.profiler.memory import ModelMemoryProfiler
from unisplit.profiler.profile_store import save_profile
from unisplit.shared.constants import DEFAULT_MEMORY_BUDGET_BYTES, DEFAULT_OVERHEAD_BYTES


def _parse_size(size_str: str) -> int:
    """Parse a human-readable size string like '24M' or '8MB' to bytes."""
    size_str = size_str.strip().upper()
    if size_str.endswith("MB"):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith("M"):
        return int(float(size_str[:-1]) * 1024 * 1024)
    elif size_str.endswith("KB"):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith("K"):
        return int(float(size_str[:-1]) * 1024)
    elif size_str.endswith("B"):
        return int(size_str[:-1])
    else:
        return int(size_str)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile model memory and compute feasible split set K(B)"
    )
    parser.add_argument(
        "--budget", type=str, default="24M",
        help="Edge memory budget (e.g., '24M', '8MB', '16777216')"
    )
    parser.add_argument(
        "--overhead", type=str, default="2M",
        help="OS/runtime overhead δ (e.g., '2M')"
    )
    parser.add_argument(
        "--output", type=str, default="profiles/default.json",
        help="Output path for feasibility report JSON"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress human-readable output"
    )
    args = parser.parse_args()

    budget_bytes = _parse_size(args.budget)
    overhead_bytes = _parse_size(args.overhead)

    # Create model and profiler
    model = IoTCNN()
    profiler = ModelMemoryProfiler(model)

    # Compute feasibility
    calculator = FeasibilityCalculator(
        profiler=profiler,
        budget_bytes=budget_bytes,
        overhead_bytes=overhead_bytes,
    )

    if not args.quiet:
        calculator.print_report()

    # Save report
    report = calculator.compute_report()
    output_path = save_profile(report, args.output)
    print(f"✓ Feasibility report saved to {output_path}")

    # Exit with error if no feasible splits
    if not report.feasible_split_ids:
        print("✗ WARNING: No feasible split points for the given budget!")
        sys.exit(1)


if __name__ == "__main__":
    main()
