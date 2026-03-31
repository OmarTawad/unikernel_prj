"""Benchmark and replay entrypoints for the edge simulator."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from unisplit.edge.simulator import SimulationResults

logger = logging.getLogger("unisplit.edge.benchmark")


def save_results(results: SimulationResults, output_path: str | Path) -> None:
    """Save simulation results to JSON lines file.

    Args:
        results: SimulationResults from a simulation run.
        output_path: Output file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for sample in results.samples:
            f.write(json.dumps(asdict(sample)) + "\n")

    # Write summary
    summary_path = output_path.with_suffix(".summary.json")
    total = max(results.total_samples, 1)
    summary = {
        "total_samples": results.total_samples,
        "accuracy": results.total_correct / total,
        "offload_rate": results.total_offloaded / total,
        "total_correct": results.total_correct,
        "total_offloaded": results.total_offloaded,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Summary saved to {summary_path}")
