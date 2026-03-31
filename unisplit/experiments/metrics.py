"""Experiment-level metrics computation.

Computes offload rate, wasted offloads, and latency distributions
from simulation results.
"""

from __future__ import annotations

import numpy as np

from unisplit.edge.simulator import SimulationResults


def compute_offload_rate(results: SimulationResults) -> float:
    """Fraction of samples sent to cloud."""
    total = max(results.total_samples, 1)
    return results.total_offloaded / total


def compute_accuracy(results: SimulationResults) -> float:
    """Overall accuracy."""
    total = max(results.total_samples, 1)
    return results.total_correct / total


def compute_wasted_offloads(
    results: SimulationResults,
    local_predictions: dict[int, int] | None = None,
) -> float:
    """Fraction of cloud offloads where edge-only would have been correct.

    If local_predictions is not provided, this is estimated from samples
    where the prediction was correct AND the sample was offloaded (these
    may not have needed cloud inference).

    Args:
        results: Simulation results.
        local_predictions: Optional dict mapping sample_idx → local prediction.

    Returns:
        Wasted offload fraction.
    """
    if not results.samples:
        return 0.0

    offloaded = [s for s in results.samples if s.offloaded]
    if not offloaded:
        return 0.0

    if local_predictions is not None:
        wasted = sum(
            1 for s in offloaded
            if local_predictions.get(s.sample_idx) == s.true_label
        )
    else:
        # Approximate: count offloaded samples that were correctly predicted
        # (These could potentially have been predicted locally too)
        wasted = sum(1 for s in offloaded if s.correct)

    return wasted / len(offloaded)


def compute_latency_stats(results: SimulationResults) -> dict:
    """Compute latency distribution statistics."""
    if not results.samples:
        return {"p50": 0, "p95": 0, "p99": 0, "mean": 0, "std": 0}

    latencies = np.array([s.total_latency_ms for s in results.samples])
    return {
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
        "mean": float(np.mean(latencies)),
        "std": float(np.std(latencies)),
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies)),
    }


def compute_split_distribution(results: SimulationResults) -> dict[int, float]:
    """Compute distribution of selected split points."""
    if not results.samples:
        return {}

    total = len(results.samples)
    counts: dict[int, int] = {}
    for s in results.samples:
        counts[s.split_id] = counts.get(s.split_id, 0) + 1

    return {k: v / total for k, v in sorted(counts.items())}


def compute_full_report(results: SimulationResults) -> dict:
    """Compute a full experiment metrics report."""
    return {
        "accuracy": compute_accuracy(results),
        "offload_rate": compute_offload_rate(results),
        "latency": compute_latency_stats(results),
        "split_distribution": compute_split_distribution(results),
        "total_samples": results.total_samples,
        "total_correct": results.total_correct,
        "total_offloaded": results.total_offloaded,
    }
