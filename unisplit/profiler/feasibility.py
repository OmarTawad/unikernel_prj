"""Feasibility calculator — computes the feasible split set K(B).

Implements the paper's Definition 2 (§3.2):
    K(B) = {k ∈ {1,...,L} : mem(k) ≤ B}
where:
    mem(k) = W_k + A_k + δ

Also validates Proposition 1: K(B) is non-empty for B ≥ mem(1).
"""

from __future__ import annotations

from unisplit.model.registry import SPLIT_REGISTRY, get_split_info
from unisplit.profiler.memory import ModelMemoryProfiler
from unisplit.shared.constants import (
    DEFAULT_MEMORY_BUDGET_BYTES,
    DEFAULT_OVERHEAD_BYTES,
    SUPPORTED_SPLIT_IDS,
)
from unisplit.shared.schemas import FeasibilityReport, SplitMemoryProfile


class FeasibilityCalculator:
    """Computes the feasible split set K(B) for a given memory budget.

    This is a core architectural component implementing the paper's
    memory-constrained split inference framework.
    """

    def __init__(
        self,
        profiler: ModelMemoryProfiler,
        budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
        overhead_bytes: int = DEFAULT_OVERHEAD_BYTES,
    ):
        """Initialize the feasibility calculator.

        Args:
            profiler: Memory profiler with layer-level measurements.
            budget_bytes: Edge memory budget B in bytes.
            overhead_bytes: OS/runtime overhead δ in bytes.
        """
        self.profiler = profiler
        self.budget_bytes = budget_bytes
        self.overhead_bytes = overhead_bytes

    def compute_split_memory(self, split_id: int) -> int:
        """Compute mem(k) = W_k + A_k + δ for a split point.

        Args:
            split_id: One of SUPPORTED_SPLIT_IDS.

        Returns:
            Total edge memory in bytes.
        """
        w_k = self.profiler.get_cumulative_weight_bytes(split_id)
        a_k = self.profiler.get_peak_activation_bytes(split_id)
        return w_k + a_k + self.overhead_bytes

    def is_feasible(self, split_id: int) -> bool:
        """Check if a split point is feasible: mem(k) ≤ B."""
        return self.compute_split_memory(split_id) <= self.budget_bytes

    def get_feasible_split_ids(self) -> list[int]:
        """Compute K(B) — the set of feasible split points.

        Returns:
            Sorted list of feasible split IDs.
        """
        return [k for k in SUPPORTED_SPLIT_IDS if self.is_feasible(k)]

    def compute_report(self) -> FeasibilityReport:
        """Compute a full feasibility report for all split points.

        Returns:
            FeasibilityReport with per-split-point profiles and the feasible set.
        """
        split_profiles = []

        for split_id in SUPPORTED_SPLIT_IDS:
            info = get_split_info(split_id)
            w_k = self.profiler.get_cumulative_weight_bytes(split_id)
            a_k = self.profiler.get_peak_activation_bytes(split_id)
            mem_k = w_k + a_k + self.overhead_bytes

            profile = SplitMemoryProfile(
                split_id=split_id,
                split_name=info.name,
                cumulative_weight_bytes=w_k,
                peak_activation_bytes=a_k,
                runtime_overhead_bytes=self.overhead_bytes,
                total_edge_memory_bytes=mem_k,
                communication_payload_float32_bytes=info.payload_float32_bytes,
                communication_payload_int8_bytes=info.payload_int8_bytes,
                output_shape=list(info.output_shape),
                feasible=mem_k <= self.budget_bytes,
            )
            split_profiles.append(profile)

        feasible_ids = [p.split_id for p in split_profiles if p.feasible]

        return FeasibilityReport(
            budget_bytes=self.budget_bytes,
            overhead_bytes=self.overhead_bytes,
            layer_profiles=self.profiler.get_layer_memory_profiles(),
            split_profiles=split_profiles,
            feasible_split_ids=feasible_ids,
            total_model_parameters=self.profiler.get_total_model_parameters(),
            total_model_weight_bytes=self.profiler.get_total_model_weight_bytes(),
        )

    def validate_proposition_1(self) -> bool:
        """Validate Proposition 1: K(B) is non-empty if B ≥ mem(0).

        The paper proves that for B ≥ mem(1), K(B) is non-empty.
        In our implementation, split_id=0 has mem(0) = 0 + 0 + δ = δ,
        so K(B) is non-empty for B ≥ δ.

        Returns:
            True if the proposition holds.
        """
        mem_0 = self.compute_split_memory(0)
        if self.budget_bytes >= mem_0:
            feasible = self.get_feasible_split_ids()
            return len(feasible) > 0
        return True  # Budget too small for any split — proposition doesn't apply

    def print_report(self) -> None:
        """Print a human-readable feasibility report to stdout."""
        report = self.compute_report()

        print(f"\n{'='*70}")
        print(f"  UniSplit Memory Feasibility Report")
        print(f"{'='*70}")
        print(f"  Budget (B):                {report.budget_bytes:>12,} bytes ({report.budget_bytes / 1024 / 1024:.1f} MB)")
        print(f"  Overhead (δ):              {report.overhead_bytes:>12,} bytes ({report.overhead_bytes / 1024 / 1024:.1f} MB)")
        print(f"  Total model parameters:    {report.total_model_parameters:>12,}")
        print(f"  Total model weight bytes:  {report.total_model_weight_bytes:>12,} bytes ({report.total_model_weight_bytes / 1024:.1f} KB)")
        print(f"{'='*70}")
        print()

        # Layer profiles
        print(f"  {'Layer':<12} {'Weights':>12} {'Activations':>12}")
        print(f"  {'-'*12} {'-'*12} {'-'*12}")
        for lp in report.layer_profiles:
            print(f"  {lp.layer_name:<12} {lp.weight_bytes:>12,} {lp.activation_bytes:>12,}")
        print()

        # Split profiles
        header = f"  {'Split':>5} {'Name':<16} {'W_k':>10} {'A_k':>10} {'δ':>10} {'mem(k)':>12} {'Payload(f32)':>14} {'Payload(i8)':>12} {'Feasible':>8}"
        print(header)
        print(f"  {'-'*len(header)}")
        for sp in report.split_profiles:
            feasible_str = "  ✓" if sp.feasible else "  ✗"
            print(
                f"  {sp.split_id:>5} {sp.split_name:<16} "
                f"{sp.cumulative_weight_bytes:>10,} {sp.peak_activation_bytes:>10,} "
                f"{sp.runtime_overhead_bytes:>10,} {sp.total_edge_memory_bytes:>12,} "
                f"{sp.communication_payload_float32_bytes:>14,} "
                f"{sp.communication_payload_int8_bytes:>12,} "
                f"{feasible_str}"
            )
        print()
        print(f"  Feasible split set K(B) = {report.feasible_split_ids}")
        print(f"  Proposition 1 valid:      {self.validate_proposition_1()}")
        print(f"{'='*70}\n")
