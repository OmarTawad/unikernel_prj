"""Tests for feasibility calculator."""

from unisplit.model.cnn import IoTCNN
from unisplit.profiler.feasibility import FeasibilityCalculator
from unisplit.profiler.memory import ModelMemoryProfiler
from unisplit.shared.constants import SUPPORTED_SPLIT_IDS


class TestFeasibility:
    def _make_calculator(self, budget_mb: float = 24.0, overhead_mb: float = 2.0):
        model = IoTCNN()
        profiler = ModelMemoryProfiler(model)
        return FeasibilityCalculator(
            profiler,
            budget_bytes=int(budget_mb * 1024 * 1024),
            overhead_bytes=int(overhead_mb * 1024 * 1024),
        )

    def test_all_feasible_at_24mb(self):
        """At 24MB budget, all splits should be feasible (model is tiny)."""
        calc = self._make_calculator(24.0)
        feasible = calc.get_feasible_split_ids()
        assert feasible == SUPPORTED_SPLIT_IDS

    def test_proposition_1(self):
        """Proposition 1: K(B) non-empty if B >= mem(0)."""
        calc = self._make_calculator(24.0)
        assert calc.validate_proposition_1()

    def test_proposition_1_at_tight_budget(self):
        """Even at very tight budgets, check proposition holds."""
        for budget_mb in [0.5, 1.0, 2.5, 4.0, 8.0, 12.0]:
            calc = self._make_calculator(budget_mb)
            mem_0 = calc.compute_split_memory(0)
            budget = int(budget_mb * 1024 * 1024)
            if budget >= mem_0:
                assert calc.validate_proposition_1()

    def test_feasible_set_shrinks_with_budget(self):
        """Fewer splits should be feasible at lower budgets."""
        budgets = [0.01, 0.1, 1.0, 24.0]
        sizes = []
        for b in budgets:
            calc = self._make_calculator(b, 0.0)  # No overhead for this test
            sizes.append(len(calc.get_feasible_split_ids()))
        # Should be non-decreasing
        for i in range(len(sizes) - 1):
            assert sizes[i] <= sizes[i + 1]

    def test_compute_report(self):
        """Verify report structure."""
        calc = self._make_calculator(24.0)
        report = calc.compute_report()
        assert len(report.split_profiles) == len(SUPPORTED_SPLIT_IDS)
        assert len(report.feasible_split_ids) > 0
        assert report.budget_bytes > 0
        assert report.total_model_parameters > 0

    def test_mem_k_formula(self):
        """mem(k) = W_k + A_k + δ."""
        calc = self._make_calculator()
        for k in SUPPORTED_SPLIT_IDS:
            w = calc.profiler.get_cumulative_weight_bytes(k)
            a = calc.profiler.get_peak_activation_bytes(k)
            d = calc.overhead_bytes
            expected = w + a + d
            actual = calc.compute_split_memory(k)
            assert actual == expected, f"mem({k}): expected {expected}, got {actual}"
