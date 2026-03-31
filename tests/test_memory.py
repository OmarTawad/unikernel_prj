"""Tests for memory profiler."""

from unisplit.model.cnn import IoTCNN
from unisplit.profiler.memory import ModelMemoryProfiler
from unisplit.shared.constants import SUPPORTED_SPLIT_IDS


class TestMemory:
    def test_profiler_returns_values(self):
        model = IoTCNN()
        profiler = ModelMemoryProfiler(model)
        profiles = profiler.profile()
        assert len(profiles) > 0

    def test_cumulative_weights_monotonic(self):
        """W_k should be non-decreasing as k increases."""
        profiler = ModelMemoryProfiler()
        prev = 0
        for k in SUPPORTED_SPLIT_IDS:
            w_k = profiler.get_cumulative_weight_bytes(k)
            assert w_k >= prev, f"W_{k}={w_k} < W_prev={prev}"
            prev = w_k

    def test_weight_bytes_zero_at_k0(self):
        """Split 0 (no edge compute) should have zero weights."""
        profiler = ModelMemoryProfiler()
        assert profiler.get_cumulative_weight_bytes(0) == 0

    def test_peak_activation_positive(self):
        """Peak activation should be positive for k > 0."""
        profiler = ModelMemoryProfiler()
        for k in [3, 6, 7, 8, 9]:
            a_k = profiler.get_peak_activation_bytes(k)
            assert a_k > 0, f"A_{k} should be positive"

    def test_communication_payload_size(self):
        """Payload sizes should match expected values."""
        profiler = ModelMemoryProfiler()
        # k=7 (after pool): 64 floats × 4 bytes = 256 bytes
        p7 = profiler.get_communication_payload_bytes(7, "float32")
        assert p7 == 64 * 4
        # k=7 int8: 64 bytes
        p7_i8 = profiler.get_communication_payload_bytes(7, "int8")
        assert p7_i8 == 64

    def test_total_model_weight_bytes(self):
        """Total weight bytes should be reasonable for the model."""
        profiler = ModelMemoryProfiler()
        total = profiler.get_total_model_weight_bytes()
        # Compact model: ~77 KB weight bytes
        assert 50_000 < total < 500_000
