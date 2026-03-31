"""Tests for context vector extraction."""

import numpy as np

from unisplit.edge.context import ContextExtractor
from unisplit.model.cnn import IoTCNN


class TestContext:
    def test_context_vector_shape(self):
        extractor = ContextExtractor()
        ctx = extractor.get_context_vector()
        assert ctx.shape == (3,)

    def test_initial_rtt_estimate(self):
        extractor = ContextExtractor(initial_rtt_ms=20.0)
        assert extractor.get_estimated_rtt() == 20.0

    def test_rtt_update(self):
        extractor = ContextExtractor(rtt_ewma_alpha=0.5, initial_rtt_ms=20.0)
        extractor.update_rtt_estimate(40.0)
        # EWMA: 0.5 * 40 + 0.5 * 20 = 30
        assert abs(extractor.get_estimated_rtt() - 30.0) < 0.1

    def test_cpu_utilization_range(self):
        extractor = ContextExtractor()
        cpu = extractor.get_cpu_utilization()
        assert 0.0 <= cpu <= 1.0

    def test_uncertainty_max_at_k0(self):
        extractor = ContextExtractor()
        h = extractor.get_uncertainty(None, None, k_min=0)
        assert h > 0  # Should be log(34)

    def test_context_with_model(self):
        model = IoTCNN()
        model.eval()
        x = np.random.randn(80).astype(np.float32)
        extractor = ContextExtractor()
        ctx = extractor.get_context_vector(model=model, x=x, k_min=3)
        assert ctx.shape == (3,)
        assert all(np.isfinite(ctx))
