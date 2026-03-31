"""Tests for the IoTCNN model."""

import torch

from unisplit.model.cnn import IoTCNN
from unisplit.shared.constants import NUM_CLASSES, NUM_FEATURES


class TestModel:
    def test_forward_shape(self, model, sample_input):
        output = model(sample_input)
        assert output.shape == (1, NUM_CLASSES)

    def test_forward_batch(self, model, batch_input):
        output = model(batch_input)
        assert output.shape == (8, NUM_CLASSES)

    def test_parameter_count(self, model):
        count = model.count_parameters()
        # Paper says ~98,432 parameters
        assert 50_000 < count < 200_000, f"Unexpected param count: {count}"

    def test_forward_to_shapes(self, model, sample_input):
        """Verify output shapes at each split point."""
        expected = {
            0: (1, 1, 80),
            3: (1, 32, 78),
            6: (1, 64, 76),
            7: (1, 64),
            8: (1, 128),
            9: (1, 34),
        }
        for split_id, expected_shape in expected.items():
            h = model.forward_to(sample_input, split_id)
            assert h.shape == expected_shape, (
                f"Split {split_id}: expected {expected_shape}, got {h.shape}"
            )

    def test_invalid_split_id(self, model, sample_input):
        import pytest
        with pytest.raises(ValueError):
            model.forward_to(sample_input, 5)
        with pytest.raises(ValueError):
            model.forward_from(sample_input, 5)
