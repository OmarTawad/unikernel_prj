"""Shared test fixtures for UniSplit tests."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from unisplit.model.cnn import IoTCNN
from unisplit.shared.config import UniSplitConfig, load_config
from unisplit.shared.constants import NUM_CLASSES, NUM_FEATURES, SUPPORTED_SPLIT_IDS


@pytest.fixture
def model():
    """Create a fresh IoTCNN model."""
    m = IoTCNN(num_features=NUM_FEATURES, num_classes=NUM_CLASSES)
    m.eval()
    return m


@pytest.fixture
def sample_input():
    """Create a single sample input tensor."""
    return torch.randn(1, NUM_FEATURES)


@pytest.fixture
def batch_input():
    """Create a batch of input tensors."""
    return torch.randn(8, NUM_FEATURES)


@pytest.fixture
def sample_numpy():
    """Create a single sample as numpy array."""
    return np.random.randn(NUM_FEATURES).astype(np.float32)


@pytest.fixture
def default_config():
    """Load default config."""
    return UniSplitConfig()


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
