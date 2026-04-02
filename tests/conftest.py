"""Shared test fixtures for UniSplit tests."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from unisplit.model.cnn import IoTCNN
from unisplit.edge_native import export_edge_k7_to_c
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


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Path to repository root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def c_runtime_build_dir(repo_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Configure and build edge-native C runtime once per test session."""
    build_dir = tmp_path_factory.mktemp("edge_runtime_build")

    subprocess.run(
        ["cmake", "-S", str(repo_root / "edge_native/runtime"), "-B", str(build_dir)],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "-j"],
        check=True,
    )

    return build_dir


@pytest.fixture
def edge_k7_c_artifacts(tmp_dir: Path) -> Path:
    """Export split-7 C artifacts into a temporary directory."""
    export_edge_k7_to_c(
        partitions_dir="partitions",
        out_dir=tmp_dir,
        model_version="v0.1.0",
        source_checkpoint="checkpoints/best.pt",
        eps=1e-5,
        export_reference=True,
    )
    return tmp_dir
