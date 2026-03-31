"""Inference backend abstraction.

PyTorchCPUBackend is the only working backend for the current VPS stage.
ONNX backends are future-ready stubs.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
import torch

from unisplit.model.cnn import IoTCNN
from unisplit.model.partition import load_cloud_partition
from unisplit.shared.constants import NUM_CLASSES, NUM_FEATURES, SUPPORTED_SPLIT_IDS


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    @abstractmethod
    def load_partitions(self, partition_dir: str, split_ids: list[int]) -> None:
        """Load cloud partitions for the given split IDs."""
        ...

    @abstractmethod
    def infer(self, activation: np.ndarray, split_id: int) -> tuple[np.ndarray, float]:
        """Run cloud partition inference.

        Args:
            activation: Intermediate activation h_k(x) as numpy array.
            split_id: Which split point this activation comes from.

        Returns:
            Tuple of (logits as numpy array, inference_time_ms).
        """
        ...

    @abstractmethod
    def infer_full(self, input_data: np.ndarray) -> tuple[np.ndarray, float]:
        """Run full model inference (testing path).

        Args:
            input_data: Raw input features.

        Returns:
            Tuple of (logits, inference_time_ms).
        """
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if backend is ready for inference."""
        ...

    @abstractmethod
    def loaded_split_ids(self) -> list[int]:
        """Return list of loaded split IDs."""
        ...


class PyTorchCPUBackend(InferenceBackend):
    """PyTorch CPU inference backend — REQUIRED NOW.

    Loads cloud partitions and runs forward_from on CPU.
    """

    def __init__(self, num_features: int = NUM_FEATURES, num_classes: int = NUM_CLASSES):
        self.num_features = num_features
        self.num_classes = num_classes
        self._models: dict[int, IoTCNN] = {}
        self._full_model: IoTCNN | None = None

    def load_partitions(self, partition_dir: str, split_ids: list[int] | None = None) -> None:
        """Load cloud partitions from directory."""
        if split_ids is None:
            split_ids = list(SUPPORTED_SPLIT_IDS)

        for split_id in split_ids:
            try:
                model = load_cloud_partition(
                    partition_dir, split_id,
                    self.num_features, self.num_classes,
                )
                model.eval()
                self._models[split_id] = model
            except FileNotFoundError:
                continue  # Skip missing partitions

        # Load full model for /infer/full endpoint
        try:
            full_model_path = f"{partition_dir}/cloud_k0/partition.pt"
            import os
            if os.path.exists(full_model_path):
                self._full_model = load_cloud_partition(
                    partition_dir, 0, self.num_features, self.num_classes,
                )
                self._full_model.eval()
        except Exception:
            pass

    def infer(self, activation: np.ndarray, split_id: int) -> tuple[np.ndarray, float]:
        """Run cloud partition inference on CPU."""
        if split_id not in self._models:
            raise ValueError(f"No cloud partition loaded for split_id={split_id}")

        model = self._models[split_id]
        tensor = torch.from_numpy(activation).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)  # Add batch dim

        start = time.perf_counter()
        with torch.no_grad():
            logits = model.forward_from(tensor, split_id)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return logits.numpy(), elapsed_ms

    def infer_full(self, input_data: np.ndarray) -> tuple[np.ndarray, float]:
        """Run full model inference."""
        # Use split_id=0 cloud partition (which runs the full model)
        if 0 not in self._models:
            raise ValueError("No full model partition loaded (split_id=0)")
        return self.infer(input_data, split_id=0)

    def is_ready(self) -> bool:
        return len(self._models) > 0

    def loaded_split_ids(self) -> list[int]:
        return sorted(self._models.keys())


class ONNXCPUBackend(InferenceBackend):
    """ONNX Runtime CPU backend — FUTURE, not implemented."""

    def load_partitions(self, partition_dir: str, split_ids: list[int]) -> None:
        raise NotImplementedError(
            "ONNX CPU backend not yet implemented. "
            "Use backend_type='pytorch_cpu' in config."
        )

    def infer(self, activation: np.ndarray, split_id: int) -> tuple[np.ndarray, float]:
        raise NotImplementedError("ONNX CPU backend not yet implemented.")

    def infer_full(self, input_data: np.ndarray) -> tuple[np.ndarray, float]:
        raise NotImplementedError("ONNX CPU backend not yet implemented.")

    def is_ready(self) -> bool:
        return False

    def loaded_split_ids(self) -> list[int]:
        return []


class ONNXGPUBackend(InferenceBackend):
    """ONNX Runtime GPU backend — FUTURE, not implemented."""

    def load_partitions(self, partition_dir: str, split_ids: list[int]) -> None:
        raise NotImplementedError(
            "ONNX GPU backend not yet implemented. "
            "Use backend_type='pytorch_cpu' in config."
        )

    def infer(self, activation: np.ndarray, split_id: int) -> tuple[np.ndarray, float]:
        raise NotImplementedError("ONNX GPU backend not yet implemented.")

    def infer_full(self, input_data: np.ndarray) -> tuple[np.ndarray, float]:
        raise NotImplementedError("ONNX GPU backend not yet implemented.")

    def is_ready(self) -> bool:
        return False

    def loaded_split_ids(self) -> list[int]:
        return []


def create_backend(backend_type: str = "pytorch_cpu", **kwargs) -> InferenceBackend:
    """Factory for inference backends.

    Args:
        backend_type: One of 'pytorch_cpu', 'onnx_cpu', 'onnx_gpu'.

    Returns:
        InferenceBackend instance.
    """
    backends = {
        "pytorch_cpu": PyTorchCPUBackend,
        "onnx_cpu": ONNXCPUBackend,
        "onnx_gpu": ONNXGPUBackend,
    }
    if backend_type not in backends:
        raise ValueError(
            f"Unknown backend_type '{backend_type}'. "
            f"Available: {list(backends.keys())}"
        )
    return backends[backend_type](**kwargs)
