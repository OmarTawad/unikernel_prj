"""Edge partition inference runner.

Loads and runs the edge partition of the model (layers 1..k).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from unisplit.model.cnn import IoTCNN
from unisplit.model.partition import load_edge_partition
from unisplit.shared.constants import NUM_CLASSES, NUM_FEATURES, SUPPORTED_SPLIT_IDS


class EdgeRunner:
    """Runs edge partition inference for a given split point.

    Loads edge partitions and executes forward_to(x, split_id).
    """

    def __init__(
        self,
        partition_dir: str,
        num_features: int = NUM_FEATURES,
        num_classes: int = NUM_CLASSES,
    ):
        self.partition_dir = partition_dir
        self.num_features = num_features
        self.num_classes = num_classes
        self._models: dict[int, IoTCNN] = {}

    def load_partitions(self, split_ids: list[int] | None = None) -> None:
        """Load edge partitions for specified split IDs."""
        if split_ids is None:
            split_ids = list(SUPPORTED_SPLIT_IDS)

        for split_id in split_ids:
            try:
                model = load_edge_partition(
                    self.partition_dir, split_id,
                    self.num_features, self.num_classes,
                )
                model.eval()
                self._models[split_id] = model
            except FileNotFoundError:
                continue

    def run(self, x: np.ndarray, split_id: int) -> tuple[np.ndarray, float]:
        """Run edge partition inference.

        Args:
            x: Input features of shape (num_features,) or (batch, num_features).
            split_id: Split point to use.

        Returns:
            Tuple of (activation h_k(x) as numpy array, inference_time_ms).
        """
        if split_id == 0:
            # No edge compute — reshape input and return
            if x.ndim == 1:
                x = x.reshape(1, 1, -1)
            elif x.ndim == 2:
                x = x.reshape(x.shape[0], 1, -1)
            return x.astype(np.float32), 0.0

        if split_id not in self._models:
            raise ValueError(
                f"No edge partition loaded for split_id={split_id}. "
                f"Available: {sorted(self._models.keys())}"
            )

        model = self._models[split_id]
        tensor = torch.from_numpy(x).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        start = time.perf_counter()
        with torch.no_grad():
            activation = model.forward_to(tensor, split_id)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return activation.numpy(), elapsed_ms

    def loaded_split_ids(self) -> list[int]:
        """Return loaded split IDs."""
        return sorted(self._models.keys())
