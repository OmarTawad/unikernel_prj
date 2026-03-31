"""PyTorch Dataset for CIC-IoT2023 preprocessed data.

Loads preprocessed numpy arrays and applies feature normalization
using precomputed statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from unisplit.shared.constants import NUM_CLASSES, NUM_FEATURES


class CICIoT2023Dataset(Dataset):
    """PyTorch Dataset for preprocessed CIC-IoT2023 data.

    Expects:
        - processed_dir/features.npy  — shape (N, 80) float32
        - processed_dir/labels.npy    — shape (N,) int64
        - metadata_dir/norm_stats.json — normalization statistics
        - split_file — .npy file with integer indices for this split
    """

    def __init__(
        self,
        processed_dir: str | Path,
        metadata_dir: str | Path,
        split_file: str | Path | None = None,
        normalize: bool = True,
    ):
        """Initialize dataset.

        Args:
            processed_dir: Directory containing features.npy and labels.npy.
            metadata_dir: Directory containing norm_stats.json.
            split_file: Path to .npy index file (train/val/test indices).
                        If None, use all samples.
            normalize: Whether to apply feature normalization.
        """
        self.processed_dir = Path(processed_dir)
        self.metadata_dir = Path(metadata_dir)
        self.normalize = normalize

        # Load data
        features_path = self.processed_dir / "features.npy"
        labels_path = self.processed_dir / "labels.npy"

        if not features_path.exists():
            raise FileNotFoundError(
                f"Features file not found: {features_path}\n"
                "Run `make preprocess-data` first."
            )
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found: {labels_path}\n"
                "Run `make preprocess-data` first."
            )

        self.features = np.load(features_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")

        # Apply split indices if provided
        if split_file is not None:
            split_path = Path(split_file)
            if not split_path.exists():
                raise FileNotFoundError(f"Split file not found: {split_path}")
            self.indices = np.load(split_path)
        else:
            self.indices = np.arange(len(self.labels))

        # Load normalization stats
        self.norm_min = None
        self.norm_range = None
        if normalize:
            stats_path = self.metadata_dir / "norm_stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    stats = json.load(f)
                self.norm_min = np.array(stats["min"], dtype=np.float32)
                self.norm_max = np.array(stats["max"], dtype=np.float32)
                self.norm_range = self.norm_max - self.norm_min
                # Avoid division by zero
                self.norm_range[self.norm_range < 1e-10] = 1.0

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        features = self.features[real_idx].astype(np.float32).copy()
        label = int(self.labels[real_idx])

        # Normalize to [0, 1] using training set statistics
        if self.normalize and self.norm_min is not None:
            features = (features - self.norm_min) / self.norm_range
            features = np.clip(features, 0.0, 1.0)

        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced data.

        Returns:
            Tensor of shape (num_classes,) with per-class weights.
        """
        labels_subset = self.labels[self.indices]
        counts = np.bincount(labels_subset, minlength=NUM_CLASSES).astype(np.float64)
        # Avoid division by zero for absent classes
        counts[counts == 0] = 1.0
        weights = 1.0 / counts
        weights = weights / weights.sum() * NUM_CLASSES  # Normalize so mean = 1.0
        return torch.tensor(weights, dtype=torch.float32)


class SyntheticDataset(Dataset):
    """Synthetic dataset for dry-run testing.

    Generates random features and labels matching the expected schema.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_features: int = NUM_FEATURES,
        num_classes: int = NUM_CLASSES,
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed)
        self.features = rng.randn(num_samples, num_features).astype(np.float32)
        self.labels = rng.randint(0, num_classes, size=num_samples)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(int(self.labels[idx]), dtype=torch.long),
        )

    def get_class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.labels, minlength=NUM_CLASSES).astype(np.float64)
        counts[counts == 0] = 1.0
        weights = 1.0 / counts
        weights = weights / weights.sum() * NUM_CLASSES
        return torch.tensor(weights, dtype=torch.float32)
