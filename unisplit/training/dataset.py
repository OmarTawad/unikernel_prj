"""PyTorch Dataset for CIC-IoT2023 preprocessed data.

Uses numpy memmap for zero-copy disk reads.
All feature normalization is applied on-the-fly using precomputed statistics.
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

    Uses memory-mapped numpy files for efficient disk access.
    Normalization is applied per-sample on read using precomputed
    training-set statistics.

    Expects:
        - processed_dir/features.npy  — shape (N, 80) float32 (memmap-safe)
        - processed_dir/labels.npy    — shape (N,) int64 (memmap-safe)
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
        self.processed_dir = Path(processed_dir)
        self.metadata_dir = Path(metadata_dir)
        self.normalize = normalize

        # Load data as memory-mapped files — no RAM spike
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
                # Use precomputed range if available, else compute
                if "range" in stats:
                    self.norm_range = np.array(stats["range"], dtype=np.float32)
                else:
                    norm_max = np.array(stats["max"], dtype=np.float32)
                    self.norm_range = norm_max - self.norm_min
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

    def get_class_weights(self, batch_size: int = 500_000) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced data.

        Reads labels in batches to avoid materializing the full label array
        for the split in RAM.

        Returns:
            Tensor of shape (num_classes,) with per-class weights.
        """
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        n = len(self.indices)

        # Process indices in batches
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx_batch = self.indices[start:end]
            labels_batch = self.labels[idx_batch]
            for lbl in labels_batch:
                if 0 <= lbl < NUM_CLASSES:
                    counts[lbl] += 1

        # Avoid division by zero for absent classes
        counts_float = counts.astype(np.float64)
        counts_float[counts_float == 0] = 1.0

        weights = 1.0 / counts_float
        weights = weights / weights.sum() * NUM_CLASSES  # Normalize so mean ≈ 1.0
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
