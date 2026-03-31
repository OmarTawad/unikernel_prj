"""Ingestion source abstraction.

ReplayFileSource is the required implementation for the current VPS workflow.
MQTTSource is a future placeholder.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger("unisplit.edge.ingestion")


class IngestionSource(ABC):
    """Abstract base for sample ingestion sources."""

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[np.ndarray, int]]:
        """Iterate over samples as (features, label) tuples."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset to the beginning."""
        ...


class ReplayFileSource(IngestionSource):
    """Replays preprocessed dataset samples from disk.

    This is the primary ingestion source for the current VPS stage.
    Reads from preprocessed .npy files sequentially.
    """

    def __init__(
        self,
        features_path: str | Path,
        labels_path: str | Path,
        indices_path: str | Path | None = None,
        shuffle: bool = False,
        seed: int = 42,
        max_samples: int = -1,
    ):
        """Initialize file replay source.

        Args:
            features_path: Path to features.npy.
            labels_path: Path to labels.npy.
            indices_path: Optional path to index file (e.g., test split).
            shuffle: Whether to shuffle replay order.
            seed: Random seed for shuffling.
            max_samples: Max samples to replay (-1 = all).
        """
        self.features = np.load(features_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")

        if indices_path is not None:
            self.indices = np.load(indices_path)
        else:
            self.indices = np.arange(len(self.labels))

        if max_samples > 0:
            self.indices = self.indices[:max_samples]

        self.shuffle = shuffle
        self.seed = seed
        self._order: np.ndarray | None = None
        self.reset()

    def reset(self) -> None:
        """Reset replay order."""
        self._order = np.arange(len(self.indices))
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(self._order)

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[tuple[np.ndarray, int]]:
        for i in self._order:
            real_idx = self.indices[i]
            features = self.features[real_idx].astype(np.float32).copy()
            label = int(self.labels[real_idx])
            yield features, label


class MQTTSource(IngestionSource):
    """MQTT-based ingestion source — FUTURE PLACEHOLDER.

    The paper mentions MQTT-based sensor ingestion, but this is not
    required for the current VPS stage. The interface is defined here
    for future alignment with the paper's architecture.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "MQTT ingestion is not implemented for the current VPS stage. "
            "Use ReplayFileSource instead. "
            "See docs/architecture.md for the MQTT integration roadmap."
        )

    def __iter__(self) -> Iterator[tuple[np.ndarray, int]]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
