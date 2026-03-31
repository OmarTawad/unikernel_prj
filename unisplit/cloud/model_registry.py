"""Cloud partition file discovery and registration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from unisplit.shared.constants import SUPPORTED_SPLIT_IDS
from unisplit.shared.schemas import ModelArtifactMeta

logger = logging.getLogger("unisplit.cloud.registry")


class ModelRegistry:
    """Discovers and tracks available model partitions."""

    def __init__(self, partition_dir: str | Path):
        self.partition_dir = Path(partition_dir)
        self._metadata: dict[int, ModelArtifactMeta] = {}
        self.model_version = ""

    def discover(self) -> list[int]:
        """Scan partition directory and register available cloud partitions.

        Returns:
            List of available split IDs.
        """
        if not self.partition_dir.exists():
            logger.warning(f"Partition directory not found: {self.partition_dir}")
            return []

        found = []
        for split_id in SUPPORTED_SPLIT_IDS:
            meta_path = self.partition_dir / f"cloud_k{split_id}" / "metadata.json"
            pt_path = self.partition_dir / f"cloud_k{split_id}" / "partition.pt"
            if meta_path.exists() and pt_path.exists():
                with open(meta_path) as f:
                    meta = ModelArtifactMeta(**json.load(f))
                self._metadata[split_id] = meta
                found.append(split_id)
                if not self.model_version:
                    self.model_version = meta.model_version

        logger.info(f"Discovered {len(found)} cloud partitions: {found}")
        return found

    def get_metadata(self, split_id: int) -> ModelArtifactMeta | None:
        """Get metadata for a specific split."""
        return self._metadata.get(split_id)

    def get_available_split_ids(self) -> list[int]:
        """Return list of available split IDs."""
        return sorted(self._metadata.keys())

    def is_ready(self) -> bool:
        """Check if at least one partition is loaded."""
        return len(self._metadata) > 0
