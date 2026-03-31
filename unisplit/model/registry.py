"""Split point registry — single source of truth for split point metadata.

Defines the 6 supported implementation split points and their properties.
Used consistently across feasibility, partitions, API contracts, and experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from unisplit.shared.constants import SUPPORTED_SPLIT_IDS, SPLIT_NAMES


@dataclass(frozen=True)
class SplitPointEntry:
    """Metadata for a single split point."""
    split_id: int
    name: str
    edge_layer_indices: tuple[int, ...]     # Layers run on edge (1-indexed from paper)
    cloud_layer_indices: tuple[int, ...]    # Layers run on cloud
    output_shape: tuple[int, ...]           # Shape of h_k(x) excluding batch dim

    @property
    def payload_floats(self) -> int:
        """Number of float values in the activation tensor."""
        return prod(self.output_shape) if self.output_shape else 0

    @property
    def payload_float32_bytes(self) -> int:
        """Size of h_k(x) in float32 bytes."""
        return self.payload_floats * 4

    @property
    def payload_int8_bytes(self) -> int:
        """Size of h_k(x) in int8 bytes."""
        return self.payload_floats * 1


# ── Registry ────────────────────────────────────────────────────────────────

SPLIT_REGISTRY: dict[int, SplitPointEntry] = {
    0: SplitPointEntry(
        split_id=0,
        name="input",
        edge_layer_indices=(),
        cloud_layer_indices=(1, 2, 3, 4, 5, 6, 7, 8, 9),
        output_shape=(1, 80),
    ),
    3: SplitPointEntry(
        split_id=3,
        name="after_block1",
        edge_layer_indices=(1, 2, 3),
        cloud_layer_indices=(4, 5, 6, 7, 8, 9),
        output_shape=(32, 78),
    ),
    6: SplitPointEntry(
        split_id=6,
        name="after_block2",
        edge_layer_indices=(1, 2, 3, 4, 5, 6),
        cloud_layer_indices=(7, 8, 9),
        output_shape=(64, 76),
    ),
    7: SplitPointEntry(
        split_id=7,
        name="after_pool",
        edge_layer_indices=(1, 2, 3, 4, 5, 6, 7),
        cloud_layer_indices=(8, 9),
        output_shape=(64,),
    ),
    8: SplitPointEntry(
        split_id=8,
        name="after_fc1",
        edge_layer_indices=(1, 2, 3, 4, 5, 6, 7, 8),
        cloud_layer_indices=(9,),
        output_shape=(128,),
    ),
    9: SplitPointEntry(
        split_id=9,
        name="local_only",
        edge_layer_indices=(1, 2, 3, 4, 5, 6, 7, 8, 9),
        cloud_layer_indices=(),
        output_shape=(34,),
    ),
}


def get_split_info(split_id: int) -> SplitPointEntry:
    """Get split point metadata by ID.

    Raises:
        ValueError: If split_id is not supported.
    """
    if split_id not in SPLIT_REGISTRY:
        raise ValueError(
            f"split_id {split_id} not supported. Must be one of {SUPPORTED_SPLIT_IDS}"
        )
    return SPLIT_REGISTRY[split_id]


def get_all_split_ids() -> list[int]:
    """Return all supported split IDs."""
    return list(SUPPORTED_SPLIT_IDS)


def get_output_shape(split_id: int) -> tuple[int, ...]:
    """Get the output shape for a split point."""
    return get_split_info(split_id).output_shape


def validate_split_id(split_id: int) -> None:
    """Validate that a split ID is supported. Raises ValueError if not."""
    if split_id not in SPLIT_REGISTRY:
        raise ValueError(
            f"split_id {split_id} not supported. Must be one of {SUPPORTED_SPLIT_IDS}"
        )
