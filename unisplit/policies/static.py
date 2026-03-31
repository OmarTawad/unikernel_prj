"""Static split point policies.

These policies ignore the context vector and always return a fixed split point.
Used as baselines in experiments.
"""

from __future__ import annotations

import numpy as np

from unisplit.policies.base import SplitPolicy


class StaticMinPolicy(SplitPolicy):
    """Always selects the shallowest feasible split (most offloading).

    Corresponds to UniSplit-Static-k_min in the paper.
    """

    def select(self, context: np.ndarray) -> int:
        return self.k_min


class StaticMaxPolicy(SplitPolicy):
    """Always selects the deepest feasible split (least offloading).

    Corresponds to UniSplit-Static-k_max in the paper.
    """

    def select(self, context: np.ndarray) -> int:
        return self.k_max


class FixedSplitPolicy(SplitPolicy):
    """Always selects a specific fixed split point.

    Used for controlled experiments at a specific split.
    """

    def __init__(self, feasible_split_ids: list[int], split_id: int = 7):
        super().__init__(feasible_split_ids)
        if split_id not in self.feasible_split_ids:
            raise ValueError(
                f"split_id {split_id} is not feasible. "
                f"Feasible: {self.feasible_split_ids}"
            )
        self._fixed_id = split_id

    def select(self, context: np.ndarray) -> int:
        return self._fixed_id


class LocalOnlyPolicy(SplitPolicy):
    """Always runs the full model locally (no cloud offloading).

    Corresponds to Edge-Only baseline in the paper.
    split_id=9 must be in the feasible set.
    """

    def __init__(self, feasible_split_ids: list[int], **kwargs):
        super().__init__(feasible_split_ids)
        if 9 not in self.feasible_split_ids:
            raise ValueError(
                "Local-only policy requires split_id=9 to be feasible. "
                "The full model may exceed the memory budget."
            )

    def select(self, context: np.ndarray) -> int:
        return 9
