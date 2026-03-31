"""Abstract base class for split point selection policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SplitPolicy(ABC):
    """Base class for split point selection policies.

    All policies operate over the feasible split set K(B).
    """

    def __init__(self, feasible_split_ids: list[int]):
        """Initialize policy with feasible split IDs.

        Args:
            feasible_split_ids: Sorted list of feasible split IDs from K(B).
        """
        if not feasible_split_ids:
            raise ValueError("feasible_split_ids cannot be empty")
        self.feasible_split_ids = sorted(feasible_split_ids)
        self.k_min = self.feasible_split_ids[0]
        self.k_max = self.feasible_split_ids[-1]

    @abstractmethod
    def select(self, context: np.ndarray) -> int:
        """Select a split point from the feasible set.

        Args:
            context: Context vector c_t = [τ̂_t, u_t, Ĥ_t] of shape (3,).

        Returns:
            Selected split_id ∈ K(B).
        """
        ...

    def update(self, context: np.ndarray, split_id: int, reward: float) -> None:
        """Update the policy with observed reward.

        Default implementation is a no-op (for static policies).

        Args:
            context: Context vector used for the decision.
            split_id: Split point that was selected.
            reward: Observed reward r_t = 1[ŷ==y] - λ·τ_t.
        """
        pass

    def reset(self) -> None:
        """Reset policy state. Default is no-op."""
        pass
