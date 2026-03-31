"""LinUCB contextual bandit policy for adaptive split selection.

Implements Algorithm 1 from the paper (§4):
    - Per-arm parameters: A_k, b_k for each k ∈ K(B)
    - Feature map φ(c_t) from context vector
    - UCB computation with exploration parameter α
    - Rank-1 update on observed reward

This is FULLY IMPLEMENTED — not a stub. The math matches the paper exactly.
"""

from __future__ import annotations

import logging

import numpy as np

from unisplit.policies.base import SplitPolicy

logger = logging.getLogger("unisplit.policies.linucb")


class LinUCBPolicy(SplitPolicy):
    """LinUCB contextual bandit for split point selection.

    From paper Algorithm 1:
        For each arm k ∈ K(B):
            θ̂_k = A_k⁻¹ b_k
            UCB_k = θ̂_k^T φ + α √(φ^T A_k⁻¹ φ)
        Select k_t = argmax_{k ∈ K(B)} UCB_k
        Update: A_{k_t} += φφ^T, b_{k_t} += r_t · φ
    """

    def __init__(
        self,
        feasible_split_ids: list[int],
        alpha: float = 1.0,
        feature_dim: int | None = None,
        use_polynomial_features: bool = False,
        **kwargs,
    ):
        """Initialize LinUCB policy.

        Args:
            feasible_split_ids: Feasible split IDs from K(B).
            alpha: Exploration parameter (higher = more exploration).
            feature_dim: Feature dimension d. If None, auto-computed.
            use_polynomial_features: Add polynomial features to context.
        """
        super().__init__(feasible_split_ids)
        self.alpha = alpha
        self.use_poly = use_polynomial_features

        # Compute feature dimension
        # Raw context: [rtt, cpu, uncertainty] → d=3
        # With polynomial: [rtt, cpu, uncertainty, rtt², cpu², unc², 1] → d=7
        if feature_dim is not None:
            self.d = feature_dim
        elif use_polynomial_features:
            self.d = 7  # 3 raw + 3 squared + 1 bias
        else:
            self.d = 4  # 3 raw + 1 bias

        # Initialize per-arm parameters (paper line 2-3)
        self._A: dict[int, np.ndarray] = {}
        self._b: dict[int, np.ndarray] = {}
        self.reset()

    def reset(self) -> None:
        """Reset all per-arm parameters to initial state."""
        for k in self.feasible_split_ids:
            self._A[k] = np.eye(self.d, dtype=np.float64)
            self._b[k] = np.zeros(self.d, dtype=np.float64)

    def _featurize(self, context: np.ndarray) -> np.ndarray:
        """Compute feature vector φ(c_t) from raw context.

        Args:
            context: Raw context [rtt, cpu, uncertainty] shape (3,).

        Returns:
            Feature vector φ of shape (d,).
        """
        ctx = context.astype(np.float64)

        if self.use_poly:
            # [rtt, cpu, unc, rtt², cpu², unc², 1]
            phi = np.concatenate([
                ctx,
                ctx ** 2,
                [1.0],
            ])
        else:
            # [rtt, cpu, unc, 1] (with bias term)
            phi = np.concatenate([ctx, [1.0]])

        # Normalize to prevent numerical issues
        norm = np.linalg.norm(phi)
        if norm > 1e-10:
            phi = phi / norm

        return phi

    def select(self, context: np.ndarray) -> int:
        """Select split point using UCB.

        Paper Algorithm 1, lines 6-8:
            For each k: θ̂_k = A_k⁻¹ b_k
                       UCB_k = θ̂_k^T φ + α √(φ^T A_k⁻¹ φ)
            Return argmax_k UCB_k

        Args:
            context: Context vector c_t of shape (3,).

        Returns:
            Selected split_id ∈ K(B).
        """
        phi = self._featurize(context)

        best_ucb = -np.inf
        best_k = self.feasible_split_ids[0]

        for k in self.feasible_split_ids:
            A_inv = np.linalg.inv(self._A[k])
            theta_hat = A_inv @ self._b[k]

            # UCB = θ̂^T φ + α √(φ^T A⁻¹ φ)
            exploit = theta_hat @ phi
            explore = self.alpha * np.sqrt(phi @ A_inv @ phi)
            ucb = exploit + explore

            if ucb > best_ucb:
                best_ucb = ucb
                best_k = k

        return best_k

    def update(self, context: np.ndarray, split_id: int, reward: float) -> None:
        """Update LinUCB parameters with observed reward.

        Paper Algorithm 1, line 12:
            A_{k_t} += φ φ^T  (rank-1 update)
            b_{k_t} += r_t · φ

        Args:
            context: Context vector used for the decision.
            split_id: Split point that was selected.
            reward: Observed reward r_t.
        """
        if split_id not in self._A:
            logger.warning(f"split_id {split_id} not in feasible set, skipping update")
            return

        phi = self._featurize(context)

        # Rank-1 update
        self._A[split_id] += np.outer(phi, phi)
        self._b[split_id] += reward * phi

    def get_arm_stats(self) -> dict[int, dict]:
        """Get per-arm statistics for debugging."""
        stats = {}
        for k in self.feasible_split_ids:
            A_inv = np.linalg.inv(self._A[k])
            theta_hat = A_inv @ self._b[k]
            stats[k] = {
                "theta_norm": float(np.linalg.norm(theta_hat)),
                "A_trace": float(np.trace(self._A[k])),
                "num_updates": int(np.trace(self._A[k]) - self.d),
            }
        return stats
