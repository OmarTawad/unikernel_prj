"""Split point policy system.

Provides static policies and LinUCB for adaptive split selection.
"""

from __future__ import annotations

from unisplit.policies.base import SplitPolicy
from unisplit.policies.static import (
    FixedSplitPolicy,
    LocalOnlyPolicy,
    StaticMaxPolicy,
    StaticMinPolicy,
)
from unisplit.policies.linucb import LinUCBPolicy


def create_policy(
    policy_type: str,
    feasible_split_ids: list[int],
    **kwargs,
) -> SplitPolicy:
    """Factory for creating split policies.

    Args:
        policy_type: One of 'static_kmin', 'static_kmax', 'fixed',
                     'local_only', 'linucb'.
        feasible_split_ids: List of feasible split IDs from K(B).
        **kwargs: Additional policy-specific arguments.

    Returns:
        SplitPolicy instance.
    """
    policies = {
        "static_kmin": StaticMinPolicy,
        "static_kmax": StaticMaxPolicy,
        "fixed": FixedSplitPolicy,
        "local_only": LocalOnlyPolicy,
        "linucb": LinUCBPolicy,
    }

    if policy_type not in policies:
        raise ValueError(
            f"Unknown policy type '{policy_type}'. "
            f"Available: {list(policies.keys())}"
        )

    cls = policies[policy_type]

    if policy_type == "fixed":
        split_id = kwargs.get("split_id", 7)
        return cls(feasible_split_ids=feasible_split_ids, split_id=split_id)
    else:
        return cls(feasible_split_ids=feasible_split_ids, **kwargs)
