"""Tests for split point policies."""

import numpy as np
import pytest

from unisplit.policies import create_policy
from unisplit.policies.linucb import LinUCBPolicy
from unisplit.policies.static import (
    FixedSplitPolicy,
    LocalOnlyPolicy,
    StaticMaxPolicy,
    StaticMinPolicy,
)
from unisplit.shared.constants import SUPPORTED_SPLIT_IDS


class TestStaticPolicies:
    def test_static_min(self):
        policy = StaticMinPolicy(SUPPORTED_SPLIT_IDS)
        ctx = np.array([20.0, 0.5, 1.0])
        assert policy.select(ctx) == 0  # min of {0,3,6,7,8,9}

    def test_static_max(self):
        policy = StaticMaxPolicy(SUPPORTED_SPLIT_IDS)
        ctx = np.array([20.0, 0.5, 1.0])
        assert policy.select(ctx) == 9  # max of {0,3,6,7,8,9}

    def test_fixed_split(self):
        policy = FixedSplitPolicy(SUPPORTED_SPLIT_IDS, split_id=7)
        ctx = np.array([20.0, 0.5, 1.0])
        assert policy.select(ctx) == 7

    def test_fixed_split_infeasible(self):
        with pytest.raises(ValueError):
            FixedSplitPolicy([0, 3], split_id=7)

    def test_local_only(self):
        policy = LocalOnlyPolicy(SUPPORTED_SPLIT_IDS)
        ctx = np.array([20.0, 0.5, 1.0])
        assert policy.select(ctx) == 9

    def test_local_only_infeasible(self):
        with pytest.raises(ValueError):
            LocalOnlyPolicy([0, 3, 6])  # 9 not in set


class TestLinUCB:
    def test_select_returns_feasible(self):
        policy = LinUCBPolicy(SUPPORTED_SPLIT_IDS, alpha=1.0)
        ctx = np.array([20.0, 0.5, 1.0])
        selected = policy.select(ctx)
        assert selected in SUPPORTED_SPLIT_IDS

    def test_update_no_crash(self):
        policy = LinUCBPolicy(SUPPORTED_SPLIT_IDS, alpha=1.0)
        ctx = np.array([20.0, 0.5, 1.0])
        k = policy.select(ctx)
        policy.update(ctx, k, reward=0.8)

    def test_reset(self):
        policy = LinUCBPolicy(SUPPORTED_SPLIT_IDS)
        ctx = np.array([20.0, 0.5, 1.0])
        policy.select(ctx)
        policy.update(ctx, 3, 1.0)
        policy.reset()
        stats = policy.get_arm_stats()
        for k_stats in stats.values():
            assert k_stats["num_updates"] == 0

    def test_policy_factory(self):
        policy = create_policy("static_kmin", SUPPORTED_SPLIT_IDS)
        assert isinstance(policy, StaticMinPolicy)

        policy = create_policy("linucb", SUPPORTED_SPLIT_IDS)
        assert isinstance(policy, LinUCBPolicy)

    def test_empty_feasible_set_raises(self):
        with pytest.raises(ValueError):
            StaticMinPolicy([])
