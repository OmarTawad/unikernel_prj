"""Tests for dataset config validation."""

from unisplit.shared.config import UniSplitConfig
from unisplit.shared.constants import NUM_CLASSES, NUM_FEATURES


class TestDatasetConfig:
    def test_feature_count(self):
        config = UniSplitConfig()
        assert config.dataset.num_features == NUM_FEATURES
        assert config.dataset.num_features == 80

    def test_class_count(self):
        config = UniSplitConfig()
        assert config.dataset.num_classes == NUM_CLASSES
        assert config.dataset.num_classes == 34

    def test_split_ratios_sum(self):
        config = UniSplitConfig()
        total = config.dataset.train_ratio + config.dataset.val_ratio + config.dataset.test_ratio
        assert abs(total - 1.0) < 1e-6

    def test_paths_not_empty(self):
        config = UniSplitConfig()
        assert config.dataset.raw_dir
        assert config.dataset.processed_dir
        assert config.dataset.metadata_dir
        assert config.dataset.splits_dir
