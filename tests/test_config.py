"""Tests for configuration loading."""

import os
import tempfile
from pathlib import Path

import yaml

from unisplit.shared.config import UniSplitConfig, load_config
from unisplit.shared.constants import DEFAULT_MEMORY_BUDGET_BYTES, NUM_CLASSES, NUM_FEATURES


class TestConfig:
    def test_default_config(self):
        config = UniSplitConfig()
        assert config.model.num_features == NUM_FEATURES
        assert config.model.num_classes == NUM_CLASSES
        assert config.memory_budget.budget_bytes == DEFAULT_MEMORY_BUDGET_BYTES

    def test_load_config_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"cloud": {"port": 9000}}, f)
            f.flush()
            config = load_config(f.name)
            assert config.cloud.port == 9000
        os.unlink(f.name)

    def test_load_config_none(self):
        config = load_config(None)
        assert config.cloud.port == 8000

    def test_env_override(self):
        os.environ["UNISPLIT_CLOUD_PORT"] = "9999"
        try:
            config = load_config(None)
            assert config.cloud.port == 9999
        finally:
            del os.environ["UNISPLIT_CLOUD_PORT"]

    def test_training_defaults(self):
        config = UniSplitConfig()
        assert config.training.batch_size == 4096
        assert config.training.epochs == 50
        assert config.training.learning_rate == 0.001

    def test_dataset_defaults(self):
        config = UniSplitConfig()
        assert config.dataset.train_ratio == 0.70
        assert config.dataset.val_ratio == 0.15
        assert config.dataset.test_ratio == 0.15
