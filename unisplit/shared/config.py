"""YAML-based configuration system with Pydantic validation.

Loads config from YAML files and supports environment variable overrides
with UNISPLIT_ prefix.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from unisplit.shared.constants import (
    DEFAULT_CLOUD_HOST,
    DEFAULT_CLOUD_PORT,
    DEFAULT_MEMORY_BUDGET_BYTES,
    DEFAULT_MODEL_VERSION,
    DEFAULT_OVERHEAD_BYTES,
    NUM_CLASSES,
    NUM_FEATURES,
    SUPPORTED_SPLIT_IDS,
)


class ModelConfig(BaseModel):
    """Model architecture configuration."""
    num_features: int = NUM_FEATURES
    num_classes: int = NUM_CLASSES
    supported_split_ids: list[int] = Field(default_factory=lambda: list(SUPPORTED_SPLIT_IDS))


class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    batch_size: int = 4096
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    seed: int = 42
    device: str = "cpu"
    use_class_weights: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    num_workers: int = 6
    checkpoint_dir: str = "checkpoints"
    metrics_log: str = "checkpoints/metrics.jsonl"
    save_every_n_epochs: int = 10
    log_every_n_steps: int = 200


class DatasetConfig(BaseModel):
    """Dataset paths and preprocessing config."""
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    metadata_dir: str = "data/metadata"
    splits_dir: str = "data/splits"
    num_features: int = NUM_FEATURES
    num_classes: int = NUM_CLASSES
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    normalize: bool = True
    label_column: str = "label"
    feature_columns: list[str] = Field(default_factory=list)
    class_names: list[str] = Field(default_factory=list)


class MemoryBudgetConfig(BaseModel):
    """Memory budget for feasibility computation."""
    budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES
    overhead_bytes: int = DEFAULT_OVERHEAD_BYTES


class CloudConfig(BaseModel):
    """Cloud inference service configuration."""
    host: str = DEFAULT_CLOUD_HOST
    port: int = DEFAULT_CLOUD_PORT
    backend_type: str = "pytorch_cpu"
    partition_dir: str = "partitions"
    model_version: str = DEFAULT_MODEL_VERSION
    request_timeout_seconds: int = 30
    log_level: str = "INFO"


class EdgeConfig(BaseModel):
    """Edge simulator configuration."""
    cloud_url: str = "http://localhost:8000"
    partition_dir: str = "partitions"
    profile_path: str = "profiles/default.json"
    default_policy: str = "static_kmin"
    request_timeout_seconds: int = 10
    rtt_ewma_alpha: float = 0.3
    log_level: str = "INFO"


class ExperimentConfig(BaseModel):
    """Experiment configuration."""
    name: str = ""
    description: str = ""
    policy: str = "static_kmin"
    policy_args: dict[str, Any] = Field(default_factory=dict)
    num_samples: int = -1
    output_dir: str = "experiments/results"
    runtime_label: str = "docker"


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "json"


class UniSplitConfig(BaseModel):
    """Root configuration model combining all sections."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    memory_budget: MemoryBudgetConfig = Field(default_factory=MemoryBudgetConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)
    edge: EdgeConfig = Field(default_factory=EdgeConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _apply_env_overrides(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Apply UNISPLIT_ environment variable overrides to config dict."""
    env_mappings = {
        "UNISPLIT_CLOUD_HOST": ("cloud", "host"),
        "UNISPLIT_CLOUD_PORT": ("cloud", "port"),
        "UNISPLIT_BACKEND_TYPE": ("cloud", "backend_type"),
        "UNISPLIT_CLOUD_URL": ("edge", "cloud_url"),
        "UNISPLIT_MEMORY_BUDGET": ("memory_budget", "budget_bytes"),
        "UNISPLIT_OVERHEAD_BYTES": ("memory_budget", "overhead_bytes"),
        "UNISPLIT_DEFAULT_POLICY": ("edge", "default_policy"),
        "UNISPLIT_BATCH_SIZE": ("training", "batch_size"),
        "UNISPLIT_EPOCHS": ("training", "epochs"),
        "UNISPLIT_LEARNING_RATE": ("training", "learning_rate"),
        "UNISPLIT_SEED": ("training", "seed"),
        "UNISPLIT_DEVICE": ("training", "device"),
        "UNISPLIT_LOG_LEVEL": ("logging", "level"),
    }
    for env_key, (section, field) in env_mappings.items():
        val = os.environ.get(env_key)
        if val is not None:
            if section not in config_dict:
                config_dict[section] = {}
            # Try to cast to int/float if appropriate
            try:
                val = int(val)  # type: ignore[assignment]
            except ValueError:
                try:
                    val = float(val)  # type: ignore[assignment]
                except ValueError:
                    pass
            config_dict[section][field] = val
    return config_dict


def load_config(path: str | Path | None = None) -> UniSplitConfig:
    """Load configuration from YAML file with env overrides.

    Args:
        path: Path to YAML config file. If None, returns defaults.

    Returns:
        Validated UniSplitConfig instance.
    """
    config_dict: dict[str, Any] = {}

    if path is not None:
        path = Path(path)
        if path.exists():
            with open(path) as f:
                loaded = yaml.safe_load(f) or {}
            config_dict.update(loaded)

    config_dict = _apply_env_overrides(config_dict)
    return UniSplitConfig(**config_dict)


def load_config_section(path: str | Path, section: str) -> dict[str, Any]:
    """Load a specific section from a YAML config file.

    Args:
        path: Path to YAML config file.
        section: Section key to extract.

    Returns:
        Dictionary of section config values.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        loaded = yaml.safe_load(f) or {}
    return loaded.get(section, {})
