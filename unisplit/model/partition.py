"""Partition export and loading for edge/cloud model splits.

Exports and loads PyTorch state_dicts for edge and cloud partitions,
along with metadata JSON files describing the partition.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

from unisplit.model.cnn import IoTCNN
from unisplit.model.registry import SPLIT_REGISTRY, get_split_info, validate_split_id
from unisplit.shared.constants import NUM_CLASSES, NUM_FEATURES, SUPPORTED_SPLIT_IDS
from unisplit.shared.schemas import ModelArtifactMeta


def _get_edge_modules(model: IoTCNN, split_id: int) -> OrderedDict[str, nn.Module]:
    """Get the modules that belong to the edge partition."""
    if split_id == 0:
        return OrderedDict()  # No edge modules

    modules = OrderedDict()
    modules["block1"] = model.block1
    if split_id >= 6:
        modules["block2"] = model.block2
    if split_id >= 7:
        modules["pool"] = model.pool
    if split_id >= 8:
        modules["fc1"] = model.fc1
    if split_id >= 9:
        modules["fc2"] = model.fc2
    return modules


def _get_cloud_modules(model: IoTCNN, split_id: int) -> OrderedDict[str, nn.Module]:
    """Get the modules that belong to the cloud partition."""
    if split_id == 9:
        return OrderedDict()  # No cloud modules

    modules = OrderedDict()
    if split_id < 3:
        modules["block1"] = model.block1
    if split_id < 6:
        modules["block2"] = model.block2
    if split_id < 7:
        modules["pool"] = model.pool
    if split_id < 8:
        modules["fc1"] = model.fc1
    modules["fc2"] = model.fc2
    return modules


def _count_params(state_dict: dict) -> int:
    """Count total elements in a state dict."""
    return sum(v.numel() for v in state_dict.values())


def export_edge_partition(
    model: IoTCNN,
    split_id: int,
    output_dir: str | Path,
    model_version: str = "v0.1.0",
    source_checkpoint: str = "",
) -> Path:
    """Export edge partition state dict and metadata.

    Args:
        model: Full IoTCNN model.
        split_id: Split point identifier.
        output_dir: Directory to save partition files.
        model_version: Version tag for the partition.
        source_checkpoint: Path to the source checkpoint.

    Returns:
        Path to the saved partition directory.
    """
    validate_split_id(split_id)
    output_dir = Path(output_dir)
    partition_dir = output_dir / f"edge_k{split_id}"
    partition_dir.mkdir(parents=True, exist_ok=True)

    # Collect edge state dict
    edge_modules = _get_edge_modules(model, split_id)
    edge_state = OrderedDict()
    for name, module in edge_modules.items():
        for param_name, param in module.state_dict().items():
            edge_state[f"{name}.{param_name}"] = param

    # Save state dict
    torch.save(edge_state, partition_dir / "partition.pt")

    # Save metadata
    split_info = get_split_info(split_id)
    meta = ModelArtifactMeta(
        split_id=split_id,
        partition_type="edge",
        model_version=model_version,
        input_shape=[1, NUM_FEATURES],
        output_shape=list(split_info.output_shape),
        parameter_count=_count_params(edge_state),
        export_timestamp=time.time(),
        source_checkpoint=source_checkpoint,
    )
    with open(partition_dir / "metadata.json", "w") as f:
        json.dump(meta.model_dump(), f, indent=2)

    return partition_dir


def export_cloud_partition(
    model: IoTCNN,
    split_id: int,
    output_dir: str | Path,
    model_version: str = "v0.1.0",
    source_checkpoint: str = "",
) -> Path:
    """Export cloud partition state dict and metadata."""
    validate_split_id(split_id)
    output_dir = Path(output_dir)
    partition_dir = output_dir / f"cloud_k{split_id}"
    partition_dir.mkdir(parents=True, exist_ok=True)

    # Collect cloud state dict
    cloud_modules = _get_cloud_modules(model, split_id)
    cloud_state = OrderedDict()
    for name, module in cloud_modules.items():
        for param_name, param in module.state_dict().items():
            cloud_state[f"{name}.{param_name}"] = param

    # Save state dict
    torch.save(cloud_state, partition_dir / "partition.pt")

    # Save metadata
    split_info = get_split_info(split_id)
    meta = ModelArtifactMeta(
        split_id=split_id,
        partition_type="cloud",
        model_version=model_version,
        input_shape=list(split_info.output_shape),
        output_shape=[NUM_CLASSES],
        parameter_count=_count_params(cloud_state),
        export_timestamp=time.time(),
        source_checkpoint=source_checkpoint,
    )
    with open(partition_dir / "metadata.json", "w") as f:
        json.dump(meta.model_dump(), f, indent=2)

    return partition_dir


def export_all_partitions(
    model: IoTCNN,
    output_dir: str | Path,
    model_version: str = "v0.1.0",
    source_checkpoint: str = "",
) -> dict[int, dict[str, Path]]:
    """Export edge and cloud partitions for all supported split IDs.

    Returns:
        Dict mapping split_id → {"edge": path, "cloud": path}.
    """
    result = {}
    for split_id in SUPPORTED_SPLIT_IDS:
        edge_path = export_edge_partition(
            model, split_id, output_dir, model_version, source_checkpoint
        )
        cloud_path = export_cloud_partition(
            model, split_id, output_dir, model_version, source_checkpoint
        )
        result[split_id] = {"edge": edge_path, "cloud": cloud_path}
    return result


def load_edge_partition(
    partition_dir: str | Path,
    split_id: int,
    num_features: int = NUM_FEATURES,
    num_classes: int = NUM_CLASSES,
) -> IoTCNN:
    """Load an edge partition into a model.

    Returns a full IoTCNN model with only the edge layers loaded.
    Use model.forward_to(x, split_id) for inference.
    """
    validate_split_id(split_id)
    partition_dir = Path(partition_dir) / f"edge_k{split_id}"

    model = IoTCNN(num_features=num_features, num_classes=num_classes)
    if split_id > 0:
        state = torch.load(partition_dir / "partition.pt", map_location="cpu", weights_only=True)
        # Load only the edge layers
        model_state = model.state_dict()
        model_state.update(state)
        model.load_state_dict(model_state, strict=False)

    model.eval()
    return model


def load_cloud_partition(
    partition_dir: str | Path,
    split_id: int,
    num_features: int = NUM_FEATURES,
    num_classes: int = NUM_CLASSES,
) -> IoTCNN:
    """Load a cloud partition into a model.

    Returns a full IoTCNN model with only the cloud layers loaded.
    Use model.forward_from(h, split_id) for inference.
    """
    validate_split_id(split_id)
    partition_dir = Path(partition_dir) / f"cloud_k{split_id}"

    model = IoTCNN(num_features=num_features, num_classes=num_classes)
    if split_id < 9:
        state = torch.load(partition_dir / "partition.pt", map_location="cpu", weights_only=True)
        model_state = model.state_dict()
        model_state.update(state)
        model.load_state_dict(model_state, strict=False)

    model.eval()
    return model


def load_partition_metadata(partition_dir: str | Path, split_id: int, partition_type: str) -> ModelArtifactMeta:
    """Load partition metadata JSON."""
    partition_dir = Path(partition_dir) / f"{partition_type}_k{split_id}"
    with open(partition_dir / "metadata.json") as f:
        return ModelArtifactMeta(**json.load(f))
