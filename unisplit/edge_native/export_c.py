"""Export edge split artifacts to a C-friendly binary format.

This exporter intentionally avoids any C-side `.pt` parsing by writing plain
little-endian float32 tensors and a compact JSON manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from unisplit.model.partition import load_edge_partition
from unisplit.model.registry import get_split_info
from unisplit.shared.constants import NUM_FEATURES

SUPPORTED_SPLIT_IDS: tuple[int, ...] = (0, 3, 6, 7, 8, 9)

# export name -> (state_dict key, shape)
_ALL_EDGE_TENSORS: dict[str, tuple[str, tuple[int, ...]]] = {
    "conv1_weight": ("block1.conv1.weight", (32, 1, 3)),
    "conv1_bias": ("block1.conv1.bias", (32,)),
    "bn1_gamma": ("block1.bn1.weight", (32,)),
    "bn1_beta": ("block1.bn1.bias", (32,)),
    "bn1_running_mean": ("block1.bn1.running_mean", (32,)),
    "bn1_running_var": ("block1.bn1.running_var", (32,)),
    "conv2_weight": ("block2.conv2.weight", (64, 32, 3)),
    "conv2_bias": ("block2.conv2.bias", (64,)),
    "bn2_gamma": ("block2.bn2.weight", (64,)),
    "bn2_beta": ("block2.bn2.bias", (64,)),
    "bn2_running_mean": ("block2.bn2.running_mean", (64,)),
    "bn2_running_var": ("block2.bn2.running_var", (64,)),
    "fc1_weight": ("fc1.linear1.weight", (128, 64)),
    "fc1_bias": ("fc1.linear1.bias", (128,)),
    "fc2_weight": ("fc2.weight", (34, 128)),
    "fc2_bias": ("fc2.bias", (34,)),
}

_REQUIRED_EXPORTS_BY_SPLIT: dict[int, tuple[str, ...]] = {
    0: (),
    3: (
        "conv1_weight",
        "conv1_bias",
        "bn1_gamma",
        "bn1_beta",
        "bn1_running_mean",
        "bn1_running_var",
    ),
    6: (
        "conv1_weight",
        "conv1_bias",
        "bn1_gamma",
        "bn1_beta",
        "bn1_running_mean",
        "bn1_running_var",
        "conv2_weight",
        "conv2_bias",
        "bn2_gamma",
        "bn2_beta",
        "bn2_running_mean",
        "bn2_running_var",
    ),
    7: (
        "conv1_weight",
        "conv1_bias",
        "bn1_gamma",
        "bn1_beta",
        "bn1_running_mean",
        "bn1_running_var",
        "conv2_weight",
        "conv2_bias",
        "bn2_gamma",
        "bn2_beta",
        "bn2_running_mean",
        "bn2_running_var",
    ),
    8: (
        "conv1_weight",
        "conv1_bias",
        "bn1_gamma",
        "bn1_beta",
        "bn1_running_mean",
        "bn1_running_var",
        "conv2_weight",
        "conv2_bias",
        "bn2_gamma",
        "bn2_beta",
        "bn2_running_mean",
        "bn2_running_var",
        "fc1_weight",
        "fc1_bias",
    ),
    9: (
        "conv1_weight",
        "conv1_bias",
        "bn1_gamma",
        "bn1_beta",
        "bn1_running_mean",
        "bn1_running_var",
        "conv2_weight",
        "conv2_bias",
        "bn2_gamma",
        "bn2_beta",
        "bn2_running_mean",
        "bn2_running_var",
        "fc1_weight",
        "fc1_bias",
        "fc2_weight",
        "fc2_bias",
    ),
}


def _tensor_to_f32_le(arr: np.ndarray) -> np.ndarray:
    """Return contiguous little-endian float32 tensor."""
    return np.ascontiguousarray(arr.astype("<f4", copy=False))


def _write_f32_tensor(path: Path, arr: np.ndarray) -> int:
    data = _tensor_to_f32_le(arr)
    data.tofile(path)
    return int(data.nbytes)


def export_edge_split_to_c(
    partitions_dir: str | Path,
    split_id: int,
    out_dir: str | Path,
    model_version: str,
    source_checkpoint: str,
    eps: float = 1e-5,
    export_reference: bool = True,
) -> Path:
    """Export edge partition tensors for one split to C-friendly files.

    Args:
        partitions_dir: Root directory that contains edge partitions.
        split_id: Supported split ID to export.
        out_dir: Output directory for `.bin` tensors and `manifest.json`.
        model_version: Model version to embed in manifest.
        source_checkpoint: Source checkpoint path string for traceability.
        eps: BatchNorm epsilon used by the C runtime.
        export_reference: Whether to export deterministic forward reference files.

    Returns:
        Path to written `manifest.json`.
    """
    partitions_dir = Path(partitions_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if split_id not in SUPPORTED_SPLIT_IDS:
        raise ValueError(f"Unsupported split_id={split_id}. Expected one of {SUPPORTED_SPLIT_IDS}")

    partition_path = partitions_dir / f"edge_k{split_id}" / "partition.pt"
    if not partition_path.exists():
        raise FileNotFoundError(f"Missing partition file: {partition_path}")

    state = torch.load(partition_path, map_location="cpu", weights_only=True)
    required_exports = _REQUIRED_EXPORTS_BY_SPLIT[split_id]

    tensor_entries: list[dict[str, Any]] = []
    for export_name in required_exports:
        state_key, expected_shape = _ALL_EDGE_TENSORS[export_name]
        if state_key not in state:
            raise KeyError(f"State dict missing required key: {state_key}")

        tensor = state[state_key].detach().cpu().numpy()
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Unexpected shape for {state_key}: {tuple(tensor.shape)} != {expected_shape}"
            )

        file_name = f"{export_name}.bin"
        nbytes = _write_f32_tensor(out_dir / file_name, tensor)

        tensor_entries.append(
            {
                "name": export_name,
                "state_key": state_key,
                "file": file_name,
                "shape": list(expected_shape),
                "dtype": "float32",
                "byte_order": "little",
                "bytes": nbytes,
            }
        )

    split_info = get_split_info(split_id)
    manifest: dict[str, Any] = {
        "schema_version": "edge_split_c_v1",
        "split_id": split_id,
        "model_version": model_version,
        "source_checkpoint": source_checkpoint,
        "eps": float(eps),
        "input_shape": [1, NUM_FEATURES],
        "output_shape": list(split_info.output_shape),
        "tensor_format": "float32-le-bin",
        "tensors": tensor_entries,
    }

    if export_reference:
        rng = np.random.default_rng(42)
        reference_input = rng.standard_normal(NUM_FEATURES).astype(np.float32)

        edge_model = load_edge_partition(partitions_dir, split_id)
        with torch.no_grad():
            inp = torch.from_numpy(reference_input).reshape(1, NUM_FEATURES)
            activation = edge_model.forward_to(inp, split_id).squeeze(0).numpy().astype(np.float32)

        input_file = "reference_input.bin"
        activation_file = "reference_activation.bin"
        _write_f32_tensor(out_dir / input_file, reference_input)
        _write_f32_tensor(out_dir / activation_file, activation)
        if split_id == 7:
            _write_f32_tensor(out_dir / "reference_activation_k7.bin", activation)

        manifest["reference"] = {
            "seed": 42,
            "input_file": input_file,
            "input_shape": [NUM_FEATURES],
            "activation_file": activation_file,
            "activation_shape": list(split_info.output_shape),
            "dtype": "float32",
            "byte_order": "little",
        }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def export_all_edge_splits_to_c(
    partitions_dir: str | Path,
    out_root_dir: str | Path,
    model_version: str,
    source_checkpoint: str,
    eps: float = 1e-5,
    export_reference: bool = True,
) -> dict[int, Path]:
    """Export all supported edge split partitions to C-friendly files."""
    out_root = Path(out_root_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    result: dict[int, Path] = {}
    for split_id in SUPPORTED_SPLIT_IDS:
        out_dir = out_root / f"edge_k{split_id}"
        manifest_path = export_edge_split_to_c(
            partitions_dir=partitions_dir,
            split_id=split_id,
            out_dir=out_dir,
            model_version=model_version,
            source_checkpoint=source_checkpoint,
            eps=eps,
            export_reference=export_reference,
        )
        result[split_id] = manifest_path
    return result


def export_edge_k7_to_c(
    partitions_dir: str | Path,
    out_dir: str | Path,
    model_version: str,
    source_checkpoint: str,
    eps: float = 1e-5,
    export_reference: bool = True,
) -> Path:
    """Backward-compatible split-7 exporter wrapper."""
    return export_edge_split_to_c(
        partitions_dir=partitions_dir,
        split_id=7,
        out_dir=out_dir,
        model_version=model_version,
        source_checkpoint=source_checkpoint,
        eps=eps,
        export_reference=export_reference,
    )
