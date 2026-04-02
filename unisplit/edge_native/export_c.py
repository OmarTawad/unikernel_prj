"""Export split-7 edge artifacts to a C-friendly binary format.

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
from unisplit.shared.constants import NUM_FEATURES

SPLIT_ID = 7

# name -> (state_dict key, expected shape)
EDGE_K7_TENSORS: dict[str, tuple[str, tuple[int, ...]]] = {
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
}


def _tensor_to_f32_le(arr: np.ndarray) -> np.ndarray:
    """Return contiguous little-endian float32 tensor."""
    return np.ascontiguousarray(arr.astype("<f4", copy=False))


def _write_f32_tensor(path: Path, arr: np.ndarray) -> int:
    data = _tensor_to_f32_le(arr)
    data.tofile(path)
    return int(data.nbytes)


def export_edge_k7_to_c(
    partitions_dir: str | Path,
    out_dir: str | Path,
    model_version: str,
    source_checkpoint: str,
    eps: float = 1e-5,
    export_reference: bool = True,
) -> Path:
    """Export split-7 edge partition tensors to C-friendly files.

    Args:
        partitions_dir: Root directory that contains `edge_k7/partition.pt`.
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

    partition_path = partitions_dir / "edge_k7" / "partition.pt"
    if not partition_path.exists():
        raise FileNotFoundError(f"Missing split-7 partition file: {partition_path}")

    state = torch.load(partition_path, map_location="cpu", weights_only=True)

    tensor_entries: list[dict[str, Any]] = []
    for export_name, (state_key, expected_shape) in EDGE_K7_TENSORS.items():
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

    manifest: dict[str, Any] = {
        "schema_version": "edge_k7_c_v1",
        "split_id": SPLIT_ID,
        "model_version": model_version,
        "source_checkpoint": source_checkpoint,
        "eps": float(eps),
        "input_shape": [1, NUM_FEATURES],
        "activation_shape": [64],
        "tensor_format": "float32-le-bin",
        "tensors": tensor_entries,
    }

    if export_reference:
        rng = np.random.default_rng(42)
        reference_input = rng.standard_normal(NUM_FEATURES).astype(np.float32)

        edge_model = load_edge_partition(partitions_dir, SPLIT_ID)
        with torch.no_grad():
            inp = torch.from_numpy(reference_input).reshape(1, NUM_FEATURES)
            activation = edge_model.forward_to(inp, SPLIT_ID).squeeze(0).numpy().astype(np.float32)

        input_file = "reference_input.bin"
        activation_file = "reference_activation_k7.bin"
        _write_f32_tensor(out_dir / input_file, reference_input)
        _write_f32_tensor(out_dir / activation_file, activation)

        manifest["reference"] = {
            "seed": 42,
            "input_file": input_file,
            "input_shape": [NUM_FEATURES],
            "activation_file": activation_file,
            "activation_shape": [64],
            "dtype": "float32",
            "byte_order": "little",
        }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
