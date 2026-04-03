"""Tests for C-friendly edge artifact export."""

import json
from pathlib import Path

from unisplit.model.registry import get_split_info


EXPECTED_TENSORS = {
    "conv1_weight": (32, 1, 3),
    "conv1_bias": (32,),
    "bn1_gamma": (32,),
    "bn1_beta": (32,),
    "bn1_running_mean": (32,),
    "bn1_running_var": (32,),
    "conv2_weight": (64, 32, 3),
    "conv2_bias": (64,),
    "bn2_gamma": (64,),
    "bn2_beta": (64,),
    "bn2_running_mean": (64,),
    "bn2_running_var": (64,),
}


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for x in shape:
        n *= x
    return n


def test_export_edge_k7_manifest_and_files(edge_k7_c_artifacts: Path):
    manifest_path = edge_k7_c_artifacts / "manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "edge_split_c_v1"
    assert manifest["split_id"] == 7
    assert manifest["input_shape"] == [1, 80]
    assert manifest["output_shape"] == [64]

    exported = {entry["name"]: entry for entry in manifest["tensors"]}
    assert set(exported.keys()) == set(EXPECTED_TENSORS.keys())

    for name, shape in EXPECTED_TENSORS.items():
        entry = exported[name]
        assert entry["shape"] == list(shape)
        assert entry["dtype"] == "float32"
        assert entry["byte_order"] == "little"
        assert entry["bytes"] == _numel(shape) * 4
        assert (edge_k7_c_artifacts / entry["file"]).exists()

    assert (edge_k7_c_artifacts / "reference_input.bin").exists()
    assert (edge_k7_c_artifacts / "reference_activation.bin").exists()
    assert not (edge_k7_c_artifacts / "bn1_num_batches_tracked.bin").exists()
    assert not (edge_k7_c_artifacts / "bn2_num_batches_tracked.bin").exists()


def test_export_all_split_manifests(edge_c_splits_artifacts: Path):
    required_names = {
        0: set(),
        3: {
            "conv1_weight",
            "conv1_bias",
            "bn1_gamma",
            "bn1_beta",
            "bn1_running_mean",
            "bn1_running_var",
        },
        6: {
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
        },
        7: {
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
        },
        8: {
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
        },
        9: {
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
        },
    }

    for split_id in (0, 3, 6, 7, 8, 9):
        artifact_dir = edge_c_splits_artifacts / f"edge_k{split_id}"
        manifest_path = artifact_dir / "manifest.json"
        assert manifest_path.exists(), f"missing manifest for split={split_id}"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_info = get_split_info(split_id)

        assert manifest["schema_version"] == "edge_split_c_v1"
        assert manifest["split_id"] == split_id
        assert manifest["input_shape"] == [1, 80]
        assert manifest["output_shape"] == list(split_info.output_shape)
        exported_names = {entry["name"] for entry in manifest["tensors"]}
        assert exported_names == required_names[split_id]
        assert (artifact_dir / "reference_input.bin").exists()
        assert (artifact_dir / "reference_activation.bin").exists()
