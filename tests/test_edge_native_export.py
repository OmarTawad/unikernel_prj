"""Tests for C-friendly edge artifact export."""

import json
from pathlib import Path


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
    assert manifest["schema_version"] == "edge_k7_c_v1"
    assert manifest["split_id"] == 7
    assert manifest["input_shape"] == [1, 80]
    assert manifest["activation_shape"] == [64]

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
    assert (edge_k7_c_artifacts / "reference_activation_k7.bin").exists()
    assert not (edge_k7_c_artifacts / "bn1_num_batches_tracked.bin").exists()
    assert not (edge_k7_c_artifacts / "bn2_num_batches_tracked.bin").exists()
