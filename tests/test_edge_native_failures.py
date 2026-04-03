"""Negative/failure-path tests for edge model loading and runtime dispatch."""

import shutil
import subprocess
from pathlib import Path


def test_missing_tensor_file_fails_model_load(c_runtime_build_dir: Path, edge_c_splits_artifacts: Path, tmp_dir: Path):
    failure_bin = c_runtime_build_dir / "unisplit_edge_failure_testbin"
    src = edge_c_splits_artifacts / "edge_k7"
    dst = tmp_dir / "edge_k7_missing_tensor"
    shutil.copytree(src, dst)

    (dst / "conv1_weight.bin").unlink()
    run = subprocess.run([str(failure_bin), "model-load", str(dst)], capture_output=True, text=True, check=False)
    assert run.returncode == 0
    assert "EXPECTED_FAIL model-load" in run.stdout


def test_malformed_manifest_fails_model_load(c_runtime_build_dir: Path, edge_c_splits_artifacts: Path, tmp_dir: Path):
    failure_bin = c_runtime_build_dir / "unisplit_edge_failure_testbin"
    src = edge_c_splits_artifacts / "edge_k7"
    dst = tmp_dir / "edge_k7_bad_manifest"
    shutil.copytree(src, dst)

    (dst / "manifest.json").write_text("{this is not valid json", encoding="utf-8")
    run = subprocess.run([str(failure_bin), "model-load", str(dst)], capture_output=True, text=True, check=False)
    assert run.returncode == 0
    assert "EXPECTED_FAIL model-load" in run.stdout


def test_unsupported_split_id_cli_fails(c_runtime_build_dir: Path, edge_c_splits_artifacts: Path):
    cli = c_runtime_build_dir / "unisplit_edge_cli"
    run = subprocess.run(
        [
            str(cli),
            "--split-id",
            "5",
            "--artifacts-root",
            str(edge_c_splits_artifacts),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "Unsupported split id" in run.stderr


def test_forward_shape_mismatch_small_output_buffer(c_runtime_build_dir: Path, edge_c_splits_artifacts: Path):
    failure_bin = c_runtime_build_dir / "unisplit_edge_failure_testbin"
    artifact_dir = edge_c_splits_artifacts / "edge_k8"
    input_bin = artifact_dir / "reference_input.bin"
    run = subprocess.run(
        [str(failure_bin), "forward-small-output", str(artifact_dir), str(input_bin)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0
    assert "EXPECTED_FAIL forward-small-output" in run.stdout
