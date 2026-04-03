"""Multi-split forward correctness tests for the generic C runtime."""

import subprocess
from pathlib import Path

import numpy as np
import torch

from unisplit.model.partition import load_edge_partition
from unisplit.model.registry import get_split_info

SPLITS = [0, 3, 6, 7, 8, 9]


def test_c_forward_matches_exported_reference_all_splits(
    c_runtime_build_dir: Path,
    edge_c_splits_artifacts: Path,
    tmp_dir: Path,
):
    binary = c_runtime_build_dir / "unisplit_edge_forward_testbin"
    assert binary.exists()

    for split_id in SPLITS:
        artifact_dir = edge_c_splits_artifacts / f"edge_k{split_id}"
        input_path = artifact_dir / "reference_input.bin"
        expected_path = artifact_dir / "reference_activation.bin"
        output_path = tmp_dir / f"forward_split{split_id}.bin"

        subprocess.run(
            [str(binary), str(artifact_dir), str(input_path), str(output_path)],
            check=True,
        )

        c_out = np.fromfile(output_path, dtype="<f4")
        py_out = np.fromfile(expected_path, dtype="<f4")
        assert c_out.shape == py_out.shape
        assert np.allclose(c_out, py_out, atol=1e-4, rtol=1e-4), f"split={split_id}"

        split_info = get_split_info(split_id)
        expected_len = int(np.prod(split_info.output_shape))
        assert c_out.size == expected_len


def test_c_forward_matches_pytorch_partition_all_splits(
    c_runtime_build_dir: Path,
    edge_c_splits_artifacts: Path,
    tmp_dir: Path,
):
    binary = c_runtime_build_dir / "unisplit_edge_forward_testbin"
    assert binary.exists()

    for split_id in SPLITS:
        artifact_dir = edge_c_splits_artifacts / f"edge_k{split_id}"
        input_path = artifact_dir / "reference_input.bin"
        output_path = tmp_dir / f"forward_vs_torch_split{split_id}.bin"

        subprocess.run(
            [str(binary), str(artifact_dir), str(input_path), str(output_path)],
            check=True,
        )

        c_out = np.fromfile(output_path, dtype="<f4")
        input_vec = np.fromfile(input_path, dtype="<f4")

        edge_model = load_edge_partition("partitions", split_id)
        with torch.no_grad():
            torch_out = (
                edge_model.forward_to(torch.from_numpy(input_vec).reshape(1, 80), split_id)
                .squeeze(0)
                .numpy()
                .astype(np.float32)
            )
        assert np.allclose(c_out, torch_out.reshape(-1), atol=1e-4, rtol=1e-4), f"split={split_id}"
