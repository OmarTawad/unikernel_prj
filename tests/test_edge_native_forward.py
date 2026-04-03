"""Forward correctness tests for edge-native split-7 C runtime."""

import subprocess
from pathlib import Path

import numpy as np
import torch

from unisplit.model.partition import load_edge_partition


def test_c_forward_matches_exported_reference(c_runtime_build_dir: Path, edge_k7_c_artifacts: Path):
    binary = c_runtime_build_dir / "unisplit_edge_k7_forward_testbin"
    assert binary.exists()

    input_path = edge_k7_c_artifacts / "reference_input.bin"
    expected_path = edge_k7_c_artifacts / "reference_activation.bin"
    output_path = edge_k7_c_artifacts / "c_forward_output.bin"

    subprocess.run(
        [str(binary), str(edge_k7_c_artifacts), str(input_path), str(output_path)],
        check=True,
    )

    c_out = np.fromfile(output_path, dtype="<f4")
    py_out = np.fromfile(expected_path, dtype="<f4")

    assert c_out.shape == (64,)
    assert py_out.shape == (64,)
    assert np.allclose(c_out, py_out, atol=1e-4, rtol=1e-4)


def test_c_forward_matches_pytorch_partition(c_runtime_build_dir: Path, edge_k7_c_artifacts: Path):
    binary = c_runtime_build_dir / "unisplit_edge_k7_forward_testbin"
    input_path = edge_k7_c_artifacts / "reference_input.bin"
    output_path = edge_k7_c_artifacts / "c_forward_output_vs_torch.bin"

    subprocess.run(
        [str(binary), str(edge_k7_c_artifacts), str(input_path), str(output_path)],
        check=True,
    )

    c_out = np.fromfile(output_path, dtype="<f4")
    input_vec = np.fromfile(input_path, dtype="<f4")

    edge_model = load_edge_partition("partitions", 7)
    with torch.no_grad():
        torch_out = (
            edge_model.forward_to(torch.from_numpy(input_vec).reshape(1, 80), 7)
            .squeeze(0)
            .numpy()
            .astype(np.float32)
        )

    assert np.allclose(c_out, torch_out, atol=1e-4, rtol=1e-4)
