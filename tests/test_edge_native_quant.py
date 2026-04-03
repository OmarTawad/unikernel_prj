"""Quantization parity tests between C runtime and Python implementation."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from unisplit.shared.quantization import quantize_int8


@pytest.mark.parametrize("length", [64, 128, 2496])
def test_c_quantization_matches_python(c_runtime_build_dir: Path, tmp_dir: Path, length: int):
    binary = c_runtime_build_dir / "unisplit_edge_k7_quant_testbin"
    assert binary.exists()

    rng = np.random.default_rng(123)
    x = rng.normal(loc=0.0, scale=1.25, size=length).astype(np.float32)

    in_path = tmp_dir / "quant_input.bin"
    out_i8_path = tmp_dir / "quant_output.bin"
    out_scale_path = tmp_dir / "quant_scale.txt"

    x.astype("<f4").tofile(in_path)

    subprocess.run(
        [str(binary), str(in_path), str(out_i8_path), str(out_scale_path)],
        check=True,
    )

    c_q = np.fromfile(out_i8_path, dtype=np.int8)
    c_scale = float(out_scale_path.read_text(encoding="utf-8").strip())

    py_q, py_params = quantize_int8(x)

    assert c_q.shape == py_q.shape
    assert np.array_equal(c_q, py_q)
    assert abs(c_scale - py_params.scale) < 1e-6
