"""Synthetic tests for C-side controller scaffolding."""

import subprocess
from pathlib import Path


def test_c_controller_linucb_sanity(c_runtime_build_dir: Path):
    binary = c_runtime_build_dir / "unisplit_edge_controller_testbin"
    assert binary.exists()

    run = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert run.returncode == 0, f"controller test failed:\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}"
    assert "CONTROLLER_OK" in run.stdout
