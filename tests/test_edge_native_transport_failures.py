"""Negative/failure-path tests for transport and cloud-client handling."""

import subprocess
from pathlib import Path


def test_transport_posix_connect_failure(c_runtime_build_dir: Path):
    failure_bin = c_runtime_build_dir / "unisplit_edge_failure_testbin"
    run = subprocess.run(
        [str(failure_bin), "transport-posix-connect-fail"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0
    assert "EXPECTED_FAIL transport-posix-connect-fail" in run.stdout


def test_cloud_client_invalid_shape_fails(c_runtime_build_dir: Path):
    failure_bin = c_runtime_build_dir / "unisplit_edge_failure_testbin"
    run = subprocess.run(
        [str(failure_bin), "cloud-invalid-shape"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0
    assert "EXPECTED_FAIL cloud-invalid-shape" in run.stdout


def test_cloud_client_non_ok_status_fails(c_runtime_build_dir: Path):
    failure_bin = c_runtime_build_dir / "unisplit_edge_failure_testbin"
    run = subprocess.run(
        [str(failure_bin), "cloud-ukstub-mode", "error"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0
    assert "EXPECTED_FAIL cloud-ukstub-mode=error" in run.stdout


def test_cloud_client_bad_json_fails(c_runtime_build_dir: Path):
    failure_bin = c_runtime_build_dir / "unisplit_edge_failure_testbin"
    run = subprocess.run(
        [str(failure_bin), "cloud-ukstub-mode", "badjson"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0
    assert "EXPECTED_FAIL cloud-ukstub-mode=badjson" in run.stdout


def test_cloud_client_ukstub_ok_path_still_works(c_runtime_build_dir: Path):
    failure_bin = c_runtime_build_dir / "unisplit_edge_failure_testbin"
    run = subprocess.run(
        [str(failure_bin), "cloud-ukstub-mode", "ok"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0
    assert "MODE_OK" in run.stdout


def test_cli_backend_swap_to_ukstub(c_runtime_build_dir: Path, edge_c_splits_artifacts: Path):
    cli = c_runtime_build_dir / "unisplit_edge_cli"
    run = subprocess.run(
        [
            str(cli),
            "--split-id",
            "7",
            "--artifacts-root",
            str(edge_c_splits_artifacts),
            "--post",
            "--transport-backend",
            "ukstub",
            "--transport-endpoint",
            "ukstub://ok",
            "--model-version",
            "v0.1.0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, f"stdout:\n{run.stdout}\nstderr:\n{run.stderr}"
    assert "TRANSPORT_BACKEND=ukstub" in run.stdout
    assert "CLOUD_OK" in run.stdout


def test_cli_rejects_unknown_transport_backend(c_runtime_build_dir: Path, edge_c_splits_artifacts: Path):
    cli = c_runtime_build_dir / "unisplit_edge_cli"
    run = subprocess.run(
        [
            str(cli),
            "--split-id",
            "7",
            "--artifacts-root",
            str(edge_c_splits_artifacts),
            "--post",
            "--transport-backend",
            "unknown_backend",
            "--model-version",
            "v0.1.0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "Transport init failed: Unsupported transport backend" in run.stderr


def test_cli_reports_lwip_backend_not_implemented(c_runtime_build_dir: Path, edge_c_splits_artifacts: Path):
    cli = c_runtime_build_dir / "unisplit_edge_cli"
    run = subprocess.run(
        [
            str(cli),
            "--split-id",
            "7",
            "--artifacts-root",
            str(edge_c_splits_artifacts),
            "--post",
            "--transport-backend",
            "lwip",
            "--transport-endpoint",
            "http://127.0.0.1:8000",
            "--model-version",
            "v0.1.0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0
    assert "Transport init failed: lwip backend is not implemented yet" in run.stderr
