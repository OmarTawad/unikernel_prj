"""Generic multi-split C runtime round-trip test against cloud /infer/split API."""

import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest


def _can_connect_loopback() -> bool:
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except (PermissionError, OSError):
        return False
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def _accept_once() -> None:
        try:
            conn, _ = server.accept()
            conn.close()
        except Exception:
            pass
        finally:
            server.close()

    threading.Thread(target=_accept_once, daemon=True).start()

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(1.0)
    try:
        client.connect(("127.0.0.1", port))
        return True
    except Exception:
        return False
    finally:
        client.close()


def test_c_cli_roundtrip_generic_splits(
    c_runtime_build_dir: Path,
    edge_c_splits_artifacts: Path,
    repo_root: Path,
):
    if not _can_connect_loopback():
        pytest.skip("Loopback TCP is unavailable in this execution sandbox.")

    port = 18081
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["UNISPLIT_CLOUD_HOST"] = "127.0.0.1"
    env["UNISPLIT_CLOUD_PORT"] = str(port)

    cloud_proc = subprocess.Popen(
        [sys.executable, "scripts/run_cloud.py", "--config", "configs/cloud.yaml"],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        deadline = time.time() + 60.0
        ready = False
        while time.time() < deadline:
            if cloud_proc.poll() is not None:
                break
            try:
                resp = httpx.get(f"{base_url}/health", timeout=1.0, trust_env=False)
                if resp.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not ready:
            if cloud_proc.poll() is None:
                cloud_proc.terminate()
                try:
                    cloud_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    cloud_proc.kill()
                    cloud_proc.wait(timeout=5)
            logs = ""
            if cloud_proc.stdout is not None:
                logs = cloud_proc.stdout.read()
            assert False, f"Cloud service did not become healthy in time. Logs:\\n{logs}"

        cli_bin = c_runtime_build_dir / "unisplit_edge_cli"
        for split_id in (3, 7, 8):
            run = subprocess.run(
                [
                    str(cli_bin),
                    "--split-id",
                    str(split_id),
                    "--artifacts-root",
                    str(edge_c_splits_artifacts),
                    "--post",
                    "--cloud-url",
                    base_url,
                    "--model-version",
                    "v0.1.0",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            assert run.returncode == 0, (
                f"C CLI failed for split={split_id}:\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}"
            )
            assert "CLOUD_OK" in run.stdout
            assert "status=ok" in run.stdout

            m = re.search(r"class=(\d+)", run.stdout)
            assert m, f"Could not parse class for split={split_id}:\n{run.stdout}"
            cls = int(m.group(1))
            assert 0 <= cls < 34

    finally:
        cloud_proc.terminate()
        try:
            cloud_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cloud_proc.kill()
            cloud_proc.wait(timeout=5)
