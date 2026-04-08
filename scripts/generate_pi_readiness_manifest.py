#!/usr/bin/env python3
"""Generate a concrete Pi-readiness manifest from the current repo state."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


VPS_ONLY_PATHS: tuple[str, ...] = (
    "unisplit/cloud",
    "deploy/docker-compose.yml",
    "deploy/Dockerfile.cloud",
    "configs/cloud.yaml",
    "unisplit/training",
    "unisplit/profiler",
    "unisplit/experiments",
    "data",
    "checkpoints",
    "profiles",
    "unisplit/edge",
    "configs/edge.yaml",
    "deploy/Dockerfile.edge",
    "partitions",
    "artifacts/roundtrip/latest",
    "artifacts/prepi",
    "artifacts/qemu",
    ".venv",
    ".pytest_cache",
    "edge_native/runtime/build",
)

PI_REQUIRED_PATHS: tuple[str, ...] = (
    "edge_native/runtime/include",
    "edge_native/runtime/src",
    "edge_native/runtime/CMakeLists.txt",
    "edge_native/unikraft_edge_selftest",
    "edge_native/unikraft_edge_selftest/generated",
    "edge_native/unikraft_pi_uart_pof",
    "edge_native/unikraft_hello",
    "edge_native/artifacts/c_splits",
    "docs/protocol.md",
    "docs/edge_native_runtime.md",
    "docs/pre_pi_validation_checklist.md",
    "docs/raspberry_pi_handoff.md",
    "docs/unikraft_qemu_validation.md",
    "configs/pi_uefi_bundle.lock.json",
    "Makefile",
    "scripts/export_edge_c_artifacts.py",
    "scripts/generate_embedded_edge_model.py",
    "scripts/stage_pi_uefi_bundle.sh",
    "scripts/build_pi_uefi_payload.sh",
    "scripts/prepare_pi_uefi_boot_media.sh",
    "scripts/build_pi_image.sh",
    "scripts/prepare_pi_boot_media.sh",
    "unisplit/edge_native/export_c.py",
)

GENERATED_DEPLOY_OUTPUTS: tuple[str, ...] = (
    "edge_native/runtime/build/unisplit_edge_cli",
    "edge_native/runtime/build/unisplit_edge_k7_cli",
    "edge_native/runtime/build/unisplit_edge_forward_testbin",
    "edge_native/runtime/build/unisplit_edge_failure_testbin",
    "edge_native/unikraft_edge_selftest/.unikraft/build/unisplit-uk-edge-selftest_qemu-arm64",
    "edge_native/unikraft_edge_selftest/.unikraft/build/unisplit-uk-edge-selftest_qemu-arm64.dbg",
    "edge_native/unikraft_edge_selftest/.unikraft/build/unisplit-uk-edge-selftest_qemu-arm64.bootinfo",
    "artifacts/pi_handoff/latest/images/pi_uefi_payload_metadata.txt",
    "artifacts/pi_handoff/latest/images/unikraft_pi_uart_pof_BOOTAA64.EFI",
    "artifacts/pi_handoff/latest/boot_media_uefi/RPI_EFI.fd",
    "artifacts/pi_handoff/latest/boot_media_uefi/start4.elf",
    "artifacts/pi_handoff/latest/boot_media_uefi/config.txt",
    "artifacts/pi_handoff/latest/boot_media_uefi/EFI/BOOT/BOOTAA64.EFI",
    "artifacts/pi_handoff/latest/boot_media_uefi/BOOT_MEDIA_README.txt",
    "artifacts/pi_handoff/latest/boot_media_uefi/boot_media_manifest.txt",
    "artifacts/pi_handoff/latest/boot_media_uefi/SHA256SUMS.txt",
    "artifacts/roundtrip/latest/summary.json",
    "artifacts/roundtrip/latest/cloud.log",
    "artifacts/roundtrip/latest/split_k3.log",
    "artifacts/roundtrip/latest/split_k7.log",
    "artifacts/roundtrip/latest/split_k8.log",
    "artifacts/qemu/unikraft_edge_selftest_arm64.log",
    "artifacts/prepi/validation_report.txt",
)

SKIP_DIRS: set[str] = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files(root: Path, *, include_hidden_builds: bool) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not root.exists():
        return
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        parts = set(p.parts)
        if any(skip in parts for skip in SKIP_DIRS):
            continue
        if not include_hidden_builds and ".unikraft" in parts:
            continue
        yield p


def collect_entries(
    repo_root: Path,
    rel_paths: tuple[str, ...],
    *,
    include_hidden_builds: bool,
    with_hash: bool,
) -> tuple[list[dict], list[str]]:
    entries: list[dict] = []
    missing: list[str] = []

    for rel in rel_paths:
        abspath = repo_root / rel
        if not abspath.exists():
            missing.append(rel)
            continue

        for f in iter_files(abspath, include_hidden_builds=include_hidden_builds):
            relf = f.relative_to(repo_root).as_posix()
            item = {
                "path": relf,
                "size_bytes": f.stat().st_size,
            }
            if with_hash:
                item["sha256"] = sha256_file(f)
            entries.append(item)

    entries.sort(key=lambda x: x["path"])
    return entries, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pi readiness manifest")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root path",
    )
    parser.add_argument(
        "--output",
        default="artifacts/pi_handoff/latest/pi_readiness_manifest.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip sha256 checksums",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_path = (repo_root / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vps_entries, vps_missing = collect_entries(
        repo_root,
        VPS_ONLY_PATHS,
        include_hidden_builds=True,
        with_hash=not args.no_hash,
    )
    pi_entries, pi_missing = collect_entries(
        repo_root,
        PI_REQUIRED_PATHS,
        include_hidden_builds=False,
        with_hash=not args.no_hash,
    )
    generated_entries, generated_missing = collect_entries(
        repo_root,
        GENERATED_DEPLOY_OUTPUTS,
        include_hidden_builds=True,
        with_hash=not args.no_hash,
    )

    payload_candidates = [
        e["path"] for e in pi_entries
        if e["path"].startswith("edge_native/artifacts/c_splits/")
        or e["path"] in {
            "configs/pi_uefi_bundle.lock.json",
            "docs/protocol.md",
            "docs/pre_pi_validation_checklist.md",
            "docs/raspberry_pi_handoff.md",
        }
    ]

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": repo_root.as_posix(),
        "vps_only_roots": list(VPS_ONLY_PATHS),
        "pi_required_roots": list(PI_REQUIRED_PATHS),
        "generated_output_paths": list(GENERATED_DEPLOY_OUTPUTS),
        "vps_only_files": vps_entries,
        "pi_required_files": pi_entries,
        "generated_outputs_present": generated_entries,
        "missing": {
            "vps_only_roots": vps_missing,
            "pi_required_roots": pi_missing,
            "generated_outputs": generated_missing,
        },
        "pi_boot_payload_candidates": sorted(payload_candidates),
        "known_pi_phase_blockers": [
            "hardware PMU/INA219 validation",
            "final on-device networking behavior verification",
            "NEON optimization and hardware timing characterization",
        ],
    }

    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"pi_required_files={len(pi_entries)} vps_only_files={len(vps_entries)}")


if __name__ == "__main__":
    main()
