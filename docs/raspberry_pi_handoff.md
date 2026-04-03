# Raspberry Pi Handoff (Pre-Hardware Freeze)

This document freezes what belongs on VPS, what belongs in the Pi phase, and
what is still blocked on hardware-only work.

## 1) VPS-Only Paths

- Cloud service + deploy:
  - `unisplit/cloud/`
  - `deploy/docker-compose.yml`
  - `deploy/Dockerfile.cloud`
  - `configs/cloud.yaml`
- Training/data/profiling:
  - `unisplit/training/`
  - `unisplit/profiler/`
  - `unisplit/experiments/`
  - `data/`
  - `checkpoints/`
  - `profiles/`
- Python edge simulator:
  - `unisplit/edge/`
  - `configs/edge.yaml`
  - `deploy/Dockerfile.edge`
- PyTorch partition sources:
  - `partitions/edge_k*/partition.pt`
  - `partitions/cloud_k*/partition.pt`
  - `partitions/*/metadata.json`
- VPS evidence/logs:
  - `artifacts/roundtrip/latest/`
  - `artifacts/prepi/`
  - `artifacts/qemu/`

## 2) Pi-Phase Required Paths

- Edge-native runtime:
  - `edge_native/runtime/include/`
  - `edge_native/runtime/src/`
  - `edge_native/runtime/CMakeLists.txt`
- Transport abstraction + backend selection:
  - `edge_native/runtime/include/transport_backend.h`
  - `edge_native/runtime/src/transport_backend_factory.c`
  - `edge_native/runtime/src/transport_common.c`
  - `edge_native/runtime/src/transport_lwip_backend_stub.c`
- Unikraft app skeleton:
  - `edge_native/unikraft_edge_selftest/`
  - Optional sanity app: `edge_native/unikraft_hello/`
- C export artifacts (deploy payload candidates):
  - `edge_native/artifacts/c_splits/edge_k{0,3,6,7,8,9}/manifest.json`
  - `edge_native/artifacts/c_splits/edge_k{0,3,6,7,8,9}/*.bin`
- Contracts/checklists:
  - `docs/protocol.md`
  - `docs/edge_native_runtime.md`
  - `docs/pre_pi_validation_checklist.md`
  - `docs/unikraft_qemu_validation.md`
  - `configs/pi_edge_runtime.env.example`
- Build/export entrypoints:
  - `Makefile` (`export-edge-c-all`, `c-edge-build`, `uk-edge-validate`, `prepi-validate`)
  - `scripts/export_edge_c_artifacts.py`
  - `unisplit/edge_native/export_c.py`

## 3) Generated During Build/Deploy

- Host runtime binaries:
  - `edge_native/runtime/build/unisplit_edge_cli`
  - `edge_native/runtime/build/unisplit_edge_k7_cli`
  - `edge_native/runtime/build/unisplit_edge_*testbin`
- Unikraft QEMU outputs:
  - `edge_native/unikraft_edge_selftest/.unikraft/build/unisplit-uk-edge-selftest_qemu-arm64`
  - `.../unisplit-uk-edge-selftest_qemu-arm64.dbg`
  - `.../unisplit-uk-edge-selftest_qemu-arm64.bootinfo`
- Validation artifacts:
  - `artifacts/roundtrip/latest/{summary.json,cloud.log,split_k3.log,split_k7.log,split_k8.log}`
  - `artifacts/qemu/unikraft_edge_selftest_arm64.log`
  - `artifacts/prepi/validation_report.txt`

## 4) Network Readiness (Strict)

Implemented now:
- Backend-pluggable transport API (`transport_client_t` + factory)
- Host `posix` backend (`HTTP/1.1` POST `/infer/split`)
- `ukstub` deterministic backend for integration/self-test
- Cloud wire contract preserved (`split_id`, base64 payload, `shape`, `dtype`, quant params, `model_version`)

Not production yet:
- `lwip` backend is currently a stub returning explicit not-implemented error.
- Unikraft selftest app uses `ukstub`, not real network I/O to VPS.

Still required before Pi can communicate with VPS:
- Real Unikraft/lwIP backend implementation under existing transport interface.
- Pi endpoint config wiring in runtime startup path (`backend`, `endpoint`, timeout policy).
- Real artifact-loading strategy in unikernel path (manifest/bin loader or embedding pipeline).
- Pi platform target selection in Kraftfile (current targets are `qemu/arm64`).

## 5) Day-1 Pi Arrival Checklist

1. Confirm VPS cloud endpoint (`/health`, `/ready`) from Pi network.
2. Regenerate artifacts on VPS: `make export-edge-c-all`.
3. Generate handoff manifest: `make pi-readiness-manifest`.
4. Build payload tarball: `make pi-boot-payload`.
5. Build Pi-target unikernel image from `edge_native/unikraft_edge_selftest` (after Pi platform target is set).
6. Boot on Pi and capture serial log with current selftest markers.
7. Swap backend from `ukstub` to real `lwip` backend and run first `/infer/split` call (`k7`, then `k3`, `k8`).
8. Save boot + roundtrip logs and compare against pre-Pi baseline markers.

## 6) Current Blockers (Pre-Hardware Repo Work)

- Add Pi platform target in Kraftfile(s).
- Replace `transport_lwip_backend_stub.c` with real implementation.
- Add runtime artifact-loading path for unikernel (manifest/bin).
