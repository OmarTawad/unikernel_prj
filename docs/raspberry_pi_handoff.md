# Raspberry Pi Handoff (Final Pre-Hardware Package)

This document freezes the exact handoff package and day-1 bring-up order.

## 1) VPS-Only Paths

- Cloud service and deploy:
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
- Python simulator + historical evidence:
  - `unisplit/edge/`
  - `configs/edge.yaml`
  - `artifacts/roundtrip/latest/`
  - `artifacts/prepi/`
  - `artifacts/qemu/`

## 2) Pi-Phase Required Paths

- Edge-native runtime + transport abstraction:
  - `edge_native/runtime/include/`
  - `edge_native/runtime/src/`
- Unikraft runtime app:
  - `edge_native/unikraft_edge_selftest/`
  - `edge_native/unikraft_edge_selftest/generated/embedded_model.c`
  - `edge_native/unikraft_edge_selftest/generated/embedded_model.h`
- C artifact exports:
  - `edge_native/artifacts/c_splits/edge_k{0,3,6,7,8,9}/`
- Pi build/boot configs:
  - `configs/pi_edge_runtime.env.example`
  - `configs/pi_boot/config.txt`
  - `configs/pi_boot/cmdline.txt.template`
- Handoff/build scripts:
  - `scripts/generate_embedded_edge_model.py`
  - `scripts/build_pi_image.sh`
  - `scripts/prepare_pi_boot_media.sh`
  - `scripts/check_pi_readiness.sh`
- Contracts/checklists:
  - `docs/protocol.md`
  - `docs/edge_native_runtime.md`
  - `docs/pre_pi_validation_checklist.md`

## 3) Locked Artifact Strategy

**Embedded model strategy is final for day-1 Pi bring-up.**

- Export source: `edge_native/artifacts/c_splits/edge_k9/` (superset tensors)
- Generator: `scripts/generate_embedded_edge_model.py`
- Embedded model loaded in unikernel by `edge_model_load_embedded(...)`

No runtime filesystem dependency is required for inference tensors on day-1.

## 4) Image Build and Boot-Media Mapping

### Build image candidate

```bash
make pi-image-build
```

Outputs:
- `artifacts/pi_handoff/latest/images/kernel8.img`
- `artifacts/pi_handoff/latest/images/image_build_metadata.txt`
- `artifacts/pi_handoff/latest/images/unisplit-uk-edge-selftest_<plat>-arm64.img`

### Prepare boot-media layout

```bash
make pi-boot-media
```

Outputs:
- `artifacts/pi_handoff/latest/boot_media/boot/kernel8.img`
- `artifacts/pi_handoff/latest/boot_media/boot/config.txt`
- `artifacts/pi_handoff/latest/boot_media/boot/cmdline.txt`
- `artifacts/pi_handoff/latest/boot_media/BOOT_MEDIA_README.txt`
- `artifacts/pi_handoff/latest/boot_media/boot_media_manifest.txt`

Copy the three files under `boot/` into the FAT32 Pi boot partition.

## 5) Runtime Config Ingestion Path

The unikernel app consumes boot cmdline arguments for:
- backend selection
- endpoint host:port
- request path
- split ID
- timeout
- retries

Rendered source values come from:
- `configs/pi_edge_runtime.env.example`
- `configs/pi_boot/cmdline.txt.template`

Day-1 locked endpoint:
- `http://204.168.156.245:8000`
- path: `/infer/split`

lwIP backend endpoint format currently required:
- `http://<ipv4>:<port>` (IPv4 literal)

To override endpoint/backend without editing committed defaults:
- `UNISPLIT_PI_ENV_FILE=/abs/path/to/pi_runtime.env make pi-boot-media`

## 6) Day-1 Serial Acceptance Markers

Required markers on serial console:

- `PI_MARKER_BOOT_START`
- `PI_MARKER_ARTIFACT_STRATEGY=embedded_edge_k9_superset_v1`
- `PI_MARKER_CONFIG_OK`
- `PI_MARKER_SPLIT_DISPATCH_OK`
- `PI_MARKER_BACKEND_INIT_OK`
- `PI_MARKER_NETWORK_READY`
- `PI_MARKER_INFER_ATTEMPT`
- `PI_MARKER_INFER_RESPONSE_OK`
- `PI_MARKER_FINAL_SUCCESS`

Any `*_FAIL` marker is an immediate day-1 stop condition.

## 7) Day-1 Bring-Up Order

1. On VPS: verify cloud `/health` and `/ready`.
2. Run `make prepi-validate` on VPS.
3. Run `make pi-image-build`.
4. Run `make pi-boot-media`.
5. Copy `boot/{kernel8.img,config.txt,cmdline.txt}` to Pi boot partition.
6. Connect UART + Ethernet, power on Pi, capture serial log.
7. Confirm required markers through `PI_MARKER_NETWORK_READY`.
8. Confirm first real `/infer/split` success (`PI_MARKER_INFER_RESPONSE_OK`).
9. Archive serial logs under `artifacts/pi_handoff/` for comparison.

## 8) Still Hardware-Only

- PMU/lib-pmu validation
- INA219 instrumentation
- final on-device network behavior validation
- NEON optimization and hardware timing claims
- MQTT-on-device deployment validation
