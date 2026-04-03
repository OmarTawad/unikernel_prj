# Pre-Pi Validation Checklist (VPS/QEMU + Pi Packaging)

This checklist freezes the pre-hardware baseline so Raspberry Pi bring-up can
focus on hardware-only work.

## Canonical Validation Sequence

```bash
make export-edge-c-all
make c-edge-build
make c-edge-forward-verify-all
make c-edge-quant-verify
make c-edge-controller-verify
make c-edge-failure-verify
make c-edge-roundtrip-vps
make uk-edge-validate
make pi-image-build
make pi-boot-media
```

One-shot equivalent for runtime validations:

```bash
make prepi-validate
```

Pi handoff helpers:

```bash
make pi-readiness-manifest
make pi-readiness-check
make pi-boot-payload
```

## Expected Success Markers

Unikraft serial log (`artifacts/qemu/unikraft_edge_selftest_arm64.log`) must contain:

- `PI_MARKER_BOOT_START`
- `PI_MARKER_ARTIFACT_STRATEGY=embedded_edge_k9_superset_v1`
- `PI_MARKER_CONFIG_OK`
- `PI_MARKER_SPLIT_DISPATCH_OK split=7`
- `PI_MARKER_CONTROLLER_OK`
- `PI_MARKER_BACKEND_INIT_OK`
- `PI_MARKER_NETWORK_READY`
- `PI_MARKER_INFER_ATTEMPT`
- `PI_MARKER_INFER_RESPONSE_OK`
- `PI_MARKER_FINAL_SUCCESS`
- `UK_SELFTEST_DONE`

Roundtrip matrix evidence (`artifacts/roundtrip/latest/summary.json`) must include:

- `"all_ok": true`
- split results for `k3`, `k7`, `k8` with `ok=true`

## Artifact Outputs

- VPS roundtrip evidence:
  - `artifacts/roundtrip/latest/summary.json`
  - `artifacts/roundtrip/latest/cloud.log`
  - `artifacts/roundtrip/latest/split_k3.log`
  - `artifacts/roundtrip/latest/split_k7.log`
  - `artifacts/roundtrip/latest/split_k8.log`
- Unikraft/QEMU evidence:
  - `artifacts/qemu/unikraft_edge_selftest_arm64.log`
- Full pre-Pi report:
  - `artifacts/prepi/validation_report.txt`
- Pi image + boot-media outputs:
  - `artifacts/pi_handoff/latest/images/kernel8.img`
  - `artifacts/pi_handoff/latest/images/image_build_metadata.txt`
  - `artifacts/pi_handoff/latest/boot_media/boot/kernel8.img`
  - `artifacts/pi_handoff/latest/boot_media/boot/config.txt`
  - `artifacts/pi_handoff/latest/boot_media/boot/cmdline.txt`
  - `artifacts/pi_handoff/latest/boot_media/BOOT_MEDIA_README.txt`
  - `artifacts/pi_handoff/latest/boot_media/boot_media_manifest.txt`

## Day-1 Endpoint Lock

- Pi boot cmdline endpoint must be:
  - `http://204.168.156.245:8000`
- Request path must be:
  - `/infer/split`
- Verify in:
  - `artifacts/pi_handoff/latest/boot_media/boot/cmdline.txt`

## Pass/Fail Interpretation

- Pass: all commands above succeed and required markers/artifact files are present.
- Fail: any command exits non-zero, any required marker is missing, or expected output files are absent.

## Deferred to Raspberry Pi Hardware Stage

- PMU (`lib-pmu`) real counter validation.
- INA219 and board-level instrumentation.
- Hardware-level network behavior verification on real Pi NIC path.
- NEON optimization and hardware timing/performance characterization.
- MQTT deployment wiring validation on actual device runtime.
