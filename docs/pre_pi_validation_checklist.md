# Pre-Pi Validation Checklist (VPS/QEMU)

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
```

One-shot equivalent:

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

- `c-edge-forward-verify-all`: all export/forward parity tests pass.
- `c-edge-quant-verify`: quantization parity passes.
- `c-edge-controller-verify`: `CONTROLLER_OK` appears in test output.
- `c-edge-roundtrip-vps`: `artifacts/roundtrip/latest/summary.json` has `"all_ok": true`.
- `uk-edge-validate` log (`artifacts/qemu/unikraft_edge_selftest_arm64.log`) contains:
  - `UK_SELFTEST_EDGE_OK`
  - `UK_SELFTEST_SPLIT_OK split=3`
  - `UK_SELFTEST_SPLIT_OK split=7`
  - `UK_SELFTEST_SPLIT_OK split=8`
  - `UK_SELFTEST_CTRL_OK`
  - `UK_SELFTEST_TRANSPORT=ukstub`
  - `UK_SELFTEST_TRANSPORT_OK`
  - `UK_SELFTEST_DONE`

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
- Pi handoff artifacts:
  - `artifacts/pi_handoff/latest/pi_readiness_manifest.json`
  - `artifacts/pi_handoff/latest/pi_boot_payload.tar.gz`

## Pass/Fail Interpretation

- Pass: all commands above succeed and artifacts contain required markers.
- Fail: any command exits non-zero, or required markers/artifact summaries are missing.

## Deferred to Raspberry Pi Stage

- PMU (`lib-pmu`) hardware validation.
- INA219 / board instrumentation.
- Real lwIP/Unikraft network backend behavior on hardware.
- NEON optimization and hardware performance/timing characterization.
- MQTT deployment wiring on actual edge runtime stack.
