# Pre-Pi Validation Checklist (QEMU + VPS + Pi UEFI Handoff)

This checklist freezes the pre-hardware baseline and the paper-aligned Pi4 UEFI
handoff flow.

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
make pi-uefi-check
```

If pftf bundle is available locally:

```bash
make pi-uefi-handoff UNISPLIT_PI_UEFI_BUNDLE=/abs/path/RPi4_UEFI_Firmware_v1.51.zip
```

One-shot baseline report:

```bash
make prepi-validate
```

## Expected Success Signals

QEMU edge self-test log (`artifacts/qemu/unikraft_edge_selftest_arm64.log`) must contain:

- `PI_MARKER_BOOT_START`
- `PI_MARKER_SPLIT_DISPATCH_OK split=7`
- `PI_MARKER_BACKEND_INIT_OK`
- `PI_MARKER_INFER_RESPONSE_OK`
- `PI_MARKER_FINAL_SUCCESS`

VPS roundtrip matrix (`artifacts/roundtrip/latest/summary.json`) must include:

- `"all_ok": true`
- split results for `k3`, `k7`, `k8` with `ok=true`

Pi UEFI payload app markers (for hardware day-1 serial):

- `PI_UEFI_POF_BOOT_START`
- `PI_UEFI_POF_UART_OK`
- `PI_UEFI_POF_DONE`

## Artifact Outputs

- Pre-Pi report:
  - `artifacts/prepi/validation_report.txt`
- VPS roundtrip evidence:
  - `artifacts/roundtrip/latest/summary.json`
  - `artifacts/roundtrip/latest/cloud.log`
  - `artifacts/roundtrip/latest/split_k3.log`
  - `artifacts/roundtrip/latest/split_k7.log`
  - `artifacts/roundtrip/latest/split_k8.log`
- QEMU Unikraft evidence:
  - `artifacts/qemu/unikraft_edge_selftest_arm64.log`
- Pi UEFI handoff outputs:
  - `artifacts/pi_handoff/latest/images/pi_uefi_payload_metadata.txt`
  - `artifacts/pi_handoff/latest/images/unikraft_pi_uart_pof_BOOTAA64.EFI`
  - `artifacts/pi_handoff/latest/boot_media_uefi/EFI/BOOT/BOOTAA64.EFI`
  - `artifacts/pi_handoff/latest/boot_media_uefi/BOOT_MEDIA_README.txt`
  - `artifacts/pi_handoff/latest/boot_media_uefi/boot_media_manifest.txt`
  - `artifacts/pi_handoff/latest/boot_media_uefi/SHA256SUMS.txt`

## SD Boot Partition Requirements (UEFI path)

- FAT16 or FAT32
- If partitioning from Linux tools: MBR type `0xef`
- Copy full contents of `artifacts/pi_handoff/latest/boot_media_uefi/` to partition root.
- Do not add `kernel8.img` or `cmdline.txt` in UEFI mode.

## Deferred to Raspberry Pi Hardware Phase

- Real Pi NIC/lwIP behavior validation on hardware
- PMU (`lib-pmu`) real counters
- INA219 instrumentation
- NEON optimization and hardware timing claims
- MQTT-on-device wiring/validation
