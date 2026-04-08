# Raspberry Pi 4 UEFI Handoff (Paper-Aligned)

This document defines the only valid Pi bring-up handoff flow for this repo.

## 1) Boot Model (Locked)

- Target board: Raspberry Pi 4 Model B
- Runtime: Unikraft, arm64 EFI payload
- Boot chain: Pi firmware -> `RPI_EFI.fd` -> `EFI/BOOT/BOOTAA64.EFI`
- First milestone: UART proof-of-life only

QEMU outputs are pre-hardware validation artifacts and are not valid Pi boot payloads.

Toolchain note:
- Current Kraft CLI in this repo exposes `qemu/fc/xen` platform targets.
- For Pi UEFI handoff we build via `qemu/arm64` with `CONFIG_KVM_BOOT_PROTO_EFI_STUB=y`,
  then package the resulting PE/EFI payload as `EFI/BOOT/BOOTAA64.EFI`.
- This is a UEFI payload path, not the legacy `kernel8.img` direct-firmware path.

## 2) Immutable Firmware Source Rule

Use exactly one pinned pftf bundle source of truth:

- Lock file: `configs/pi_uefi_bundle.lock.json`
- Current pinned bundle:
  - `RPi4_UEFI_Firmware_v1.51.zip`
  - SHA256: `000b6c518e83bb93262ed6b264a0e9498509c46513dabf58c0dbb73d4c2e7c18`

Guardrail:
- Extract the bundle as-is.
- Do not rename or rewrite bundle files.
- Only add one file afterward: `EFI/BOOT/BOOTAA64.EFI`.

## 3) Commands (Canonical)

```bash
# 1) Tooling + repo checks
make pi-uefi-check

# 2) Stage immutable pftf bundle tree
make pi-uefi-stage UNISPLIT_PI_UEFI_BUNDLE=/abs/path/RPi4_UEFI_Firmware_v1.51.zip

# 3) Build minimal Unikraft UEFI UART proof payload
make pi-uefi-build

# 4) Assemble final boot-media handoff tree
make pi-uefi-boot-media

# 5) One-shot
make pi-uefi-handoff UNISPLIT_PI_UEFI_BUNDLE=/abs/path/RPi4_UEFI_Firmware_v1.51.zip
```

Legacy direct-firmware targets are intentionally disabled:
- `make pi-image-build`
- `make pi-boot-media`

## 4) Artifact Outputs

Output root:
- `artifacts/pi_handoff/latest/boot_media_uefi/`

Expected key outputs:
- `artifacts/pi_handoff/latest/images/pi_uefi_payload_metadata.txt`
- `artifacts/pi_handoff/latest/images/unikraft_pi_uart_pof_BOOTAA64.EFI`
- `artifacts/pi_handoff/latest/boot_media_uefi/EFI/BOOT/BOOTAA64.EFI`
- `artifacts/pi_handoff/latest/boot_media_uefi/BOOT_MEDIA_README.txt`
- `artifacts/pi_handoff/latest/boot_media_uefi/boot_media_manifest.txt`
- `artifacts/pi_handoff/latest/boot_media_uefi/SHA256SUMS.txt`

## 5) SD Boot Partition Requirements

- Filesystem: FAT16 or FAT32
- If partitioning from Linux tools, use MBR partition type `0xef`
- Copy the full tree from:
  - `artifacts/pi_handoff/latest/boot_media_uefi/`
  to the root of the Pi boot partition (no extra nesting)

UEFI mode rules:
- Do not add `kernel8.img`
- Do not add `cmdline.txt`

## 6) Acceptance Stages

Firmware-stage success:
- Pi reaches UEFI and attempts to launch `EFI/BOOT/BOOTAA64.EFI`.

App-stage success (UART markers in order):
- `PI_UEFI_POF_BOOT_START`
- `PI_UEFI_POF_UART_OK`
- `PI_UEFI_POF_DONE`

Marker source:
- `edge_native/unikraft_pi_uart_pof/main.c`

## 7) What This Handoff Does Not Cover Yet

Deferred until hardware/network phase:
- Full edge inference runtime on Pi
- lwIP cloud roundtrip on Pi NIC path
- PMU/lib-pmu, INA219
- NEON optimization and hardware timing claims
- MQTT-on-device validation
