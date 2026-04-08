#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATEST_DIR="${ROOT_DIR}/artifacts/pi_handoff/latest"
BOOT_MEDIA_DIR="${LATEST_DIR}/boot_media_uefi"
IMAGES_DIR="${LATEST_DIR}/images"
META_FILE="${IMAGES_DIR}/pi_uefi_payload_metadata.txt"
BASELINE_SUMS="${LATEST_DIR}/UEFI_BUNDLE_BASELINE_SHA256SUMS.txt"
GATE_REPORT="${LATEST_DIR}/pi_payload_model_blocker_report.txt"
MANIFEST_FILE="${BOOT_MEDIA_DIR}/boot_media_manifest.txt"
SHA_FILE="${BOOT_MEDIA_DIR}/SHA256SUMS.txt"
README_FILE="${BOOT_MEDIA_DIR}/BOOT_MEDIA_README.txt"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[pi-uefi-media] Missing required file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "[pi-uefi-media] Missing required directory: $1" >&2
    exit 1
  fi
}

require_dir "${BOOT_MEDIA_DIR}"
require_file "${META_FILE}"
require_file "${BASELINE_SUMS}"
require_file "${GATE_REPORT}"

GATE_STATUS="$(grep -E '^status=' "${GATE_REPORT}" | head -n1 | cut -d= -f2- || true)"
if [[ "${GATE_STATUS}" != "accepted" ]]; then
  echo "[pi-uefi-media] Refusing packaging because payload model gate status is '${GATE_STATUS:-missing}'." >&2
  echo "[pi-uefi-media] See: ${GATE_REPORT}" >&2
  exit 1
fi

PAYLOAD_PLATFORM="$(grep -E '^platform=' "${META_FILE}" | head -n1 | cut -d= -f2-)"
PAYLOAD_BOOT_PROTO="$(grep -E '^boot_protocol=' "${META_FILE}" | head -n1 | cut -d= -f2-)"
PAYLOAD_ARCH="$(grep -E '^arch=' "${META_FILE}" | head -n1 | cut -d= -f2-)"
PAYLOAD_SRC="$(grep -E '^source_payload=' "${META_FILE}" | head -n1 | cut -d= -f2-)"
DTB_SRC="${BOOT_MEDIA_DIR}/bcm2711-rpi-4-b.dtb"
DTB_DST="${BOOT_MEDIA_DIR}/EFI/BOOT/bcm2711-rpi-4-b.dtb"

if grep -Eq '^kraft_platform=qemu$' "${META_FILE}"; then
  echo "[pi-uefi-media] Refusing metadata with kraft_platform=qemu." >&2
  exit 1
fi

if [[ "${PAYLOAD_PLATFORM}" != "efi" ]]; then
  echo "[pi-uefi-media] Refusing payload with platform=${PAYLOAD_PLATFORM} (expected efi)." >&2
  exit 1
fi

if [[ "${PAYLOAD_ARCH}" != "arm64" ]]; then
  echo "[pi-uefi-media] Refusing payload with arch=${PAYLOAD_ARCH} (expected arm64)." >&2
  exit 1
fi

if [[ "${PAYLOAD_BOOT_PROTO}" != "efi_stub" ]]; then
  echo "[pi-uefi-media] Refusing payload without EFI stub boot protocol." >&2
  exit 1
fi

if [[ -z "${PAYLOAD_SRC}" ]]; then
  echo "[pi-uefi-media] Missing source_payload in metadata." >&2
  exit 1
fi

if [[ "${PAYLOAD_SRC}" == *qemu* || "${PAYLOAD_SRC}" == *QEMU* ]]; then
  echo "[pi-uefi-media] Refusing qemu-derived payload source path: ${PAYLOAD_SRC}" >&2
  exit 1
fi

require_file "${PAYLOAD_SRC}"
require_file "${DTB_SRC}"

# Guardrail: keep staged pftf tree immutable prior to payload injection.
while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  expected="${line%% *}"
  rest="${line#* }"
  rel="${rest#* }"
  target="${BOOT_MEDIA_DIR}/${rel}"
  if [[ ! -f "${target}" ]]; then
    echo "[pi-uefi-media] Staged bundle drift detected (missing): ${rel}" >&2
    exit 1
  fi
  actual="$(sha256sum "${target}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "[pi-uefi-media] Staged bundle drift detected (hash mismatch): ${rel}" >&2
    exit 1
  fi
done < "${BASELINE_SUMS}"

mkdir -p "${BOOT_MEDIA_DIR}/EFI/BOOT"
cp "${PAYLOAD_SRC}" "${BOOT_MEDIA_DIR}/EFI/BOOT/BOOTAA64.EFI"
cp "${DTB_SRC}" "${DTB_DST}"
require_file "${DTB_DST}"

if find "${BOOT_MEDIA_DIR}" -maxdepth 2 -type f \( -name 'kernel8.img' -o -name 'cmdline.txt' \) | grep -q .; then
  echo "[pi-uefi-media] UEFI mode forbids kernel8.img/cmdline.txt in handoff output." >&2
  exit 1
fi

cat > "${README_FILE}" <<README
UniSplit Raspberry Pi 4 UEFI Boot Media
======================================

This directory is the authoritative SD boot partition tree for paper-aligned
Pi bring-up (Unikraft on arm64 via UEFI).

Rules:
1. Keep the extracted pftf/RPi4 bundle tree unchanged.
2. Added payload only at: EFI/BOOT/BOOTAA64.EFI
3. Do not add kernel8.img or cmdline.txt in UEFI mode.
4. Boot partition requirements:
   - FAT16 or FAT32
   - If partitioning with Linux tools, use MBR partition type 0xef

Firmware stage success signal:
- UEFI starts and attempts to launch EFI/BOOT/BOOTAA64.EFI.

App stage UART proof-of-life markers (in order):
- PI_UEFI_POF_BOOT_START
- PI_UEFI_POF_UART_OK
- PI_UEFI_POF_DONE
README

(
  cd "${BOOT_MEDIA_DIR}"
  find . -type f | sed 's#^\./##' | sort > .manifest_files.tmp
  sha256sum $(cat .manifest_files.tmp) > "${SHA_FILE}"
)

cat > "${MANIFEST_FILE}" <<MAN
timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
mode=pi4_uefi
platform=efi
boot_protocol=${PAYLOAD_BOOT_PROTO}
arch=arm64
boot_media_dir=${BOOT_MEDIA_DIR}
payload_source=${PAYLOAD_SRC}
payload_destination=${BOOT_MEDIA_DIR}/EFI/BOOT/BOOTAA64.EFI
dtb_source=${DTB_SRC}
dtb_destination=${DTB_DST}
forbidden_files=kernel8.img,cmdline.txt
baseline_checksums=${BASELINE_SUMS}
sha256_sums=${SHA_FILE}
MAN

rm -f "${BOOT_MEDIA_DIR}/.manifest_files.tmp"

echo "[ok] Pi UEFI boot media prepared."
echo "[dir ] ${BOOT_MEDIA_DIR}"
echo "[boot] ${BOOT_MEDIA_DIR}/EFI/BOOT/BOOTAA64.EFI"
echo "[dtb ] ${DTB_DST}"
echo "[file] ${README_FILE}"
echo "[file] ${MANIFEST_FILE}"
echo "[file] ${SHA_FILE}"
