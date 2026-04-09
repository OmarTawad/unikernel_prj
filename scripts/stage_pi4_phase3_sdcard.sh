#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="${ROOT_DIR}/configs/pi_uefi_bundle.lock.json"
IMAGE_PATH="${ROOT_DIR}/artifacts/pi_phase3/latest/images/kernel8.img"
OUT_DIR="${ROOT_DIR}/artifacts/pi_phase3/latest/sdcard"
CONFIG_SRC="${ROOT_DIR}/configs/pi_boot/config.txt"
MANIFEST_FILE="${OUT_DIR}/pi4_phase3_sdcard_manifest.txt"
README_FILE="${OUT_DIR}/README_PHASE3.txt"
SHA_FILE="${OUT_DIR}/SHA256SUMS.txt"
TMP_DIR="${ROOT_DIR}/artifacts/pi_phase3/latest/.stage_tmp"
BUNDLE_PATH="${1:-${UNISPLIT_PI_FIRMWARE_BUNDLE:-${UNISPLIT_PI_UEFI_BUNDLE:-}}}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[pi4-phase3-stage] Missing required tool: $1" >&2
    exit 1
  }
}

need_cmd python3
need_cmd sha256sum
need_cmd unzip

if [[ -z "${BUNDLE_PATH}" ]]; then
  echo "[pi4-phase3-stage] Missing firmware bundle path." >&2
  echo "[pi4-phase3-stage] Set UNISPLIT_PI_FIRMWARE_BUNDLE=/abs/path/RPi4_UEFI_Firmware_v1.51.zip" >&2
  exit 1
fi

if [[ ! -f "${BUNDLE_PATH}" ]]; then
  echo "[pi4-phase3-stage] Firmware bundle not found: ${BUNDLE_PATH}" >&2
  exit 1
fi

if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "[pi4-phase3-stage] Missing built kernel image: ${IMAGE_PATH}" >&2
  exit 1
fi

python3 - "${LOCK_FILE}" "${BUNDLE_PATH}" <<'PY'
import hashlib
import json
import pathlib
import sys
import zipfile

lock = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
bundle = pathlib.Path(sys.argv[2])

sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
expected = lock["sha256"].strip().lower()
if sha != expected:
    raise SystemExit(f"sha256 mismatch: expected {expected} got {sha}")

required = {
    "start4.elf",
    "fixup4.dat",
    "bcm2711-rpi-4-b.dtb",
    "overlays/miniuart-bt.dtbo",
}

with zipfile.ZipFile(bundle, "r") as zf:
    names = set(zf.namelist())

missing = sorted(required - names)
if missing:
    raise SystemExit("bundle missing required entries: " + ", ".join(missing))
PY

mkdir -p "${OUT_DIR}" "${OUT_DIR}/overlays" "${TMP_DIR}"
rm -f "${OUT_DIR}/start4.elf" "${OUT_DIR}/fixup4.dat" "${OUT_DIR}/bcm2711-rpi-4-b.dtb" \
  "${OUT_DIR}/kernel8.img" "${OUT_DIR}/config.txt" "${OUT_DIR}/overlays/miniuart-bt.dtbo" \
  "${OUT_DIR}/RPI_EFI.fd" "${OUT_DIR}/cmdline.txt"
rm -rf "${OUT_DIR}/EFI"

unzip -q -o "${BUNDLE_PATH}" \
  start4.elf \
  fixup4.dat \
  bcm2711-rpi-4-b.dtb \
  overlays/miniuart-bt.dtbo \
  -d "${TMP_DIR}"

cp "${TMP_DIR}/start4.elf" "${OUT_DIR}/start4.elf"
cp "${TMP_DIR}/fixup4.dat" "${OUT_DIR}/fixup4.dat"
cp "${TMP_DIR}/bcm2711-rpi-4-b.dtb" "${OUT_DIR}/bcm2711-rpi-4-b.dtb"
cp "${TMP_DIR}/overlays/miniuart-bt.dtbo" "${OUT_DIR}/overlays/miniuart-bt.dtbo"
cp "${CONFIG_SRC}" "${OUT_DIR}/config.txt"
cp "${IMAGE_PATH}" "${OUT_DIR}/kernel8.img"

if find "${OUT_DIR}" -maxdepth 2 -type f \( -name 'RPI_EFI.fd' -o -name 'BOOTAA64.EFI' -o -name 'cmdline.txt' \) | grep -q .; then
  echo "[pi4-phase3-stage] Forbidden UEFI file detected in direct-boot SD tree." >&2
  exit 1
fi

cat > "${README_FILE}" <<'EOF'
UniSplit Raspberry Pi 4 Phase-3 SD Card Tree
===========================================

This directory is the direct-boot Phase-3 SD layout built from the frozen
Phase-2 hardware-pass baseline. The locked config.txt is unchanged.

Required files:
- config.txt
- kernel8.img
- start4.elf
- fixup4.dat
- bcm2711-rpi-4-b.dtb
- overlays/miniuart-bt.dtbo

Forbidden files for this phase:
- RPI_EFI.fd
- EFI/
- BOOTAA64.EFI
- cmdline.txt

Expected UART markers for the first Phase-3 retest:
- UNISPLIT_RPI4_P1_ASM_ENTRY
- UNISPLIT_RPI4_P1_POST_MMU
- UNISPLIT_RPI4_P2_PRE_BOOTINFO_CALL
- UNISPLIT_RPI4_P2_POST_BOOTINFO
- UNISPLIT_RPI4_P2_UKPLAT_ENTRY
- UNISPLIT_RPI4_P3_MAIN_ENTRY
- UNISPLIT_RPI4_P3_SELFTEST_START
- UNISPLIT_RPI4_P3_SELFTEST_PASS
- UNISPLIT_RPI4_P3_SELFTEST_DONE
EOF

(
  cd "${OUT_DIR}"
  find . -type f | sed 's#^\./##' | sort > .sha_manifest.tmp
  sha256sum $(cat .sha_manifest.tmp) > "${SHA_FILE}"
  rm -f .sha_manifest.tmp
)

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "bundle_path=${BUNDLE_PATH}"
  echo "lock_file=${LOCK_FILE}"
  echo "kernel_image=${IMAGE_PATH}"
  echo "sdcard_dir=${OUT_DIR}"
  echo "required_files=config.txt,kernel8.img,start4.elf,fixup4.dat,bcm2711-rpi-4-b.dtb,overlays/miniuart-bt.dtbo"
  echo "forbidden_files=RPI_EFI.fd,EFI/,BOOTAA64.EFI,cmdline.txt"
  echo "sha256_sums=${SHA_FILE}"
} > "${MANIFEST_FILE}"

echo "[ok] Pi4 phase-3 SD card tree prepared."
echo "[dir ] ${OUT_DIR}"
echo "[file] ${MANIFEST_FILE}"
