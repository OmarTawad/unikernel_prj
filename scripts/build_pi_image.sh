#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${ROOT_DIR}/edge_native/unikraft_edge_selftest"
OUT_DIR="${ROOT_DIR}/artifacts/pi_handoff/latest/images"
APP_NAME="unisplit-uk-edge-selftest"
ARCH="arm64"

# Current Kraft toolchain on this repo exposes qemu/fc/xen targets.
# Use qemu by default to produce deterministic AArch64 kernel image candidate.
PI_PLAT="${UNISPLIT_PI_KRAFT_PLAT:-qemu}"

mkdir -p "${OUT_DIR}"

echo "[pi-image] Refreshing embedded model source from exported artifacts..."
make -C "${ROOT_DIR}" uk-edge-embed-artifacts >/dev/null

echo "[pi-image] Cleaning previous Unikraft build output..."
kraft --no-prompt --log-type basic clean --proper "${APP_DIR}"

echo "[pi-image] Building app for plat=${PI_PLAT} arch=${ARCH}..."
if ! kraft --no-prompt --log-type basic build --plat "${PI_PLAT}" --arch "${ARCH}" "${APP_DIR}"; then
  echo "[pi-image] Build failed for plat=${PI_PLAT}." >&2
  echo "[pi-image] Tip: this toolchain currently supports qemu/fc/xen. Set UNISPLIT_PI_KRAFT_PLAT accordingly." >&2
  exit 1
fi

BUILD_DIR="${APP_DIR}/.unikraft/build"
KERNEL_SRC="${BUILD_DIR}/${APP_NAME}_${PI_PLAT}-${ARCH}"
DBG_SRC="${BUILD_DIR}/${APP_NAME}_${PI_PLAT}-${ARCH}.dbg"
BOOTINFO_SRC="${BUILD_DIR}/${APP_NAME}_${PI_PLAT}-${ARCH}.bootinfo"

if [[ ! -f "${KERNEL_SRC}" ]]; then
  echo "[pi-image] Expected kernel output not found: ${KERNEL_SRC}" >&2
  exit 1
fi

cp "${KERNEL_SRC}" "${OUT_DIR}/kernel8.img"
cp "${KERNEL_SRC}" "${OUT_DIR}/${APP_NAME}_${PI_PLAT}-${ARCH}.img"
if [[ -f "${DBG_SRC}" ]]; then
  cp "${DBG_SRC}" "${OUT_DIR}/${APP_NAME}_${PI_PLAT}-${ARCH}.dbg"
fi
if [[ -f "${BOOTINFO_SRC}" ]]; then
  cp "${BOOTINFO_SRC}" "${OUT_DIR}/${APP_NAME}_${PI_PLAT}-${ARCH}.bootinfo"
fi

cat > "${OUT_DIR}/image_build_metadata.txt" <<META
timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
platform=${PI_PLAT}
arch=${ARCH}
source_kernel=${KERNEL_SRC}
output_kernel8=${OUT_DIR}/kernel8.img
output_named_image=${OUT_DIR}/${APP_NAME}_${PI_PLAT}-${ARCH}.img
META

echo "[ok] Pi image candidate built."
echo "[file] ${OUT_DIR}/kernel8.img"
echo "[meta] ${OUT_DIR}/image_build_metadata.txt"
