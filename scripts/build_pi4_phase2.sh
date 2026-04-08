#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FROZEN_ROOT="${ROOT_DIR}/artifacts/pi_phase1/frozen-hwpass-20260408"
SOURCE_TREE="${FROZEN_ROOT}/source_tree/unikraft"
APP_DIR="${ROOT_DIR}/edge_native/unikraft_pi4_phase2_bootrestore"
UK_DIR="${APP_DIR}/.unikraft/unikraft"
BUILD_DIR="${APP_DIR}/.unikraft/build"
APP_CONFIG="${APP_DIR}/.config.pi4-phase2"
DEFCONFIG="${ROOT_DIR}/configs/pi_phase2/unisplit_pi4_phase2_bootrestore.defconfig"
PATCH_FILE="${ROOT_DIR}/configs/patches/unikraft_pi4_phase2_bootrestore.patch"
ARTIFACT_ROOT="${ROOT_DIR}/artifacts/pi_phase2/latest"
IMAGES_DIR="${ARTIFACT_ROOT}/images"
AUDIT_FILE="${ARTIFACT_ROOT}/pi4_phase2_build_audit.txt"
META_FILE="${IMAGES_DIR}/kernel8_metadata.txt"
MODE="build"
JOBS="${UNISPLIT_PI4_JOBS:-$(nproc)}"

usage() {
  cat <<'EOF' >&2
Usage: scripts/build_pi4_phase2.sh [--audit-only]
EOF
  exit 2
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[pi4-phase2] Missing required tool: $1" >&2
    exit 1
  }
}

if [[ $# -gt 1 ]]; then
  usage
fi
if [[ $# -eq 1 ]]; then
  case "$1" in
    --audit-only)
      MODE="audit"
      ;;
    *)
      usage
      ;;
  esac
fi

need_cmd aarch64-linux-gnu-gcc
need_cmd aarch64-linux-gnu-objcopy
need_cmd make
need_cmd patch
need_cmd python3
need_cmd rg
need_cmd sha256sum
need_cmd strings

mkdir -p "${ARTIFACT_ROOT}" "${APP_DIR}/.unikraft"

if [[ ! -d "${SOURCE_TREE}" ]]; then
  echo "[pi4-phase2] Missing frozen source snapshot: ${SOURCE_TREE}" >&2
  exit 1
fi

if [[ ! -f "${FROZEN_ROOT}/FROZEN_BASELINE.txt" ]]; then
  echo "[pi4-phase2] Missing frozen baseline metadata: ${FROZEN_ROOT}/FROZEN_BASELINE.txt" >&2
  exit 1
fi

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "[pi4-phase2] Missing Phase 2 patch file: ${PATCH_FILE}" >&2
  exit 1
fi

rm -rf "${UK_DIR}" "${BUILD_DIR}"
cp -a "${SOURCE_TREE}" "${UK_DIR}"

if ! patch -p1 --dry-run -d "${UK_DIR}" < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[pi4-phase2] Could not apply Phase 2 patch cleanly in ${UK_DIR}: ${PATCH_FILE}" >&2
  exit 1
fi
patch -p1 -d "${UK_DIR}" < "${PATCH_FILE}" >/dev/null

cp "${DEFCONFIG}" "${APP_CONFIG}"

echo "[pi4-phase2] Configuring Pi4 phase-2 build..."
make -C "${UK_DIR}" \
  A="${APP_DIR}" \
  O="${BUILD_DIR}" \
  C="${APP_CONFIG}" \
  CROSS_COMPILE=aarch64-linux-gnu- \
  olddefconfig >/dev/null

echo "[pi4-phase2] Building Pi4 phase-2 image..."
make -C "${UK_DIR}" \
  A="${APP_DIR}" \
  O="${BUILD_DIR}" \
  C="${APP_CONFIG}" \
  CROSS_COMPILE=aarch64-linux-gnu- \
  -j"${JOBS}" \
  all >/dev/null

CFG_FILE="${BUILD_DIR}/config"
if [[ ! -f "${CFG_FILE}" ]]; then
  echo "[pi4-phase2] Missing build config: ${CFG_FILE}" >&2
  exit 1
fi

required_cfg=(
  '^CONFIG_PLAT_KVM=y$'
  '^CONFIG_KVM_VMM_RPI4=y$'
  '^CONFIG_KVM_BOOT_PROTO_LXBOOT=y$'
  '^CONFIG_ARCH_ARM_64=y$'
  '^CONFIG_MCPU_ARM64_CORTEX_A72=y$'
  '^CONFIG_LIBPL011=y$'
  '^CONFIG_LIBPL011_EARLY_CONSOLE=y$'
  '^CONFIG_LIBUKINTCTLR_GICV2=y$'
  '^CONFIG_APPPI4PHASE2_BOOTRESTORE=y$'
)

for pattern in "${required_cfg[@]}"; do
  if ! rg -n "${pattern}" "${CFG_FILE}" >/dev/null; then
    echo "[pi4-phase2] Missing required config: ${pattern}" >&2
    exit 1
  fi
done

rejected_cfg=(
  '^CONFIG_KVM_VMM_QEMU=y$'
  '^CONFIG_KVM_VMM_FIRECRACKER=y$'
  '^CONFIG_KVM_BOOT_PROTO_EFI_STUB=y$'
  '^CONFIG_APPPI4PHASE1_BOOTINFO_BYPASS=y$'
)

for pattern in "${rejected_cfg[@]}"; do
  if rg -n "${pattern}" "${CFG_FILE}" >/dev/null; then
    echo "[pi4-phase2] Rejected config present: ${pattern}" >&2
    exit 1
  fi
done

RPI4_CMD="$(find "${BUILD_DIR}" -type f -name 'rpi4_bpt64.arm.o.cmd' | head -n 1)"
if [[ -z "${RPI4_CMD}" ]]; then
  echo "[pi4-phase2] Missing Pi4 page-table build command in ${BUILD_DIR}" >&2
  exit 1
fi

if find "${BUILD_DIR}" -type f -name 'qemu_bpt64.arm.o.cmd' | grep -q .; then
  echo "[pi4-phase2] QEMU page-table object detected in Pi4 build output." >&2
  exit 1
fi

if find "${BUILD_DIR}" -maxdepth 1 -type f \( -name '*.efi' -o -name '*BOOTAA64*' \) | grep -q .; then
  echo "[pi4-phase2] EFI artifacts detected in direct-boot build output." >&2
  exit 1
fi

IMG_CMD_PATH="$(find "${BUILD_DIR}" -maxdepth 1 -type f -name '*_kvm-arm64.img.cmd' | sort | head -n 1)"
BIN_CMD_PATH="$(find "${BUILD_DIR}" -maxdepth 1 -type f -name '*_kvm-arm64.bin.cmd' | sort | head -n 1)"
IMG_PATH="$(find "${BUILD_DIR}" -maxdepth 1 -type f -name '*_kvm-arm64' | sort | head -n 1)"

if [[ -z "${IMG_PATH}" ]]; then
  echo "[pi4-phase2] Missing Pi4 base image (*_kvm-arm64) in ${BUILD_DIR}" >&2
  exit 1
fi

if [[ -z "${IMG_CMD_PATH}" || -z "${BIN_CMD_PATH}" ]]; then
  echo "[pi4-phase2] Missing Linux direct-boot command traces (*.bin.cmd / *.img.cmd)." >&2
  exit 1
fi

required_markers=(
  'UNISPLIT_RPI4_P1_ASM_ENTRY'
  'UNISPLIT_RPI4_P1_POST_MMU'
  'UNISPLIT_RPI4_P2_PRE_BOOTINFO_CALL'
  'UNISPLIT_RPI4_P2_POST_BOOTINFO'
  'UNISPLIT_RPI4_P2_UKPLAT_ENTRY'
  'UNISPLIT_RPI4_P2_MAIN_ENTRY'
  'UNISPLIT_RPI4_P2_MAIN_DONE'
  'UNISPLIT_RPI4_P1_BOOTINFO_ENTRY'
  'UNISPLIT_RPI4_P1_DTB_PTR=0x'
  'UNISPLIT_RPI4_P1_FDT_MRD_ENTRY'
  'UNISPLIT_RPI4_P1_BOOTINFO_CRASH:'
)

for marker in "${required_markers[@]}"; do
  if ! strings "${IMG_PATH}" | grep -F "${marker}" >/dev/null; then
    echo "[pi4-phase2] Missing required marker ${marker} in ${IMG_PATH}" >&2
    exit 1
  fi
done

python3 - "${IMG_PATH}" <<'PY'
import pathlib
import sys

img = pathlib.Path(sys.argv[1]).read_bytes()
if len(img) < 64:
    raise SystemExit("image too small for Linux arm64 boot header")

if img[:4] != bytes.fromhex("4d5a0091"):
    raise SystemExit("missing Linux arm64 CODE0 header at offset 0")

if img[56:60] != b"ARMd":
    raise SystemExit("missing Linux arm64 magic at offset 56")
PY

SOURCE_AUDIT_LINES=(
  "frozen_source_root=${FROZEN_ROOT}"
  "source_tree=${SOURCE_TREE}"
  "patched_tree=${UK_DIR}"
  "patch_file=${PATCH_FILE}"
  "has_kvm_vmm_rpi4=$(rg -n 'config KVM_VMM_RPI4' "${UK_DIR}/plat/kvm/Config.uk" >/dev/null && echo yes || echo no)"
  "has_rpi4_bpt64=$([[ -f "${UK_DIR}/plat/kvm/arm/rpi4_bpt64.S" ]] && echo yes || echo no)"
  "has_rpi4_ram_base=$(rg -n '#define RAM_BASE_ADDR 0x00000000' "${UK_DIR}/plat/kvm/include/kvm-arm64/image.h" >/dev/null && echo yes || echo no)"
)

BUILD_SHA="$(sha256sum "${IMG_PATH}" | awk '{print $1}')"
BUILD_SIZE="$(wc -c < "${IMG_PATH}" | tr -d '[:space:]')"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "mode=${MODE}"
  for line in "${SOURCE_AUDIT_LINES[@]}"; do
    echo "${line}"
  done
  echo "config_file=${CFG_FILE}"
  echo "image_path=${IMG_PATH}"
  echo "image_sha256=${BUILD_SHA}"
  echo "image_size_bytes=${BUILD_SIZE}"
  echo "linux_header_command=${IMG_CMD_PATH}"
  echo "raw_binary_command=${BIN_CMD_PATH}"
  echo "rpi4_page_table_cmd=${RPI4_CMD}"
  echo "phase1_frozen_baseline=${FROZEN_ROOT}/FROZEN_BASELINE.txt"
  echo "markers=UNISPLIT_RPI4_P1_ASM_ENTRY,UNISPLIT_RPI4_P1_POST_MMU,UNISPLIT_RPI4_P2_PRE_BOOTINFO_CALL,UNISPLIT_RPI4_P1_BOOTINFO_ENTRY,UNISPLIT_RPI4_P1_DTB_PTR=0x,UNISPLIT_RPI4_P1_FDT_MRD_ENTRY,UNISPLIT_RPI4_P1_BOOTINFO_CRASH:,UNISPLIT_RPI4_P2_POST_BOOTINFO,UNISPLIT_RPI4_P2_UKPLAT_ENTRY,UNISPLIT_RPI4_P2_MAIN_ENTRY,UNISPLIT_RPI4_P2_MAIN_DONE"
} > "${AUDIT_FILE}"

if [[ "${MODE}" == "build" ]]; then
  mkdir -p "${IMAGES_DIR}"
  cp "${IMG_PATH}" "${IMAGES_DIR}/kernel8.img"
  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "source_image=${IMG_PATH}"
    echo "staged_image=${IMAGES_DIR}/kernel8.img"
    echo "sha256=${BUILD_SHA}"
    echo "build_config=${CFG_FILE}"
    echo "audit_file=${AUDIT_FILE}"
    echo "frozen_source_root=${FROZEN_ROOT}"
  } > "${META_FILE}"

  echo "[ok] Pi4 phase-2 build completed."
  echo "[img ] ${IMAGES_DIR}/kernel8.img"
else
  echo "[ok] Pi4 phase-2 audit completed."
fi

echo "[file] ${AUDIT_FILE}"
