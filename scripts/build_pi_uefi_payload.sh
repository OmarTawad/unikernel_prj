#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${ROOT_DIR}/edge_native/unikraft_pi_uart_pof"
OUT_DIR="${ROOT_DIR}/artifacts/pi_handoff/latest/images"
META_FILE="${OUT_DIR}/pi_uefi_payload_metadata.txt"
ARCH="arm64"
PLAT="qemu"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[pi-uefi-build] Missing required tool: $1" >&2
    exit 1
  }
}

need_cmd kraft
need_cmd find
need_cmd sha256sum
need_cmd python3
need_cmd rg

mkdir -p "${OUT_DIR}"
rm -f "${OUT_DIR}/kernel8.img" "${OUT_DIR}/image_build_metadata.txt"
find "${OUT_DIR}" -maxdepth 1 -type f \( -name '*_qemu-arm64.img' -o -name '*_qemu-arm64.dbg' -o -name '*_qemu-arm64.bootinfo' \) -delete

if [[ ! -f "${APP_DIR}/Kraftfile" ]]; then
  echo "[pi-uefi-build] Missing app Kraftfile: ${APP_DIR}/Kraftfile" >&2
  exit 1
fi

if ! rg -n "platform:\s*qemu" "${APP_DIR}/Kraftfile" >/dev/null 2>&1; then
  echo "[pi-uefi-build] App Kraftfile does not define platform=qemu target required by this Kraft toolchain." >&2
  exit 1
fi

echo "[pi-uefi-build] Cleaning previous app build output..."
if ! kraft --no-prompt --log-type basic clean --proper "${APP_DIR}" >/dev/null 2>&1; then
  echo "[pi-uefi-build] No prior cleanable build output; continuing."
fi

echo "[pi-uefi-build] Forcing EFI-stub KConfig for paper-aligned UEFI payload..."
kraft --no-prompt --log-type basic set --force --plat "${PLAT}" --arch "${ARCH}" \
  -w "${APP_DIR}" \
  CONFIG_OPTIMIZE_PIE=y \
  CONFIG_KVM_BOOT_PROTO_QEMU_VIRT=n \
  CONFIG_KVM_BOOT_PROTO_LXBOOT=n \
  CONFIG_KVM_BOOT_PROTO_EFI_STUB=y

CFG_FILE="$(find "${APP_DIR}" -maxdepth 1 -type f -name '.config.*_qemu-arm64' | head -n 1)"
if [[ -z "${CFG_FILE}" ]]; then
  echo "[pi-uefi-build] Could not locate generated target config for qemu/arm64." >&2
  exit 1
fi
if ! rg -n '^CONFIG_KVM_BOOT_PROTO_EFI_STUB=y$' "${CFG_FILE}" >/dev/null; then
  echo "[pi-uefi-build] EFI stub was not enabled in ${CFG_FILE}." >&2
  exit 1
fi
if ! rg -n '^CONFIG_OPTIMIZE_PIE=y$' "${CFG_FILE}" >/dev/null; then
  echo "[pi-uefi-build] OPTIMIZE_PIE must be enabled for EFI stub in ${CFG_FILE}." >&2
  exit 1
fi

echo "[pi-uefi-build] Building Unikraft payload for plat=${PLAT} arch=${ARCH}..."
kraft --no-prompt --log-type basic build --plat "${PLAT}" --arch "${ARCH}" "${APP_DIR}"

BUILD_DIR="${APP_DIR}/.unikraft/build"
if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "[pi-uefi-build] Build directory not found: ${BUILD_DIR}" >&2
  exit 1
fi

mapfile -t EFI_CANDIDATES < <(
  python3 - "${BUILD_DIR}" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for p in sorted(root.glob("*")):
    if not p.is_file():
        continue
    data = p.read_bytes()
    if len(data) < 0x40 or data[:2] != b"MZ":
        continue
    pe_off = int.from_bytes(data[0x3C:0x40], "little")
    if pe_off + 4 > len(data):
        continue
    if data[pe_off:pe_off + 4] == b"PE\x00\x00":
        print(p.as_posix())
PY
)

if [[ ${#EFI_CANDIDATES[@]} -eq 0 ]]; then
  mapfile -t EFI_CANDIDATES < <(find "${BUILD_DIR}" -maxdepth 1 -type f \( -iname "*.efi" -o -iname "*efi*" \) | sort)
fi

if [[ ${#EFI_CANDIDATES[@]} -eq 0 ]]; then
  echo "[pi-uefi-build] Could not discover EFI payload artifact in ${BUILD_DIR}" >&2
  echo "[pi-uefi-build] Files present:" >&2
  find "${BUILD_DIR}" -maxdepth 1 -type f -printf ' - %f\n' >&2
  exit 1
fi

PAYLOAD_SRC="${EFI_CANDIDATES[0]}"
if [[ ${#EFI_CANDIDATES[@]} -gt 1 ]]; then
  for cand in "${EFI_CANDIDATES[@]}"; do
    if [[ "${cand}" =~ [Bb][Oo][Oo][Tt][Aa][Aa]64\.efi$ ]]; then
      PAYLOAD_SRC="${cand}"
      break
    fi
  done
fi

PAYLOAD_DST="${OUT_DIR}/unikraft_pi_uart_pof_BOOTAA64.EFI"
cp "${PAYLOAD_SRC}" "${PAYLOAD_DST}"

PAYLOAD_SHA="$(sha256sum "${PAYLOAD_DST}" | awk '{print $1}')"

cat > "${META_FILE}" <<META
timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
app_dir=${APP_DIR}
platform=efi
kraft_platform=${PLAT}
boot_protocol=efi_stub
arch=${ARCH}
source_payload=${PAYLOAD_SRC}
output_payload=${PAYLOAD_DST}
output_payload_sha256=${PAYLOAD_SHA}
META

echo "[ok] Built UEFI payload."
echo "[file] ${PAYLOAD_DST}"
echo "[meta] ${META_FILE}"
