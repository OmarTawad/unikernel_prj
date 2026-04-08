#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${ROOT_DIR}/edge_native/unikraft_pi_uart_pof"
LATEST_DIR="${ROOT_DIR}/artifacts/pi_handoff/latest"
BUILD_CFG="${APP_DIR}/.unikraft/build/config"
FALLBACK_CFG="${APP_DIR}/.config.unisplit-uk-pi-uart-pof_qemu-arm64"
KRAFTFILE="${APP_DIR}/Kraftfile"
META_FILE="${LATEST_DIR}/images/pi_uefi_payload_metadata.txt"
REPORT_FILE="${LATEST_DIR}/pi_payload_model_blocker_report.txt"

mkdir -p "${LATEST_DIR}"

CFG_FILE="${BUILD_CFG}"
if [[ ! -f "${CFG_FILE}" ]]; then
  CFG_FILE="${FALLBACK_CFG}"
fi

if [[ ! -f "${CFG_FILE}" ]]; then
  echo "[pi-uefi-gate] Missing active config: ${BUILD_CFG} (or fallback ${FALLBACK_CFG})" >&2
  exit 2
fi

if [[ ! -f "${KRAFTFILE}" ]]; then
  echo "[pi-uefi-gate] Missing Kraftfile: ${KRAFTFILE}" >&2
  exit 2
fi

status="accepted"
reason="payload model appears non-KVM and non-QEMU-derived"

cfg_line() {
  local key="$1"
  rg -n "^${key}=" "${CFG_FILE}" || true
}

kraft_line() {
  rg -n "platform:\\s*" "${KRAFTFILE}" || true
}

meta_line() {
  local key="$1"
  if [[ -f "${META_FILE}" ]]; then
    rg -n "^${key}=" "${META_FILE}" || true
  fi
}

reject_reasons=()
if rg -n '^CONFIG_PLAT_KVM=y$' "${CFG_FILE}" >/dev/null; then
  reject_reasons+=("CONFIG_PLAT_KVM=y")
fi
if rg -n '^CONFIG_KVM_VMM_QEMU=y$' "${CFG_FILE}" >/dev/null; then
  reject_reasons+=("CONFIG_KVM_VMM_QEMU=y")
fi
if rg -n 'platform:\s*qemu' "${KRAFTFILE}" >/dev/null; then
  reject_reasons+=("Kraftfile target platform=qemu")
fi

if [[ ${#reject_reasons[@]} -gt 0 ]]; then
  status="rejected"
  reason="valid PE/EFI but runtime model is KVM/QEMU-derived"
fi

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "status=${status}"
  echo "reason=${reason}"
  echo "config_file=${CFG_FILE}"
  echo "kraftfile=${KRAFTFILE}"
  echo "metadata_file=${META_FILE}"
  echo
  echo "[matched_rejection_lines]"
  cfg_line "CONFIG_PLAT_KVM"
  cfg_line "CONFIG_KVM_VMM_QEMU"
  cfg_line "CONFIG_KVM_BOOT_PROTO_EFI_STUB"
  cfg_line "CONFIG_KVM_BOOT_PROTO_EFI_STUB_CMDLINE_FNAME"
  cfg_line "CONFIG_KVM_BOOT_PROTO_EFI_STUB_INITRD_FNAME"
  cfg_line "CONFIG_KVM_BOOT_PROTO_EFI_STUB_DTB_FNAME"
  echo
  echo "[kraftfile_targets]"
  kraft_line
  echo
  echo "[metadata_snapshot]"
  meta_line "platform"
  meta_line "arch"
  meta_line "boot_protocol"
  meta_line "source_payload"
} > "${REPORT_FILE}"

if [[ "${status}" == "rejected" ]]; then
  echo "[pi-uefi-gate] REJECTED: ${reason}" >&2
  echo "[pi-uefi-gate] Report: ${REPORT_FILE}" >&2
  exit 1
fi

echo "[pi-uefi-gate] ACCEPTED"
echo "[pi-uefi-gate] Report: ${REPORT_FILE}"
