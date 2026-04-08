#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART_DIR="${ROOT_DIR}/artifacts/prepi"
REPORT="${ART_DIR}/validation_report.txt"
mkdir -p "${ART_DIR}"

run_step() {
  local title="$1"
  local cmd="$2"
  {
    echo "============================================================"
    echo "[STEP] ${title}"
    echo "[CMD ] ${cmd}"
  } | tee -a "${REPORT}"
  if bash -lc "cd '${ROOT_DIR}' && ${cmd}" >>"${REPORT}" 2>&1; then
    echo "[PASS] ${title}" | tee -a "${REPORT}"
  else
    echo "[FAIL] ${title}" | tee -a "${REPORT}"
    return 1
  fi
}

: > "${REPORT}"
echo "Pre-Pi Validation Report" | tee -a "${REPORT}"
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "${REPORT}"
echo "Git SHA: $(git -C "${ROOT_DIR}" rev-parse --short HEAD || echo unknown)" | tee -a "${REPORT}"

run_step "Export all edge C artifacts" "make export-edge-c-all"
run_step "Build edge-native runtime binaries" "make c-edge-build"
run_step "Forward parity checks (all splits)" "make c-edge-forward-verify-all"
run_step "Quantization parity checks" "make c-edge-quant-verify"
run_step "Controller sanity checks" "make c-edge-controller-verify"
run_step "Failure-path hardening checks" "make c-edge-failure-verify"
run_step "VPS roundtrip matrix evidence" "make c-edge-roundtrip-vps"
run_step "Unikraft edge selftest validation" "make uk-edge-validate"
run_step "Pi UEFI tooling/repo checks" "make pi-uefi-check"

if [[ -n "${UNISPLIT_PI_UEFI_BUNDLE:-}" && -f "${UNISPLIT_PI_UEFI_BUNDLE}" ]]; then
  run_step "Pi UEFI full handoff assembly" "make pi-uefi-handoff"
else
  {
    echo "============================================================"
    echo "[SKIP] Pi UEFI full handoff assembly"
    echo "[REASON] UNISPLIT_PI_UEFI_BUNDLE not set to a local pftf/RPi4 firmware zip."
    echo "[TIP] Set UNISPLIT_PI_UEFI_BUNDLE and rerun to validate full SD handoff output."
  } | tee -a "${REPORT}"
fi

echo "============================================================" | tee -a "${REPORT}"
echo "[PASS] prepi-validate complete" | tee -a "${REPORT}"
echo "[REPORT] ${REPORT}" | tee -a "${REPORT}"
