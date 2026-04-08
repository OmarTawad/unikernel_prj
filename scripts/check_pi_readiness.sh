#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART_ROOT="${ROOT_DIR}/artifacts/pi_handoff"
LATEST_DIR="${ART_ROOT}/latest"
MANIFEST_PATH="${LATEST_DIR}/pi_readiness_manifest.json"

req_path() {
  local p="$1"
  if [[ ! -e "${ROOT_DIR}/${p}" ]]; then
    echo "[FAIL] Missing required path: ${p}" >&2
    exit 1
  fi
}

req_split_artifacts() {
  local split="$1"
  local base="${ROOT_DIR}/edge_native/artifacts/c_splits/edge_k${split}"
  if [[ ! -d "${base}" ]]; then
    echo "[FAIL] Missing split artifact dir: ${base}" >&2
    exit 1
  fi
  if [[ ! -f "${base}/manifest.json" ]]; then
    echo "[FAIL] Missing manifest: ${base}/manifest.json" >&2
    exit 1
  fi
}

echo "[check] Regenerating embedded model source..."
make -C "${ROOT_DIR}" uk-edge-embed-artifacts >/dev/null

echo "[check] Verifying core Pi-phase inputs..."
req_path "edge_native/runtime/include/transport_backend.h"
req_path "edge_native/runtime/src/transport_backend_factory.c"
req_path "edge_native/runtime/src/transport_lwip_backend.c"
req_path "edge_native/unikraft_pi_uart_pof/Kraftfile"
req_path "edge_native/unikraft_pi_uart_pof/Config.uk"
req_path "edge_native/unikraft_pi_uart_pof/Makefile.uk"
req_path "edge_native/unikraft_pi_uart_pof/main.c"
req_path "configs/pi_uefi_bundle.lock.json"
req_path "scripts/stage_pi_uefi_bundle.sh"
req_path "scripts/build_pi_uefi_payload.sh"
req_path "scripts/prepare_pi_uefi_boot_media.sh"
req_path "docs/protocol.md"
req_path "docs/pre_pi_validation_checklist.md"
req_path "docs/raspberry_pi_handoff.md"

echo "[check] Verifying exported split artifacts..."
for split in 0 3 6 7 8 9; do
  req_split_artifacts "${split}"
done

echo "[check] Generating Pi readiness manifest..."
mkdir -p "${LATEST_DIR}"
"${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/generate_pi_readiness_manifest.py" \
  --repo-root "${ROOT_DIR}" \
  --output "artifacts/pi_handoff/latest/pi_readiness_manifest.json"

echo "[check] Verifying Pi UEFI tooling target sanity..."
make -C "${ROOT_DIR}" pi-uefi-check >/dev/null

echo "[check] Checking generated validation evidence..."
req_path "artifacts/roundtrip/latest/summary.json"
req_path "artifacts/qemu/unikraft_edge_selftest_arm64.log"
req_path "artifacts/prepi/validation_report.txt"

if [[ -n "${UNISPLIT_PI_UEFI_BUNDLE:-}" && -f "${UNISPLIT_PI_UEFI_BUNDLE}" ]]; then
  echo "[check] Building full Pi UEFI handoff outputs..."
  make -C "${ROOT_DIR}" pi-uefi-handoff >/dev/null
  req_path "artifacts/pi_handoff/latest/images/pi_uefi_payload_metadata.txt"
  req_path "artifacts/pi_handoff/latest/images/unikraft_pi_uart_pof_BOOTAA64.EFI"
  req_path "artifacts/pi_handoff/latest/boot_media_uefi/RPI_EFI.fd"
  req_path "artifacts/pi_handoff/latest/boot_media_uefi/start4.elf"
  req_path "artifacts/pi_handoff/latest/boot_media_uefi/config.txt"
  req_path "artifacts/pi_handoff/latest/boot_media_uefi/EFI/BOOT/BOOTAA64.EFI"
  req_path "artifacts/pi_handoff/latest/boot_media_uefi/BOOT_MEDIA_README.txt"
  req_path "artifacts/pi_handoff/latest/boot_media_uefi/boot_media_manifest.txt"
  req_path "artifacts/pi_handoff/latest/boot_media_uefi/SHA256SUMS.txt"
else
  echo "[check] Skipping full Pi UEFI handoff output checks (UNISPLIT_PI_UEFI_BUNDLE not set)."
fi

echo "[ok] Pi readiness checks passed."
echo "[manifest] ${MANIFEST_PATH}"
