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
req_path "edge_native/unikraft_edge_selftest/Kraftfile"
req_path "edge_native/unikraft_edge_selftest/Config.uk"
req_path "edge_native/unikraft_edge_selftest/Makefile.uk"
req_path "edge_native/unikraft_edge_selftest/generated/embedded_model.c"
req_path "edge_native/unikraft_edge_selftest/generated/embedded_model.h"
req_path "scripts/build_pi_image.sh"
req_path "scripts/prepare_pi_boot_media.sh"
req_path "docs/protocol.md"
req_path "docs/pre_pi_validation_checklist.md"
req_path "configs/pi_edge_runtime.env.example"
req_path "configs/pi_boot/config.txt"
req_path "configs/pi_boot/cmdline.txt.template"

echo "[check] Verifying exported split artifacts..."
for split in 0 3 6 7 8 9; do
  req_split_artifacts "${split}"
done

echo "[check] Generating Pi readiness manifest..."
mkdir -p "${LATEST_DIR}"
"${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/generate_pi_readiness_manifest.py" \
  --repo-root "${ROOT_DIR}" \
  --output "artifacts/pi_handoff/latest/pi_readiness_manifest.json"

echo "[check] Building deterministic Pi image candidate and boot-media layout..."
"${ROOT_DIR}/scripts/prepare_pi_boot_media.sh"

echo "[check] Checking generated validation evidence..."
req_path "artifacts/roundtrip/latest/summary.json"
req_path "artifacts/qemu/unikraft_edge_selftest_arm64.log"
req_path "artifacts/prepi/validation_report.txt"
req_path "artifacts/pi_handoff/latest/images/kernel8.img"
req_path "artifacts/pi_handoff/latest/boot_media/boot/kernel8.img"
req_path "artifacts/pi_handoff/latest/boot_media/boot/config.txt"
req_path "artifacts/pi_handoff/latest/boot_media/boot/cmdline.txt"

echo "[ok] Pi readiness checks passed."
echo "[manifest] ${MANIFEST_PATH}"
