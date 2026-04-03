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

echo "[check] Verifying core Pi-phase inputs..."
req_path "edge_native/runtime/include/transport_backend.h"
req_path "edge_native/runtime/src/transport_backend_factory.c"
req_path "edge_native/runtime/src/transport_lwip_backend_stub.c"
req_path "edge_native/unikraft_edge_selftest/Kraftfile"
req_path "edge_native/unikraft_edge_selftest/Config.uk"
req_path "edge_native/unikraft_edge_selftest/Makefile.uk"
req_path "docs/protocol.md"
req_path "docs/pre_pi_validation_checklist.md"
req_path "configs/pi_edge_runtime.env.example"

echo "[check] Verifying exported split artifacts..."
for split in 0 3 6 7 8 9; do
  req_split_artifacts "${split}"
done

echo "[check] Generating Pi readiness manifest..."
mkdir -p "${LATEST_DIR}"
"${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/generate_pi_readiness_manifest.py" \
  --repo-root "${ROOT_DIR}" \
  --output "artifacts/pi_handoff/latest/pi_readiness_manifest.json"

echo "[check] Checking generated validation evidence..."
req_path "artifacts/roundtrip/latest/summary.json"
req_path "artifacts/qemu/unikraft_edge_selftest_arm64.log"
req_path "artifacts/prepi/validation_report.txt"

echo "[ok] Pi readiness checks passed."
echo "[manifest] ${MANIFEST_PATH}"
