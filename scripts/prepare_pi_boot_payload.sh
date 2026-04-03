#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${ROOT_DIR}/artifacts/pi_handoff"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${OUT_ROOT}/${STAMP}"
PAYLOAD_DIR="${RUN_DIR}/pi_boot_payload"
PAYLOAD_TAR="${RUN_DIR}/pi_boot_payload.tar.gz"
LATEST_DIR="${OUT_ROOT}/latest"

mkdir -p "${PAYLOAD_DIR}"

echo "[prepare] Refreshing exported C artifacts..."
make -C "${ROOT_DIR}" export-edge-c-all >/dev/null

echo "[prepare] Building readiness manifest..."
"${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/generate_pi_readiness_manifest.py" \
  --repo-root "${ROOT_DIR}" \
  --output "artifacts/pi_handoff/${STAMP}/pi_readiness_manifest.json"

echo "[prepare] Assembling Pi boot payload layout..."
mkdir -p "${PAYLOAD_DIR}/c_splits"
cp -a "${ROOT_DIR}/edge_native/artifacts/c_splits/." "${PAYLOAD_DIR}/c_splits/"
cp "${ROOT_DIR}/configs/pi_edge_runtime.env.example" "${PAYLOAD_DIR}/"
cp "${ROOT_DIR}/docs/protocol.md" "${PAYLOAD_DIR}/"
cp "${ROOT_DIR}/docs/pre_pi_validation_checklist.md" "${PAYLOAD_DIR}/"
cp "${RUN_DIR}/pi_readiness_manifest.json" "${PAYLOAD_DIR}/"

echo "[prepare] Creating tarball..."
tar -czf "${PAYLOAD_TAR}" -C "${PAYLOAD_DIR}" .

mkdir -p "${LATEST_DIR}"
find "${LATEST_DIR}" -maxdepth 1 -type l -delete
cp "${RUN_DIR}/pi_readiness_manifest.json" "${LATEST_DIR}/pi_readiness_manifest.json"
cp "${PAYLOAD_TAR}" "${LATEST_DIR}/pi_boot_payload.tar.gz"
printf "%s\n" "${RUN_DIR}" > "${LATEST_DIR}/SOURCE_RUN_DIR.txt"

echo "[ok] Pi boot payload prepared."
echo "[dir ] ${RUN_DIR}"
echo "[file] ${PAYLOAD_TAR}"
