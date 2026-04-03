#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${ROOT_DIR}/edge_native/unikraft_edge_selftest"
LOG_DIR="${ROOT_DIR}/artifacts/qemu"
LOG_FILE="${LOG_DIR}/unikraft_edge_selftest_arm64.log"

mkdir -p "${LOG_DIR}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required tool: $1" >&2
    exit 1
  }
}

need_cmd qemu-system-aarch64
need_cmd kraft

echo "[build] Building Unikraft edge selftest for qemu/arm64..."
kraft --no-prompt --log-type basic clean --proper "${APP_DIR}"
kraft --no-prompt --log-type basic build --plat qemu --arch arm64 "${APP_DIR}"

echo "[run] Booting edge selftest in emulation-only mode (TCG)..."
set +e
timeout 45s kraft --no-prompt --log-type basic run --plat qemu --arch arm64 --disable-acceleration "${APP_DIR}" \
  2>&1 | tee "${LOG_FILE}"
RC=${PIPESTATUS[0]}
set -e

if [[ ${RC} -ne 0 && ${RC} -ne 124 ]]; then
  echo "Run failed with exit code ${RC}" >&2
  exit ${RC}
fi

required_markers=(
  "UK_SELFTEST_EDGE_OK"
  "UK_SELFTEST_SPLIT_OK split=3"
  "UK_SELFTEST_SPLIT_OK split=7"
  "UK_SELFTEST_SPLIT_OK split=8"
  "UK_SELFTEST_CTRL_OK"
  "UK_SELFTEST_TRANSPORT=ukstub"
  "UK_SELFTEST_TRANSPORT_OK"
  "UK_SELFTEST_DONE"
)

for marker in "${required_markers[@]}"; do
  if ! grep -Fq "${marker}" "${LOG_FILE}"; then
    echo "Missing selftest marker in log: ${marker}" >&2
    exit 1
  fi
done

echo "[ok] Unikraft edge selftest markers verified."
echo "[log] ${LOG_FILE}"
