#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${ROOT_DIR}/edge_native/unikraft_hello"
LOG_DIR="${ROOT_DIR}/artifacts/qemu"
LOG_FILE="${LOG_DIR}/unikraft_hello_arm64.log"

mkdir -p "${LOG_DIR}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required tool: $1" >&2
    exit 1
  }
}

need_cmd qemu-system-aarch64
need_cmd aarch64-linux-gnu-gcc
need_cmd kraft
need_cmd flex
need_cmd bison
need_cmd socat

echo "[check] QEMU accel support:"
qemu-system-aarch64 -accel help

echo "[build] Building Unikraft hello for qemu/arm64..."
kraft --no-prompt --log-type basic build --plat qemu --arch arm64 "${APP_DIR}"

echo "[run] Booting in emulation-only mode (TCG)..."
set +e
timeout 30s kraft --no-prompt --log-type basic run --plat qemu --arch arm64 --disable-acceleration "${APP_DIR}" \
  2>&1 | tee "${LOG_FILE}"
RC=${PIPESTATUS[0]}
set -e

if [[ ${RC} -ne 0 && ${RC} -ne 124 ]]; then
  echo "Run failed with exit code ${RC}" >&2
  exit ${RC}
fi

if ! grep -Eiq "hello|world" "${LOG_FILE}"; then
  echo "Did not detect hello-world output in ${LOG_FILE}" >&2
  exit 1
fi

echo "[ok] T01 QEMU ARM64 unikernel boot validation passed (correctness mode)."
echo "[note] Do not use this run for performance/timing claims."
