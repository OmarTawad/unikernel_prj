#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATEST_DIR="${ROOT_DIR}/artifacts/pi_handoff/latest"
REPORT_FILE="${LATEST_DIR}/pi_nonkvm_discovery_report.txt"
TMP_DIR="${LATEST_DIR}/tmp_discovery"
APP_POF="edge_native/unikraft_pi_uart_pof"
APP_HELLO="edge_native/unikraft_hello"

mkdir -p "${LATEST_DIR}"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

run_cmd() {
  local name="$1"
  shift
  local cmd="$*"
  local out_file="${TMP_DIR}/${name}.out"
  local rc_file="${TMP_DIR}/${name}.rc"

  set +e
  bash -lc "${cmd}" >"${out_file}" 2>&1
  local rc=$?
  set -e
  echo "${rc}" > "${rc_file}"
}

collect_cmd() {
  local name="$1"
  echo "command=${name}"
  echo "exit_code=$(cat "${TMP_DIR}/${name}.rc")"
  echo "stdout_stderr_begin"
  sed -n '1,120p' "${TMP_DIR}/${name}.out"
  echo "stdout_stderr_end"
}

run_cmd "kraft_version" "cd '${ROOT_DIR}' && kraft version"
run_cmd "plat_dirs" "cd '${ROOT_DIR}' && find '${APP_POF}/.unikraft/unikraft/plat' -maxdepth 2 -type d | sort"
run_cmd "pof_build_efi_arm64" "cd '${ROOT_DIR}' && kraft --no-prompt --log-type basic build --plat efi --arch arm64 '${APP_POF}'"
run_cmd "hello_build_efi_arm64" "cd '${ROOT_DIR}' && kraft --no-prompt --log-type basic build --plat efi --arch arm64 '${APP_HELLO}'"
run_cmd "pof_build_qemu_arm64" "cd '${ROOT_DIR}' && kraft --no-prompt --log-type basic build --plat qemu --arch arm64 '${APP_POF}'"

viable_path=""
if [[ "$(cat "${TMP_DIR}/pof_build_efi_arm64.rc")" == "0" ]]; then
  viable_path="pof: --plat efi --arch arm64"
elif [[ "$(cat "${TMP_DIR}/hello_build_efi_arm64.rc")" == "0" ]]; then
  viable_path="hello: --plat efi --arch arm64"
fi

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "objective=discover first true non-KVM Pi/UEFI-capable minimal payload path"
  echo "repo_root=${ROOT_DIR}"
  if [[ -n "${viable_path}" ]]; then
    echo "status=found_viable_nonkvm_path"
    echo "viable_path=${viable_path}"
    echo "next_command=use discovered efi/arm64 path to produce BOOTAA64.EFI from minimal payload"
  else
    echo "status=hard_blocked"
    echo "blocker=no true non-KVM efi/arm64 target path found in current local Kraft/Unikraft setup"
    echo "next_command=confirm/install toolchain/platform support that exposes a real non-KVM efi/arm64 target"
  fi
  echo
  echo "[attempt_1]"
  collect_cmd "pof_build_efi_arm64"
  echo
  echo "[attempt_2]"
  collect_cmd "hello_build_efi_arm64"
  echo
  echo "[reference]"
  collect_cmd "pof_build_qemu_arm64"
  echo
  echo "[environment]"
  collect_cmd "kraft_version"
  echo
  echo "[plat_tree]"
  collect_cmd "plat_dirs"
} > "${REPORT_FILE}"

echo "[pi-uefi-discovery] Report: ${REPORT_FILE}"
if [[ -n "${viable_path}" ]]; then
  echo "[pi-uefi-discovery] Found viable path: ${viable_path}"
else
  echo "[pi-uefi-discovery] Hard blocked: no viable non-KVM path detected."
fi
