#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART_DIR="${ROOT_DIR}/artifacts/roundtrip/latest"
CLOUD_LOG="${ART_DIR}/cloud.log"
SUMMARY_JSON="${ART_DIR}/summary.json"
CLI_BIN="${ROOT_DIR}/edge_native/runtime/build/unisplit_edge_cli"
ARTIFACT_ROOT="${ROOT_DIR}/edge_native/artifacts/c_splits"
SPLITS=(3 7 8)
PORT="${UNISPLIT_VPS_ROUNDTRIP_PORT:-18090}"
BASE_URL="http://127.0.0.1:${PORT}"

mkdir -p "${ART_DIR}"
: > "${CLOUD_LOG}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required tool: $1" >&2
    exit 1
  }
}

need_cmd "${ROOT_DIR}/.venv/bin/python"
need_cmd "${CLI_BIN}"
need_cmd git

echo "[roundtrip] Starting cloud on ${BASE_URL}"
UNISPLIT_CLOUD_HOST=127.0.0.1 \
UNISPLIT_CLOUD_PORT="${PORT}" \
"${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/run_cloud.py" --config "${ROOT_DIR}/configs/cloud.yaml" \
  > "${CLOUD_LOG}" 2>&1 &
CLOUD_PID=$!

cleanup() {
  if kill -0 "${CLOUD_PID}" >/dev/null 2>&1; then
    kill "${CLOUD_PID}" >/dev/null 2>&1 || true
    wait "${CLOUD_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

python_health_check() {
  "${ROOT_DIR}/.venv/bin/python" - <<'PY'
import httpx
import os
import sys
url = os.environ["HEALTH_URL"]
try:
    r = httpx.get(url, timeout=1.0, trust_env=False)
    if r.status_code == 200:
        sys.exit(0)
except Exception:
    pass
sys.exit(1)
PY
}

deadline=$((SECONDS + 60))
while (( SECONDS < deadline )); do
  if ! kill -0 "${CLOUD_PID}" >/dev/null 2>&1; then
    echo "[roundtrip] Cloud process exited early" >&2
    tail -n 120 "${CLOUD_LOG}" >&2 || true
    exit 1
  fi
  if HEALTH_URL="${BASE_URL}/health" python_health_check; then
    echo "[roundtrip] Cloud is healthy"
    break
  fi
  sleep 1
done
if (( SECONDS >= deadline )); then
  echo "[roundtrip] Cloud did not become healthy in time" >&2
  tail -n 120 "${CLOUD_LOG}" >&2 || true
  exit 1
fi

TMP_RESULTS="${ART_DIR}/results.tmp.jsonl"
: > "${TMP_RESULTS}"
ALL_OK=true
for split in "${SPLITS[@]}"; do
  log_file="${ART_DIR}/split_k${split}.log"
  echo "[roundtrip] Running split ${split}"
  set +e
  "${CLI_BIN}" \
    --split-id "${split}" \
    --artifacts-root "${ARTIFACT_ROOT}" \
    --post \
    --cloud-url "${BASE_URL}" \
    --model-version "v0.1.0" \
    > "${log_file}" 2>&1
  rc=$?
  set -e

  ok=false
  if [[ ${rc} -eq 0 ]] && grep -q "CLOUD_OK" "${log_file}" && grep -q "status=ok" "${log_file}"; then
    ok=true
  fi
  py_ok=False
  if [[ "${ok}" == "true" ]]; then
    py_ok=True
  fi

  class_val=""
  if grep -Eq "class=[0-9]+" "${log_file}"; then
    class_val="$(grep -Eo "class=[0-9]+" "${log_file}" | tail -n1 | cut -d= -f2)"
  fi

  "${ROOT_DIR}/.venv/bin/python" - <<PY >> "${TMP_RESULTS}"
import json
print(json.dumps({
    "split_id": ${split},
    "exit_code": ${rc},
    "ok": ${py_ok},
    "class": "${class_val}",
    "log_file": "$(basename "${log_file}")"
}))
PY

  if [[ "${ok}" != "true" ]]; then
    echo "[roundtrip] Split ${split} failed (rc=${rc})" >&2
    tail -n 80 "${log_file}" >&2 || true
    ALL_OK=false
  fi
done

git_sha="$(git -C "${ROOT_DIR}" rev-parse --short HEAD || echo "unknown")"
timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
py_all_ok=False
if [[ "${ALL_OK}" == "true" ]]; then
  py_all_ok=True
fi
"${ROOT_DIR}/.venv/bin/python" - <<PY > "${SUMMARY_JSON}"
import json
from pathlib import Path

tmp_path = Path("${TMP_RESULTS}")
results = [json.loads(line) for line in tmp_path.read_text(encoding="utf-8").splitlines() if line.strip()]
summary = {
    "timestamp": "${timestamp}",
    "base_url": "${BASE_URL}",
    "git_sha": "${git_sha}",
    "splits": [int(x) for x in "${SPLITS[*]}".split()],
    "all_ok": ${py_all_ok},
    "results": results,
}
print(json.dumps(summary, indent=2))
PY

echo "[roundtrip] Wrote ${SUMMARY_JSON}"
cat "${SUMMARY_JSON}"

if [[ "${ALL_OK}" != "true" ]]; then
  exit 1
fi
