#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="${ROOT_DIR}/configs/pi_uefi_bundle.lock.json"
LATEST_DIR="${ROOT_DIR}/artifacts/pi_handoff/latest"
BOOT_MEDIA_DIR="${LATEST_DIR}/boot_media_uefi"
BASELINE_SUMS="${LATEST_DIR}/UEFI_BUNDLE_BASELINE_SHA256SUMS.txt"
BASELINE_FILES="${LATEST_DIR}/UEFI_BUNDLE_BASELINE_FILES.txt"
META_FILE="${LATEST_DIR}/uefi_bundle_stage_metadata.txt"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[pi-uefi-stage] Missing required tool: $1" >&2
    exit 1
  }
}

need_cmd python3
need_cmd unzip
need_cmd sha256sum

if [[ ! -f "${LOCK_FILE}" ]]; then
  echo "[pi-uefi-stage] Missing lock file: ${LOCK_FILE}" >&2
  exit 1
fi

BUNDLE_PATH="${1:-${UNISPLIT_PI_UEFI_BUNDLE:-}}"
if [[ -z "${BUNDLE_PATH}" ]]; then
  echo "[pi-uefi-stage] Missing bundle input." >&2
  echo "[pi-uefi-stage] Usage: UNISPLIT_PI_UEFI_BUNDLE=/abs/path/RPi4_UEFI_Firmware_v1.51.zip make pi-uefi-stage" >&2
  exit 1
fi

if [[ ! -f "${BUNDLE_PATH}" ]]; then
  echo "[pi-uefi-stage] Bundle file not found: ${BUNDLE_PATH}" >&2
  exit 1
fi

VALIDATION_JSON="$(python3 - "${LOCK_FILE}" "${BUNDLE_PATH}" <<'PY'
import hashlib
import json
import pathlib
import sys
import zipfile

lock_path = pathlib.Path(sys.argv[1])
bundle_path = pathlib.Path(sys.argv[2])
lock = json.loads(lock_path.read_text(encoding="utf-8"))
sha_expected = lock.get("sha256", "").strip().lower()
sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
if not sha_expected:
    raise SystemExit("lock file missing sha256")
if sha != sha_expected:
    raise SystemExit(f"sha256 mismatch: expected {sha_expected} got {sha}")
with zipfile.ZipFile(bundle_path, "r") as zf:
    names = set(zf.namelist())
missing = [entry for entry in lock.get("required_entries", []) if entry not in names]
if missing:
    raise SystemExit("bundle missing required entries: " + ", ".join(missing))
out = {
    "version": lock.get("version", "unknown"),
    "archive_filename": lock.get("archive_filename", "unknown"),
    "source_url": lock.get("source_url", ""),
    "sha256": sha,
}
print(json.dumps(out))
PY
)"

mkdir -p "${LATEST_DIR}"
rm -rf "${BOOT_MEDIA_DIR}"
rm -rf "${LATEST_DIR}/boot_media"
mkdir -p "${BOOT_MEDIA_DIR}"

unzip -q "${BUNDLE_PATH}" -d "${BOOT_MEDIA_DIR}"

python3 - "${LOCK_FILE}" "${BOOT_MEDIA_DIR}" <<'PY'
import json
import pathlib
import sys

lock = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
root = pathlib.Path(sys.argv[2])
missing = [entry for entry in lock["required_entries"] if not (root / entry).exists()]
if missing:
    raise SystemExit("staged tree missing required entries: " + ", ".join(missing))
PY

(
  cd "${BOOT_MEDIA_DIR}"
  find . -type f | sed 's#^\./##' | sort > "${BASELINE_FILES}"
  sha256sum $(cat "${BASELINE_FILES}") > "${BASELINE_SUMS}"
)

python3 - "${VALIDATION_JSON}" "${BUNDLE_PATH}" "${BOOT_MEDIA_DIR}" > "${META_FILE}" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

meta = json.loads(sys.argv[1])
bundle = pathlib.Path(sys.argv[2]).resolve()
boot_dir = pathlib.Path(sys.argv[3]).resolve()
print(f"timestamp_utc={datetime.now(timezone.utc).isoformat()}")
print(f"bundle_path={bundle}")
print(f"bundle_sha256={meta['sha256']}")
print(f"bundle_version={meta['version']}")
print(f"bundle_archive={meta['archive_filename']}")
print(f"source_url={meta['source_url']}")
print(f"staged_boot_media_dir={boot_dir}")
PY

echo "[ok] Staged immutable pftf bundle tree."
echo "[dir ] ${BOOT_MEDIA_DIR}"
echo "[meta] ${META_FILE}"
echo "[list] ${BASELINE_FILES}"
echo "[sha ] ${BASELINE_SUMS}"
