#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATEST_DIR="${ROOT_DIR}/artifacts/pi_handoff/latest"
IMAGE_DIR="${LATEST_DIR}/images"
BOOT_DIR="${LATEST_DIR}/boot_media"
BOOT_PART_DIR="${BOOT_DIR}/boot"
ENV_TEMPLATE="${ROOT_DIR}/configs/pi_edge_runtime.env.example"
CONFIG_TEMPLATE="${ROOT_DIR}/configs/pi_boot/config.txt"
CMDLINE_TEMPLATE="${ROOT_DIR}/configs/pi_boot/cmdline.txt.template"

# Optional env file override for endpoint/runtime settings.
PI_ENV_FILE="${UNISPLIT_PI_ENV_FILE:-${ENV_TEMPLATE}}"

# Build deterministic image candidate first.
"${ROOT_DIR}/scripts/build_pi_image.sh"

if [[ ! -f "${PI_ENV_FILE}" ]]; then
  echo "[pi-boot] Missing env file: ${PI_ENV_FILE}" >&2
  exit 1
fi

# shellcheck disable=SC1090
set -a
source "${PI_ENV_FILE}"
set +a

: "${UNISPLIT_EDGE_SPLIT_ID:=7}"
: "${UNISPLIT_TRANSPORT_BACKEND:=lwip}"
: "${UNISPLIT_TRANSPORT_ENDPOINT:=http://192.168.1.10:8000}"
: "${UNISPLIT_TRANSPORT_PATH:=/infer/split}"
: "${UNISPLIT_TRANSPORT_TIMEOUT_SECONDS:=10}"
: "${UNISPLIT_TRANSPORT_RETRIES:=3}"

mkdir -p "${BOOT_PART_DIR}"
cp "${IMAGE_DIR}/kernel8.img" "${BOOT_PART_DIR}/kernel8.img"
cp "${CONFIG_TEMPLATE}" "${BOOT_PART_DIR}/config.txt"

CMDLINE_LINE="console=serial0,115200 console=tty1 root=/dev/ram0 rw -- --split-id ${UNISPLIT_EDGE_SPLIT_ID} --backend ${UNISPLIT_TRANSPORT_BACKEND} --endpoint ${UNISPLIT_TRANSPORT_ENDPOINT} --path ${UNISPLIT_TRANSPORT_PATH} --timeout ${UNISPLIT_TRANSPORT_TIMEOUT_SECONDS} --retries ${UNISPLIT_TRANSPORT_RETRIES}"
printf "%s\n" "${CMDLINE_LINE}" > "${BOOT_PART_DIR}/cmdline.txt"

cp "${ENV_TEMPLATE}" "${BOOT_DIR}/pi_edge_runtime.env.example"
cp "${CMDLINE_TEMPLATE}" "${BOOT_DIR}/cmdline.txt.template"
cp "${LATEST_DIR}/pi_readiness_manifest.json" "${BOOT_DIR}/pi_readiness_manifest.json" 2>/dev/null || true

cat > "${BOOT_DIR}/BOOT_MEDIA_README.txt" <<README
UniSplit Pi Boot Media Layout
=============================

Boot partition files (copy into FAT32 boot partition):
- boot/kernel8.img
- boot/config.txt
- boot/cmdline.txt

kernel8.img source:
- ${IMAGE_DIR}/kernel8.img

Runtime arguments in cmdline.txt (passed into unikernel main argv):
- split: ${UNISPLIT_EDGE_SPLIT_ID}
- backend: ${UNISPLIT_TRANSPORT_BACKEND}
- endpoint: ${UNISPLIT_TRANSPORT_ENDPOINT}
- path: ${UNISPLIT_TRANSPORT_PATH}
- timeout: ${UNISPLIT_TRANSPORT_TIMEOUT_SECONDS}
- retries: ${UNISPLIT_TRANSPORT_RETRIES}

Expected serial markers for first bring-up:
- PI_MARKER_BOOT_START
- PI_MARKER_ARTIFACT_STRATEGY=embedded_edge_k9_superset_v1
- PI_MARKER_CONFIG_OK
- PI_MARKER_BACKEND_INIT_OK
- PI_MARKER_NETWORK_READY
- PI_MARKER_INFER_ATTEMPT
- PI_MARKER_INFER_RESPONSE_OK
- PI_MARKER_FINAL_SUCCESS
README

cat > "${BOOT_DIR}/boot_media_manifest.txt" <<MAN
timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
boot_dir=${BOOT_PART_DIR}
kernel=${BOOT_PART_DIR}/kernel8.img
config=${BOOT_PART_DIR}/config.txt
cmdline=${BOOT_PART_DIR}/cmdline.txt
pi_env_file=${PI_ENV_FILE}
MAN

echo "[ok] Pi boot-media layout prepared."
echo "[dir ] ${BOOT_DIR}"
echo "[boot] ${BOOT_PART_DIR}/kernel8.img"
echo "[boot] ${BOOT_PART_DIR}/config.txt"
echo "[boot] ${BOOT_PART_DIR}/cmdline.txt"
