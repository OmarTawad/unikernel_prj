#!/usr/bin/env bash
# ============================================================================
# CIC-IoT2023 Dataset Acquisition Script
# ============================================================================
#
# The CIC-IoT2023 dataset is published by the Canadian Institute for
# Cybersecurity at the University of New Brunswick.
#
# Dataset page: https://www.unb.ca/cic/datasets/iotdataset-2023.html
#
# The dataset contains ~46.6M labelled network flow records across
# 34 classes (33 attack types + benign).
#
# This script attempts to download the CSV files. If automated download
# fails (due to portal restrictions), it provides manual instructions.
# ============================================================================

set -euo pipefail

RAW_DIR="${1:-data/raw}"
mkdir -p "$RAW_DIR"

echo "============================================"
echo "  CIC-IoT2023 Dataset Acquisition"
echo "============================================"
echo ""
echo "Target directory: $RAW_DIR"
echo ""

# Check if data already exists
CSV_COUNT=$(find "$RAW_DIR" -name "*.csv" 2>/dev/null | wc -l)
if [ "$CSV_COUNT" -gt 0 ]; then
    echo "Found $CSV_COUNT CSV files already in $RAW_DIR"
    echo "To re-download, remove existing files first."
    exit 0
fi

# Attempt automated download
# The CIC datasets are typically hosted on their own servers
# These URLs may change — update as needed
echo "Attempting automated download..."
echo ""

DOWNLOAD_SUCCESS=false

# Try known mirror/download locations
# NOTE: CIC-IoT2023 is often distributed as multiple CSV files
# The exact URLs depend on the dataset version and hosting
DATASET_URLS=(
    # Add actual URLs here when available
    # "https://example.com/cic-iot-2023/part1.csv"
)

if [ ${#DATASET_URLS[@]} -gt 0 ]; then
    for url in "${DATASET_URLS[@]}"; do
        filename=$(basename "$url")
        echo "Downloading $filename..."
        if wget -q --show-progress -O "$RAW_DIR/$filename" "$url" 2>/dev/null; then
            echo "  ✓ Downloaded $filename"
        else
            echo "  ✗ Failed to download $filename"
        fi
    done
    
    CSV_COUNT=$(find "$RAW_DIR" -name "*.csv" 2>/dev/null | wc -l)
    if [ "$CSV_COUNT" -gt 0 ]; then
        DOWNLOAD_SUCCESS=true
    fi
fi

if [ "$DOWNLOAD_SUCCESS" = false ]; then
    echo "============================================"
    echo "  MANUAL DOWNLOAD REQUIRED"
    echo "============================================"
    echo ""
    echo "Automated download is not available for CIC-IoT2023."
    echo "Please download the dataset manually:"
    echo ""
    echo "1. Visit: https://www.unb.ca/cic/datasets/iotdataset-2023.html"
    echo ""
    echo "2. Request access and download the CSV files"
    echo ""
    echo "3. Place all CSV files in: $(realpath $RAW_DIR)/"
    echo ""
    echo "Expected files:"
    echo "   - One or more CSV files with network flow records"
    echo "   - Each file should have columns including 'label' and"
    echo "     80 statistical network flow features"
    echo ""
    echo "4. After placing files, run: make preprocess-data"
    echo ""
    echo "Expected column format:"
    echo "   - flow_duration, Header_Length, Protocol Type, Rate, ..."
    echo "   - label (string: 'Benign', 'DDoS-SYN_Flood', etc.)"
    echo ""
    echo "Total expected: ~46.6M rows across all files"
    echo "============================================"
    exit 1
fi

echo ""
echo "✓ Dataset files are ready in $RAW_DIR"
echo "Next step: make preprocess-data"
