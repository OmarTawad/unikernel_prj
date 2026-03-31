# Dataset: CIC-IoT2023

## Overview

CIC-IoT2023 is published by the Canadian Institute for Cybersecurity (UNB).
It contains ~46.6M labelled network flow records with 33 attack types + benign.

- **Paper reference**: Neto et al., 2023
- **Features**: 80 statistical flow-level features
- **Classes**: 34 (33 attacks + benign)
- **Split**: 70% train / 15% val / 15% test (stratified)

## Acquisition

```bash
# Attempt automated download (may require manual steps)
bash scripts/download_dataset.sh

# Or manually:
# 1. Visit https://www.unb.ca/cic/datasets/iotdataset-2023.html
# 2. Download CSV files
# 3. Place in data/raw/
```

## Preprocessing

```bash
# Full preprocessing
make preprocess-data

# Validate processed data
python scripts/preprocess_dataset.py --validate-only

# Test with limited rows
python scripts/preprocess_dataset.py --config configs/dataset.yaml --max-rows 1000
```

### Pipeline steps:
1. Discover CSV files in `data/raw/`
2. Load and merge all CSVs
3. Select 80 statistical features
4. Encode 34 class labels
5. Clean data (replace inf/nan)
6. Compute normalization statistics (training set only)
7. Generate stratified 70/15/15 splits
8. Save to `data/processed/`, `data/metadata/`, `data/splits/`

## Directory Structure

```
data/
├── raw/              # Raw CIC-IoT2023 CSV files (user provides)
├── processed/
│   ├── features.npy  # (N, 80) float32
│   └── labels.npy    # (N,) int64
├── metadata/
│   ├── norm_stats.json    # Min/max/mean/std per feature
│   ├── label_map.json     # Class name → index mapping
│   └── feature_columns.json
└── splits/
    ├── train_indices.npy  # 70% indices
    ├── val_indices.npy    # 15% indices
    └── test_indices.npy   # 15% indices
```

## 34 Classes

Benign, DDoS (14 types), DoS (4), Recon (4), Web attacks (6), Brute Force (2), Spoofing (2), Mirai (2)
