#!/usr/bin/env python3
"""Preprocessing entry point for CIC-IoT2023 dataset.

Usage:
    # Full preprocessing
    python scripts/preprocess_dataset.py --config configs/dataset.yaml

    # Testing with limited rows per CSV
    python scripts/preprocess_dataset.py --config configs/dataset.yaml --max-rows 1000

    # Validate existing processed data
    python scripts/preprocess_dataset.py --validate-only
"""

import argparse
import sys

from unisplit.shared.config import load_config
from unisplit.shared.logging import setup_logging
from unisplit.training.preprocessing import run_preprocessing, validate_processed_dataset


def main():
    parser = argparse.ArgumentParser(description="Preprocess CIC-IoT2023 dataset")
    parser.add_argument("--config", default="configs/dataset.yaml", help="Dataset config")
    parser.add_argument("--max-rows", type=int, default=None,
                       help="Max rows per CSV (for testing)")
    parser.add_argument("--chunk-size", type=int, default=50_000,
                       help="Rows per streaming chunk (default: 50000)")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate existing processed data")
    args = parser.parse_args()

    logger = setup_logging(level="INFO", fmt="plain", name="unisplit.preprocessing")

    config = load_config(args.config)
    dc = config.dataset

    if args.validate_only:
        checks = validate_processed_dataset(dc.processed_dir, dc.metadata_dir, dc.splits_dir)
        print("\nDataset Validation:")
        all_pass = True
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
            if not passed:
                all_pass = False
        sys.exit(0 if all_pass else 1)

    run_preprocessing(
        raw_dir=dc.raw_dir,
        processed_dir=dc.processed_dir,
        metadata_dir=dc.metadata_dir,
        splits_dir=dc.splits_dir,
        label_column=dc.label_column,
        feature_columns=dc.feature_columns or None,
        class_names=dc.class_names or None,
        train_ratio=dc.train_ratio,
        val_ratio=dc.val_ratio,
        test_ratio=dc.test_ratio,
        chunk_size=args.chunk_size,
        max_rows_per_file=args.max_rows,
    )


if __name__ == "__main__":
    main()
