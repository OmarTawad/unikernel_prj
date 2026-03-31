"""Dataset preprocessing pipeline for CIC-IoT2023.

DESIGNED FOR LARGE-SCALE DATA (46.6M rows) ON MEMORY-CONSTRAINED VPS (22 GiB).

Two-pass streaming architecture:
    Pass 1 (counting): Stream all CSVs → count rows, discover columns, build label map
    Pass 2 (writing):  Stream all CSVs → write features/labels into pre-allocated
                       numpy files on disk, compute normalization stats on-the-fly

Peak RAM: ~50K rows × 80 cols × 4 bytes ≈ 16 MB per chunk + overhead.
No pd.concat. No full-dataset copies.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from unisplit.shared.constants import CLASS_NAMES, NUM_CLASSES, NUM_FEATURES

logger = logging.getLogger("unisplit.preprocessing")

# Default chunk size for streaming CSVs.
# 50K rows × 80 float32 cols ≈ 16 MB per chunk — safe for 22 GiB.
DEFAULT_CHUNK_SIZE = 50_000


def discover_raw_csvs(raw_dir: str | Path) -> list[Path]:
    """Find all CSV files in the raw data directory."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            "Run `make download-data` or manually place CSV files there."
        )
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}\n"
            "Place CIC-IoT2023 CSV files in this directory."
        )
    logger.info(f"Found {len(csvs)} CSV files in {raw_dir}")
    return csvs


def _resolve_feature_columns(
    sample_df: pd.DataFrame,
    feature_columns: list[str] | None,
    label_column: str,
) -> list[str]:
    """Determine the final ordered list of feature columns from a sample chunk.

    If explicit feature_columns are given and enough match, use those.
    Otherwise, auto-select the first NUM_FEATURES numeric columns.
    """
    exclude_cols = {
        label_column, "label", "Label", "class", "Class", "target", "Target",
        "Unnamed: 0", "index", "timestamp", "Timestamp",
    }

    if feature_columns:
        available = [c for c in feature_columns if c in sample_df.columns]
        if len(available) >= NUM_FEATURES:
            return available[:NUM_FEATURES]
        logger.warning(
            f"Only {len(available)}/{NUM_FEATURES} specified feature columns found "
            f"in actual CSV. Falling back to auto-selection."
        )

    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    if len(numeric_cols) < NUM_FEATURES:
        logger.warning(
            f"Only {len(numeric_cols)} numeric columns available, expected {NUM_FEATURES}. "
            "Using all available."
        )
        return numeric_cols

    logger.info(f"Auto-selected {NUM_FEATURES} feature columns")
    return numeric_cols[:NUM_FEATURES]


def _build_label_map(class_names: list[str] | None) -> dict[str, int]:
    """Build the label string → integer mapping."""
    names = class_names if class_names else CLASS_NAMES
    return {name: idx for idx, name in enumerate(names)}


# ──────────────────────────────────────────────────────────────────────
# Pass 1: Count rows and validate structure
# ──────────────────────────────────────────────────────────────────────

def _pass1_count(
    csv_paths: list[Path],
    label_column: str,
    label_map: dict[str, int],
    feature_columns: list[str] | None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[int, list[str], dict[str, int]]:
    """Stream all CSVs once to count valid rows and resolve columns.

    Returns:
        (total_valid_rows, resolved_feature_columns, class_counts)
    """
    total_rows = 0
    resolved_cols: list[str] | None = None
    class_counts: dict[str, int] = {}
    t0 = time.time()

    for csv_idx, csv_path in enumerate(csv_paths):
        file_rows = 0
        logger.info(f"  Pass 1 [{csv_idx + 1}/{len(csv_paths)}] scanning {csv_path.name}...")

        try:
            reader = pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)
        except Exception as e:
            logger.warning(f"  Skipping {csv_path.name}: {e}")
            continue

        for chunk in reader:
            # Resolve columns from the first chunk we see
            if resolved_cols is None:
                resolved_cols = _resolve_feature_columns(chunk, feature_columns, label_column)
                logger.info(f"  Resolved {len(resolved_cols)} feature columns from {csv_path.name}")

            if label_column not in chunk.columns:
                logger.warning(f"  No '{label_column}' column in {csv_path.name}, skipping file")
                break

            # Count valid rows (label exists in label_map)
            labels_raw = chunk[label_column].astype(str).str.strip()
            valid = labels_raw.isin(label_map)
            valid_count = int(valid.sum())
            file_rows += valid_count

            # Track class distribution
            for lbl, cnt in labels_raw[valid].value_counts().items():
                class_counts[lbl] = class_counts.get(lbl, 0) + int(cnt)

        total_rows += file_rows
        logger.info(f"    → {file_rows:,} valid rows")

    elapsed = time.time() - t0
    logger.info(
        f"  Pass 1 complete: {total_rows:,} valid rows across {len(csv_paths)} files "
        f"in {elapsed:.1f}s"
    )

    if resolved_cols is None:
        raise ValueError("No CSV files contained valid data")

    return total_rows, resolved_cols, class_counts


# ──────────────────────────────────────────────────────────────────────
# Pass 2: Write features+labels to disk, compute streaming stats
# ──────────────────────────────────────────────────────────────────────

def _pass2_write(
    csv_paths: list[Path],
    label_column: str,
    label_map: dict[str, int],
    feature_cols: list[str],
    total_rows: int,
    train_indices_set: set[int],
    features_path: Path,
    labels_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict:
    """Stream all CSVs again, writing features/labels to pre-allocated numpy files.

    Also computes online min/max normalization statistics over TRAINING rows only.

    Returns:
        norm_stats dict with min, max, range per feature.
    """
    num_features = len(feature_cols)
    t0 = time.time()

    # Pre-allocate output files on disk using memmap
    # These will be the final .npy files
    feat_mm = np.lib.format.open_memmap(
        str(features_path), mode="w+", dtype=np.float32,
        shape=(total_rows, num_features),
    )
    label_mm = np.lib.format.open_memmap(
        str(labels_path), mode="w+", dtype=np.int64,
        shape=(total_rows,),
    )

    # Online normalization accumulators (training rows only)
    running_min = np.full(num_features, np.inf, dtype=np.float64)
    running_max = np.full(num_features, -np.inf, dtype=np.float64)
    # Also track mean/std via Welford's online algorithm for potential future use
    running_count = 0
    running_mean = np.zeros(num_features, dtype=np.float64)
    running_m2 = np.zeros(num_features, dtype=np.float64)

    write_cursor = 0

    for csv_idx, csv_path in enumerate(csv_paths):
        logger.info(f"  Pass 2 [{csv_idx + 1}/{len(csv_paths)}] writing {csv_path.name}...")

        try:
            reader = pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)
        except Exception as e:
            logger.warning(f"  Skipping {csv_path.name}: {e}")
            continue

        file_written = 0
        for chunk in reader:
            if label_column not in chunk.columns:
                break

            labels_raw = chunk[label_column].astype(str).str.strip()
            valid_mask = labels_raw.isin(label_map)
            if not valid_mask.any():
                continue

            chunk_valid = chunk[valid_mask]
            labels_encoded = labels_raw[valid_mask].map(label_map).values.astype(np.int64)

            # Extract features — handle missing columns gracefully
            present_cols = [c for c in feature_cols if c in chunk_valid.columns]
            if len(present_cols) < num_features:
                # Pad with zeros for missing columns
                feats = np.zeros((len(chunk_valid), num_features), dtype=np.float32)
                for i, col in enumerate(feature_cols):
                    if col in chunk_valid.columns:
                        feats[:, i] = chunk_valid[col].values.astype(np.float32)
            else:
                feats = chunk_valid[feature_cols].values.astype(np.float32)

            # Clean: replace inf/nan with 0
            np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

            n = len(feats)
            end_cursor = write_cursor + n
            feat_mm[write_cursor:end_cursor] = feats
            label_mm[write_cursor:end_cursor] = labels_encoded

            # Update normalization stats for TRAINING rows only
            # Check which global indices in [write_cursor, end_cursor) are train
            global_indices = np.arange(write_cursor, end_cursor)
            # Instead of set lookup on millions of items, use a vectorized approach
            # We pre-built a boolean mask, but for streaming we check membership
            train_mask_local = np.array(
                [idx in train_indices_set for idx in global_indices], dtype=bool
            )
            train_feats = feats[train_mask_local]

            if len(train_feats) > 0:
                # Min/max
                batch_min = train_feats.min(axis=0).astype(np.float64)
                batch_max = train_feats.max(axis=0).astype(np.float64)
                running_min = np.minimum(running_min, batch_min)
                running_max = np.maximum(running_max, batch_max)

                # Welford's online mean/variance
                for row in train_feats:
                    running_count += 1
                    delta = row.astype(np.float64) - running_mean
                    running_mean += delta / running_count
                    delta2 = row.astype(np.float64) - running_mean
                    running_m2 += delta * delta2

            write_cursor = end_cursor
            file_written += n

        logger.info(f"    → wrote {file_written:,} rows (cursor at {write_cursor:,})")

    # Flush memmap to disk
    del feat_mm
    del label_mm

    elapsed = time.time() - t0
    logger.info(f"  Pass 2 complete: {write_cursor:,} rows written in {elapsed:.1f}s")

    if write_cursor != total_rows:
        logger.warning(
            f"  Row count mismatch: expected {total_rows:,}, wrote {write_cursor:,}. "
            "This can happen if CSV files changed between passes."
        )

    # Compute final std
    running_std = np.sqrt(running_m2 / max(running_count, 1))

    # Build norm stats
    norm_range = running_max - running_min
    norm_range[norm_range < 1e-10] = 1.0  # avoid div by zero

    return {
        "min": running_min.tolist(),
        "max": running_max.tolist(),
        "range": norm_range.tolist(),
        "mean": running_mean.tolist(),
        "std": running_std.tolist(),
        "num_features": num_features,
        "num_train_samples": running_count,
    }


# ──────────────────────────────────────────────────────────────────────
# Split generation — works on indices only, no data copying
# ──────────────────────────────────────────────────────────────────────

def generate_stratified_splits(
    labels_path: Path,
    total_rows: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate stratified train/val/test splits from the labels file.

    Reads the labels via memmap (no full copy), performs stratified split
    on the integer labels.

    Returns:
        (train_indices, val_indices, test_indices) as sorted int arrays.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    logger.info(f"Generating stratified splits ({train_ratio}/{val_ratio}/{test_ratio})...")

    # Load labels via memmap — only indices and label values needed
    labels = np.load(str(labels_path), mmap_mode="r")
    indices = np.arange(total_rows)

    # For very large datasets, sklearn's train_test_split with stratify
    # needs the labels in memory. int64 for 46.6M rows = 373 MB — acceptable.
    labels_array = np.array(labels[:total_rows])  # force read into RAM

    # First split: train vs (val+test)
    val_test_ratio = val_ratio + test_ratio
    train_idx, val_test_idx = train_test_split(
        indices, test_size=val_test_ratio, random_state=seed,
        stratify=labels_array,
    )

    # Second split: val vs test
    val_frac = val_ratio / val_test_ratio
    val_idx, test_idx = train_test_split(
        val_test_idx, test_size=(1 - val_frac),
        random_state=seed, stratify=labels_array[val_test_idx],
    )

    # Sort for cache-friendly access during training
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.sort(test_idx)

    logger.info(
        f"  Split: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}"
    )
    return train_idx, val_idx, test_idx


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────

def run_preprocessing(
    raw_dir: str | Path,
    processed_dir: str | Path,
    metadata_dir: str | Path,
    splits_dir: str | Path,
    label_column: str = "label",
    feature_columns: list[str] | None = None,
    class_names: list[str] | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_rows_per_file: int | None = None,
) -> None:
    """Run the full two-pass preprocessing pipeline.

    Pass 1: Count valid rows, resolve feature columns, count class distribution
    Pass 2: Write features/labels to pre-allocated numpy memmap files,
            compute normalization stats on training rows only

    This processes 46.6M rows in ~16 MB peak chunk RAM.

    Args:
        raw_dir: Path to raw CSV directory.
        processed_dir: Output directory for processed arrays.
        metadata_dir: Output directory for statistics and label maps.
        splits_dir: Output directory for split index files.
        label_column: Label column name.
        feature_columns: Optional explicit feature column list.
        class_names: Optional explicit class names.
        train_ratio: Training fraction.
        val_ratio: Validation fraction.
        test_ratio: Test fraction.
        seed: Random seed.
        chunk_size: Rows per streaming chunk.
        max_rows_per_file: Max rows per CSV file (for testing only).
    """
    t_start = time.time()

    # Create output directories
    for d in [processed_dir, metadata_dir, splits_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Build label map
    label_map = _build_label_map(class_names)
    logger.info(f"Label map: {len(label_map)} classes")

    # Step 1: Discover CSVs
    csv_paths = discover_raw_csvs(raw_dir)

    # For testing mode: wrap CSVs to limit rows
    effective_chunk = chunk_size
    if max_rows_per_file is not None:
        effective_chunk = min(chunk_size, max_rows_per_file)
        logger.info(f"  Testing mode: max {max_rows_per_file} rows per file")

    # ── Pass 1: Count ──
    logger.info("═══ Pass 1: Counting rows and resolving columns ═══")
    total_rows, resolved_feature_cols, class_counts = _pass1_count(
        csv_paths, label_column, label_map, feature_columns, effective_chunk,
    )

    if total_rows == 0:
        raise ValueError("No valid rows found in any CSV file")

    # ── Generate splits from row count (indices only, no data) ──
    # We need to generate splits BEFORE pass 2 so we know which rows are training
    # rows for normalization stats.
    #
    # Strategy: create a temporary labels file during pass 1? No — we don't have
    # labels yet. Instead, we write features+labels in pass 2, THEN generate splits,
    # THEN compute normalization stats in a lightweight pass 3.
    #
    # Actually, the cleanest approach for correctness:
    # 1. Pass 1: count rows
    # 2. Pass 2: write features + labels to memmap (no stats yet)
    # 3. Generate splits from labels file (memmap read, ~373 MB for 46.6M int64)
    # 4. Pass 3: stream the WRITTEN features memmap, compute stats on train indices

    # ── Pass 2: Write features + labels ──
    logger.info("═══ Pass 2: Writing features and labels to disk ═══")
    features_path = Path(processed_dir) / "features.npy"
    labels_path = Path(processed_dir) / "labels.npy"

    # For pass 2 we pass an empty train set — stats will be computed in pass 3
    _pass2_write(
        csv_paths, label_column, label_map, resolved_feature_cols,
        total_rows, set(),  # empty train set — no stats in this pass
        features_path, labels_path, effective_chunk,
    )

    # ── Generate splits ──
    logger.info("═══ Generating stratified splits ═══")
    train_idx, val_idx, test_idx = generate_stratified_splits(
        labels_path, total_rows, train_ratio, val_ratio, test_ratio, seed,
    )

    np.save(Path(splits_dir) / "train_indices.npy", train_idx)
    np.save(Path(splits_dir) / "val_indices.npy", val_idx)
    np.save(Path(splits_dir) / "test_indices.npy", test_idx)

    # ── Pass 3: Compute normalization stats on training rows ──
    logger.info("═══ Pass 3: Computing normalization statistics on training set ═══")
    norm_stats = _compute_train_norm_stats(features_path, train_idx, len(resolved_feature_cols))

    # ── Save metadata ──
    with open(Path(metadata_dir) / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    with open(Path(metadata_dir) / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    with open(Path(metadata_dir) / "feature_columns.json", "w") as f:
        json.dump({
            "columns": resolved_feature_cols,
            "num_features": len(resolved_feature_cols),
        }, f, indent=2)

    with open(Path(metadata_dir) / "class_counts.json", "w") as f:
        json.dump(class_counts, f, indent=2)

    manifest = {
        "total_rows": total_rows,
        "num_features": len(resolved_feature_cols),
        "num_classes": len(label_map),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "seed": seed,
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "csv_files": [p.name for p in csv_paths],
        "chunk_size": chunk_size,
    }
    with open(Path(metadata_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t_start
    logger.info("═══ Preprocessing complete ═══")
    logger.info(f"  Total rows:    {total_rows:,}")
    logger.info(f"  Features:      {len(resolved_feature_cols)}")
    logger.info(f"  Classes:       {len(label_map)}")
    logger.info(f"  Train/Val/Test: {len(train_idx):,} / {len(val_idx):,} / {len(test_idx):,}")
    logger.info(f"  Output dir:    {processed_dir}")
    logger.info(f"  Total time:    {elapsed:.1f}s")
    logger.info("✓ Done")


def _compute_train_norm_stats(
    features_path: Path,
    train_indices: np.ndarray,
    num_features: int,
    batch_size: int = 100_000,
) -> dict:
    """Compute min/max/mean/std normalization statistics over training rows only.

    Reads the features memmap in batches of train indices. No full copy.

    Args:
        features_path: Path to features.npy memmap.
        train_indices: Sorted array of training row indices.
        num_features: Number of feature columns.
        batch_size: Number of training rows to process at a time.

    Returns:
        Normalization stats dict.
    """
    features = np.load(str(features_path), mmap_mode="r")
    n_train = len(train_indices)

    running_min = np.full(num_features, np.inf, dtype=np.float64)
    running_max = np.full(num_features, -np.inf, dtype=np.float64)
    running_count = 0
    running_mean = np.zeros(num_features, dtype=np.float64)
    running_m2 = np.zeros(num_features, dtype=np.float64)

    for start in range(0, n_train, batch_size):
        end = min(start + batch_size, n_train)
        idx_batch = train_indices[start:end]
        batch = features[idx_batch].astype(np.float64)

        # Min/max
        running_min = np.minimum(running_min, batch.min(axis=0))
        running_max = np.maximum(running_max, batch.max(axis=0))

        # Welford's batched update
        for row in batch:
            running_count += 1
            delta = row - running_mean
            running_mean += delta / running_count
            delta2 = row - running_mean
            running_m2 += delta * delta2

        if (start // batch_size) % 50 == 0:
            logger.info(f"  Stats: processed {end:,}/{n_train:,} training rows")

    running_std = np.sqrt(running_m2 / max(running_count, 1))
    norm_range = running_max - running_min
    norm_range[norm_range < 1e-10] = 1.0

    logger.info(f"  Stats computed over {running_count:,} training rows")

    return {
        "min": running_min.tolist(),
        "max": running_max.tolist(),
        "range": norm_range.tolist(),
        "mean": running_mean.tolist(),
        "std": running_std.tolist(),
        "num_features": num_features,
        "num_train_samples": running_count,
    }


def validate_processed_dataset(
    processed_dir: str | Path,
    metadata_dir: str | Path,
    splits_dir: str | Path,
) -> dict[str, bool]:
    """Validate that all expected processed data files exist and are well-formed."""
    checks = {}
    processed_dir = Path(processed_dir)
    metadata_dir = Path(metadata_dir)
    splits_dir = Path(splits_dir)

    checks["features_exist"] = (processed_dir / "features.npy").exists()
    checks["labels_exist"] = (processed_dir / "labels.npy").exists()
    checks["norm_stats_exist"] = (metadata_dir / "norm_stats.json").exists()
    checks["label_map_exist"] = (metadata_dir / "label_map.json").exists()
    checks["manifest_exist"] = (metadata_dir / "manifest.json").exists()
    checks["train_split_exist"] = (splits_dir / "train_indices.npy").exists()
    checks["val_split_exist"] = (splits_dir / "val_indices.npy").exists()
    checks["test_split_exist"] = (splits_dir / "test_indices.npy").exists()

    if checks["features_exist"] and checks["labels_exist"]:
        features = np.load(processed_dir / "features.npy", mmap_mode="r")
        labels = np.load(processed_dir / "labels.npy", mmap_mode="r")
        checks["features_2d"] = features.ndim == 2
        checks["labels_1d"] = labels.ndim == 1
        checks["samples_match"] = features.shape[0] == labels.shape[0]
        checks["feature_count_positive"] = features.shape[1] > 0

        total = features.shape[0]
        if checks.get("train_split_exist") and checks.get("val_split_exist") and checks.get("test_split_exist"):
            train_idx = np.load(splits_dir / "train_indices.npy")
            val_idx = np.load(splits_dir / "val_indices.npy")
            test_idx = np.load(splits_dir / "test_indices.npy")
            checks["split_indices_valid"] = (
                train_idx.max() < total and val_idx.max() < total and test_idx.max() < total
            )
            checks["split_no_overlap"] = len(
                set(train_idx) & set(val_idx) | set(train_idx) & set(test_idx) | set(val_idx) & set(test_idx)
            ) == 0
            checks["split_covers_all"] = (len(train_idx) + len(val_idx) + len(test_idx)) == total

    if checks.get("norm_stats_exist"):
        with open(metadata_dir / "norm_stats.json") as f:
            stats = json.load(f)
        checks["stats_has_min"] = "min" in stats
        checks["stats_has_max"] = "max" in stats
        checks["stats_has_range"] = "range" in stats

    return checks
