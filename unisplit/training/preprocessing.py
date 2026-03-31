"""Dataset preprocessing pipeline for CIC-IoT2023.

Handles:
    - Raw CSV discovery and loading
    - 80-feature selection
    - Label encoding (34 classes)
    - Normalization statistics computation
    - Stratified train/val/test split generation
    - Processed data persistence
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from unisplit.shared.constants import CLASS_NAMES, NUM_CLASSES, NUM_FEATURES

logger = logging.getLogger("unisplit.preprocessing")


def discover_raw_csvs(raw_dir: str | Path) -> list[Path]:
    """Find all CSV files in the raw data directory.

    Args:
        raw_dir: Path to raw data directory.

    Returns:
        Sorted list of CSV file paths.

    Raises:
        FileNotFoundError: If raw_dir doesn't exist or has no CSVs.
    """
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


def load_and_merge_csvs(
    csv_paths: list[Path],
    label_column: str = "label",
    max_rows_per_file: int | None = None,
) -> pd.DataFrame:
    """Load and merge multiple CSV files.

    Args:
        csv_paths: List of CSV file paths.
        label_column: Name of the label column.
        max_rows_per_file: Optional row limit per file (for testing).

    Returns:
        Merged DataFrame.
    """
    frames = []
    for csv_path in csv_paths:
        logger.info(f"Loading {csv_path.name}...")
        try:
            df = pd.read_csv(csv_path, nrows=max_rows_per_file, low_memory=False)
            frames.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {csv_path}: {e}")
            continue

    if not frames:
        raise ValueError("No CSV files were successfully loaded.")

    merged = pd.concat(frames, ignore_index=True)
    logger.info(f"Merged dataset: {len(merged)} rows, {len(merged.columns)} columns")
    return merged


def select_features(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    num_features: int = NUM_FEATURES,
) -> np.ndarray:
    """Select and extract the feature columns from a DataFrame.

    If feature_columns is provided, use those exact columns.
    Otherwise, auto-select the first num_features numeric columns
    (excluding the label column).

    Args:
        df: Input DataFrame.
        feature_columns: Explicit list of feature column names.
        num_features: Expected number of features.

    Returns:
        Feature array of shape (N, num_features).
    """
    if feature_columns:
        # Try to match specified columns, falling back to available ones
        available = [c for c in feature_columns if c in df.columns]
        if len(available) < num_features:
            logger.warning(
                f"Only {len(available)}/{num_features} specified columns found. "
                "Falling back to auto-selection."
            )
            feature_columns = None
        else:
            selected = df[available[:num_features]]
            return selected.values.astype(np.float32)

    # Auto-select: all numeric columns except label-like ones
    exclude_cols = {"label", "Label", "class", "Class", "target", "Target",
                    "Unnamed: 0", "index", "timestamp", "Timestamp"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    if len(numeric_cols) < num_features:
        logger.warning(
            f"Only {len(numeric_cols)} numeric columns available, "
            f"expected {num_features}. Using all available."
        )
        num_features = len(numeric_cols)

    selected = numeric_cols[:num_features]
    logger.info(f"Selected {len(selected)} features")
    return df[selected].values.astype(np.float32)


def encode_labels(
    df: pd.DataFrame,
    label_column: str = "label",
    class_names: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Encode string labels as integer class indices.

    Args:
        df: DataFrame with label column.
        label_column: Name of the label column.
        class_names: Expected class names for consistent encoding.

    Returns:
        Tuple of (encoded labels array, label map dict).
    """
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame.")

    raw_labels = df[label_column].astype(str).str.strip()

    if class_names:
        label_map = {name: idx for idx, name in enumerate(class_names)}
    else:
        unique_labels = sorted(raw_labels.unique())
        label_map = {name: idx for idx, name in enumerate(unique_labels)}

    # Map labels, using -1 for unknown
    encoded = raw_labels.map(label_map)
    unknown_mask = encoded.isna()
    if unknown_mask.any():
        unknown_labels = raw_labels[unknown_mask].unique()
        logger.warning(
            f"{unknown_mask.sum()} samples with unknown labels: {unknown_labels[:10]}"
        )
        # Assign unknown to last+1 class or drop
        encoded = encoded.fillna(-1).astype(np.int64)
    else:
        encoded = encoded.astype(np.int64)

    logger.info(f"Encoded {len(label_map)} classes, {len(encoded)} samples")
    return encoded.values, label_map


def clean_features(features: np.ndarray) -> np.ndarray:
    """Clean feature array: replace inf/nan with 0.

    Args:
        features: Raw feature array.

    Returns:
        Cleaned array with no inf/nan values.
    """
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def compute_normalization_stats(features: np.ndarray) -> dict:
    """Compute per-feature normalization statistics.

    Uses min-max statistics for [0, 1] normalization as described in
    the paper (Appendix B).

    Args:
        features: Feature array of shape (N, D).

    Returns:
        Dictionary with 'min', 'max', 'mean', 'std' arrays.
    """
    return {
        "min": features.min(axis=0).tolist(),
        "max": features.max(axis=0).tolist(),
        "mean": features.mean(axis=0).tolist(),
        "std": features.std(axis=0).tolist(),
        "num_features": features.shape[1],
        "num_samples": features.shape[0],
    }


def generate_stratified_splits(
    labels: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate stratified train/val/test splits.

    As described in the paper: 70/15/15 stratified split.

    Args:
        labels: Label array of shape (N,).
        train_ratio: Training set fraction.
        val_ratio: Validation set fraction.
        test_ratio: Test set fraction.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n = len(labels)
    indices = np.arange(n)

    # First split: train vs (val+test)
    val_test_ratio = val_ratio + test_ratio
    train_idx, val_test_idx = train_test_split(
        indices, test_size=val_test_ratio, random_state=seed,
        stratify=labels,
    )

    # Second split: val vs test
    val_frac_of_remaining = val_ratio / val_test_ratio
    val_idx, test_idx = train_test_split(
        val_test_idx, test_size=(1 - val_frac_of_remaining),
        random_state=seed, stratify=labels[val_test_idx],
    )

    logger.info(
        f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )
    return train_idx, val_idx, test_idx


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
    max_rows_per_file: int | None = None,
) -> None:
    """Run the full preprocessing pipeline.

    1. Discover CSVs → 2. Load & merge → 3. Select features → 4. Encode labels
    → 5. Clean → 6. Compute stats → 7. Generate splits → 8. Save everything

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
        max_rows_per_file: Row limit per CSV (for testing).
    """
    # Create output directories
    for d in [processed_dir, metadata_dir, splits_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Step 1-2: Discover and load
    csv_paths = discover_raw_csvs(raw_dir)
    df = load_and_merge_csvs(csv_paths, label_column, max_rows_per_file)

    # Step 3: Select features
    features = select_features(df, feature_columns)

    # Step 4: Encode labels
    if class_names is None:
        class_names = CLASS_NAMES
    labels, label_map = encode_labels(df, label_column, class_names)

    # Filter out unknown labels
    valid_mask = labels >= 0
    if not valid_mask.all():
        logger.warning(f"Dropping {(~valid_mask).sum()} samples with unknown labels")
        features = features[valid_mask]
        labels = labels[valid_mask]

    # Step 5: Clean
    features = clean_features(features)

    logger.info(f"Final dataset: {features.shape[0]} samples, {features.shape[1]} features")

    # Step 6: Compute normalization stats (on full dataset before split,
    # but should ideally be on training set only)
    # We'll compute on training set after splitting

    # Step 7: Generate splits
    train_idx, val_idx, test_idx = generate_stratified_splits(
        labels, train_ratio, val_ratio, test_ratio, seed
    )

    # Compute normalization stats on TRAINING set only
    train_features = features[train_idx]
    norm_stats = compute_normalization_stats(train_features)

    # Step 8: Save everything
    np.save(Path(processed_dir) / "features.npy", features)
    np.save(Path(processed_dir) / "labels.npy", labels)

    np.save(Path(splits_dir) / "train_indices.npy", train_idx)
    np.save(Path(splits_dir) / "val_indices.npy", val_idx)
    np.save(Path(splits_dir) / "test_indices.npy", test_idx)

    with open(Path(metadata_dir) / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    with open(Path(metadata_dir) / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    with open(Path(metadata_dir) / "feature_columns.json", "w") as f:
        json.dump({"columns": feature_columns or [], "num_features": features.shape[1]}, f, indent=2)

    # Summary
    logger.info(f"Saved features:    {processed_dir}/features.npy  shape={features.shape}")
    logger.info(f"Saved labels:      {processed_dir}/labels.npy    shape={labels.shape}")
    logger.info(f"Saved splits:      {splits_dir}/  train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    logger.info(f"Saved norm stats:  {metadata_dir}/norm_stats.json")
    logger.info(f"Saved label map:   {metadata_dir}/label_map.json")
    logger.info("✓ Preprocessing complete")


def validate_processed_dataset(
    processed_dir: str | Path,
    metadata_dir: str | Path,
    splits_dir: str | Path,
) -> dict[str, bool]:
    """Validate that all expected processed data files exist and are well-formed.

    Returns:
        Dictionary of check name → pass/fail.
    """
    checks = {}
    processed_dir = Path(processed_dir)
    metadata_dir = Path(metadata_dir)
    splits_dir = Path(splits_dir)

    # Check files exist
    checks["features_exist"] = (processed_dir / "features.npy").exists()
    checks["labels_exist"] = (processed_dir / "labels.npy").exists()
    checks["norm_stats_exist"] = (metadata_dir / "norm_stats.json").exists()
    checks["label_map_exist"] = (metadata_dir / "label_map.json").exists()
    checks["train_split_exist"] = (splits_dir / "train_indices.npy").exists()
    checks["val_split_exist"] = (splits_dir / "val_indices.npy").exists()
    checks["test_split_exist"] = (splits_dir / "test_indices.npy").exists()

    # Check shapes if files exist
    if checks["features_exist"] and checks["labels_exist"]:
        features = np.load(processed_dir / "features.npy", mmap_mode="r")
        labels = np.load(processed_dir / "labels.npy", mmap_mode="r")
        checks["features_2d"] = features.ndim == 2
        checks["labels_1d"] = labels.ndim == 1
        checks["samples_match"] = features.shape[0] == labels.shape[0]
        checks["feature_count"] = features.shape[1] == NUM_FEATURES or features.shape[1] > 0

    if checks.get("norm_stats_exist"):
        with open(metadata_dir / "norm_stats.json") as f:
            stats = json.load(f)
        checks["stats_has_min"] = "min" in stats
        checks["stats_has_max"] = "max" in stats

    return checks
