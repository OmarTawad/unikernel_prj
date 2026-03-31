"""Metrics computation for training and evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Optional class names for the report.

    Returns:
        Dictionary with accuracy, f1_weighted, f1_macro, and report.
    """
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1_w),
        "f1_macro": float(f1_m),
        "classification_report": report,
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)
