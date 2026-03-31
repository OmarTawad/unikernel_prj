"""Evaluation loop for validation and test sets."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from unisplit.training.metrics import compute_metrics

logger = logging.getLogger("unisplit.evaluator")


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict:
    """Run evaluation on a dataset.

    Args:
        model: Trained model.
        dataloader: Evaluation DataLoader.
        criterion: Loss function.
        device: Compute device.
        class_names: Optional class names for metrics.

    Returns:
        Dictionary with loss, accuracy, f1 scores, and classification report.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["loss"] = avg_loss

    return metrics
