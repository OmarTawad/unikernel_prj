"""Checkpoint save/load utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger("unisplit.checkpoint")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_f1: float,
    path: str | Path,
    extra: dict | None = None,
) -> Path:
    """Save a training checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state to save.
        epoch: Current epoch number.
        val_f1: Validation F1 score.
        path: Output path.
        extra: Additional metadata to save.

    Returns:
        Path to saved checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_f1": val_f1,
    }
    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path} (epoch={epoch}, val_f1={val_f1:.4f})")
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> dict:
    """Load a training checkpoint.

    Args:
        path: Checkpoint file path.
        model: Model to load state into.
        optimizer: Optional optimizer to load state into.
        device: Device to map tensors to.

    Returns:
        Full checkpoint dict with metadata.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(
        f"Checkpoint loaded: {path} "
        f"(epoch={checkpoint.get('epoch', '?')}, val_f1={checkpoint.get('val_f1', '?')})"
    )
    return checkpoint
