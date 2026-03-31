"""Checkpoint save/load utilities with full training state.

Checkpoints contain everything needed for exact resume:
- model_state_dict
- optimizer_state_dict
- scheduler_state_dict
- epoch, global_step
- best_val_f1, best_epoch
- rng states (python, numpy, torch)
- config_hash for drift detection
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("unisplit.checkpoint")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    val_f1: float,
    best_val_f1: float,
    best_epoch: int,
    path: str | Path,
    config_hash: str = "",
    extra: dict | None = None,
) -> Path:
    """Save a complete training checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state to save.
        scheduler: LR scheduler state to save.
        epoch: Current epoch number (just completed).
        global_step: Global step counter across all epochs.
        val_f1: Validation F1 for THIS epoch.
        best_val_f1: Best validation F1 seen so far.
        best_epoch: Epoch of the best validation F1.
        path: Output path.
        config_hash: Hash of the config for drift detection.
        extra: Additional metadata to save.

    Returns:
        Path to saved checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "global_step": global_step,
        "val_f1": val_f1,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "config_hash": config_hash,
        # RNG states for exact reproducibility
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.random.get_rng_state(),
    }
    if extra:
        checkpoint.update(extra)

    # Write to temp file first, then atomic rename to avoid corruption
    tmp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.rename(path)

    logger.info(
        f"Checkpoint saved: {path} "
        f"(epoch={epoch}, step={global_step}, val_f1={val_f1:.4f}, "
        f"best_f1={best_val_f1:.4f}@{best_epoch})"
    )
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    device: str = "cpu",
    restore_rng: bool = True,
) -> dict:
    """Load a training checkpoint with full state restore.

    Args:
        path: Checkpoint file path.
        model: Model to load state into.
        optimizer: Optional optimizer to load state into.
        scheduler: Optional scheduler to load state into.
        device: Device to map tensors to.
        restore_rng: Whether to restore RNG states.

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

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if restore_rng:
        if "rng_python" in checkpoint:
            random.setstate(checkpoint["rng_python"])
        if "rng_numpy" in checkpoint:
            np.random.set_state(checkpoint["rng_numpy"])
        if "rng_torch" in checkpoint:
            torch.random.set_rng_state(checkpoint["rng_torch"])

    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("global_step", 0)
    best_f1 = checkpoint.get("best_val_f1", checkpoint.get("val_f1", 0.0))
    best_ep = checkpoint.get("best_epoch", epoch)

    logger.info(
        f"Checkpoint loaded: {path} "
        f"(epoch={epoch}, step={step}, best_f1={best_f1:.4f}@{best_ep})"
    )
    return checkpoint


def get_config_hash(config) -> str:
    """Hash the training-relevant config fields for drift detection."""
    try:
        config_dict = config.model_dump() if hasattr(config, "model_dump") else str(config)
        return hashlib.sha256(json.dumps(config_dict, sort_keys=True, default=str).encode()).hexdigest()[:16]
    except Exception:
        return ""
