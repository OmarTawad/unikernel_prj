"""Training loop for the IoTCNN model.

Implements:
    - Full epoch-based training with validation
    - CrossEntropyLoss with optional class weights
    - Adam optimizer with ReduceLROnPlateau scheduler
    - Checkpoint saving (best, latest, periodic)
    - Metrics logging to JSONL
    - Reproducible seed handling
    - Dry-run mode for architecture validation
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from unisplit.training.checkpoint import save_checkpoint
from unisplit.training.evaluator import evaluate
from unisplit.training.metrics import compute_metrics

logger = logging.getLogger("unisplit.trainer")


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """Training orchestrator for the IoTCNN model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        class_weights: torch.Tensor | None = None,
        checkpoint_dir: str = "checkpoints",
        metrics_log: str = "checkpoints/metrics.jsonl",
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        save_every_n_epochs: int = 5,
        class_names: list[str] | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_log = Path(metrics_log)
        self.metrics_log.parent.mkdir(parents=True, exist_ok=True)
        self.save_every_n_epochs = save_every_n_epochs
        self.class_names = class_names

        # Loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=scheduler_factor,
            patience=scheduler_patience, verbose=True,
        )

        self.best_val_f1 = 0.0
        self.best_epoch = 0

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch.

        Returns:
            Dictionary with train_loss and train_f1.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / max(num_batches, 1)
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        metrics = compute_metrics(y_true, y_pred)

        return {
            "train_loss": avg_loss,
            "train_accuracy": metrics["accuracy"],
            "train_f1_weighted": metrics["f1_weighted"],
            "train_f1_macro": metrics["f1_macro"],
        }

    def train(self, epochs: int) -> dict:
        """Run the full training loop.

        Args:
            epochs: Number of training epochs.

        Returns:
            Dictionary with final training summary.
        """
        logger.info(f"Starting training for {epochs} epochs on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        history = []

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = evaluate(
                self.model, self.val_loader, self.criterion,
                self.device, self.class_names,
            )

            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            # Scheduler step
            self.scheduler.step(val_metrics["f1_weighted"])

            # Logging
            log_entry = {
                "epoch": epoch,
                "lr": lr,
                "epoch_time_s": round(epoch_time, 2),
                **train_metrics,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_weighted": val_metrics["f1_weighted"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
            history.append(log_entry)

            # Write metrics to JSONL
            with open(self.metrics_log, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            logger.info(
                f"Epoch {epoch:>3}/{epochs} | "
                f"train_loss={train_metrics['train_loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_f1={val_metrics['f1_weighted']:.4f} | "
                f"lr={lr:.6f} | "
                f"time={epoch_time:.1f}s"
            )

            # Save best checkpoint
            if val_metrics["f1_weighted"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1_weighted"]
                self.best_epoch = epoch
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    val_metrics["f1_weighted"],
                    self.checkpoint_dir / "best.pt",
                )

            # Save latest
            save_checkpoint(
                self.model, self.optimizer, epoch,
                val_metrics["f1_weighted"],
                self.checkpoint_dir / "latest.pt",
            )

            # Periodic save
            if epoch % self.save_every_n_epochs == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    val_metrics["f1_weighted"],
                    self.checkpoint_dir / f"epoch_{epoch}.pt",
                )

        logger.info(
            f"Training complete. Best val_f1={self.best_val_f1:.4f} at epoch {self.best_epoch}"
        )

        return {
            "best_val_f1": self.best_val_f1,
            "best_epoch": self.best_epoch,
            "total_epochs": epochs,
            "history": history,
        }


def dry_run(model: nn.Module, device: torch.device, num_batches: int = 2) -> bool:
    """Run a dry-run training pass with synthetic data.

    Validates that the model architecture, loss, and gradient flow work
    correctly without requiring real data.

    Args:
        model: Model to test.
        device: Compute device.
        num_batches: Number of batches to run.

    Returns:
        True if successful.
    """
    from unisplit.training.dataset import SyntheticDataset
    from unisplit.training.dataloader import create_dataloader

    logger.info("Starting dry-run with synthetic data...")

    dataset = SyntheticDataset(num_samples=64, seed=42)
    loader = create_dataloader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    losses = []

    for batch_idx, (features, labels) in enumerate(loader):
        if batch_idx >= num_batches:
            break

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        logger.info(f"  Dry-run batch {batch_idx + 1}: loss={loss.item():.4f}, logits_shape={logits.shape}")

    # Verify gradient flow
    has_grads = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters() if p.requires_grad
    )

    logger.info(f"  Loss values: {[f'{l:.4f}' for l in losses]}")
    logger.info(f"  Gradient flow: {'✓' if has_grads else '✗'}")
    logger.info(f"  Output shape: (batch, {model.num_classes})")

    if has_grads:
        logger.info("✓ Dry-run passed — architecture is valid")
    else:
        logger.error("✗ Dry-run failed — no gradients detected")

    return has_grads
