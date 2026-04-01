"""Training loop for the IoTCNN model — production-grade for long CPU runs.

Implements:
    - Full epoch-based training with validation
    - CrossEntropyLoss with optional class weights
    - Adam optimizer with ReduceLROnPlateau scheduler
    - Complete checkpoint save/resume (model + optimizer + scheduler + RNG + best tracking)
    - Step-level logging with ETA estimation
    - Metrics logging to JSONL (append mode, never overwrites)
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

from unisplit.training.checkpoint import (
    get_config_hash,
    load_checkpoint,
    save_checkpoint,
)
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
    """Training orchestrator for the IoTCNN model.

    Supports:
    - Full training from scratch
    - Exact resume from any checkpoint
    - Step-level progress logging
    - ETA estimation for long runs
    """

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
        save_every_n_epochs: int = 10,
        log_every_n_steps: int = 200,
        class_names: list[str] | None = None,
        config=None,
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
        self.log_every_n_steps = log_every_n_steps
        self.class_names = class_names
        self.config_hash = get_config_hash(config) if config else ""

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
            patience=scheduler_patience,
        )

        # Training state — these get overwritten by resume()
        self.start_epoch = 1
        self.global_step = 0
        self.best_val_f1 = 0.0
        self.best_epoch = 0

    def resume_from(self, checkpoint_path: str | Path) -> None:
        """Resume training state from a checkpoint.

        Restores model, optimizer, scheduler, epoch counter, best tracking,
        and RNG states. The next call to train() will continue from the
        correct epoch.
        """
        ckpt = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            device=str(self.device),
            restore_rng=True,
        )

        self.start_epoch = ckpt.get("epoch", 0) + 1  # Continue from NEXT epoch
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_f1 = ckpt.get("best_val_f1", 0.0)
        self.best_epoch = ckpt.get("best_epoch", 0)

        # Warn on config drift
        saved_hash = ckpt.get("config_hash", "")
        if saved_hash and self.config_hash and saved_hash != self.config_hash:
            logger.warning(
                f"Config hash mismatch: checkpoint={saved_hash}, "
                f"current={self.config_hash}. Training config may have changed."
            )

        logger.info(
            f"Resuming from epoch {self.start_epoch} "
            f"(step={self.global_step}, best_f1={self.best_val_f1:.4f}@{self.best_epoch})"
        )

    def train_epoch(self, epoch: int, total_epochs: int) -> dict:
        """Train for one epoch with step-level logging.

        Returns:
            Dictionary with train_loss and train_f1.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []
        epoch_start = time.time()
        steps_in_epoch = len(self.train_loader)

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
            self.global_step += 1

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

            # Step-level logging with ETA
            if (batch_idx + 1) % self.log_every_n_steps == 0 or (batch_idx + 1) == steps_in_epoch:
                elapsed = time.time() - epoch_start
                steps_done = batch_idx + 1
                steps_per_sec = steps_done / max(elapsed, 0.01)
                remaining_steps = steps_in_epoch - steps_done
                eta_s = remaining_steps / max(steps_per_sec, 0.01)

                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]["lr"]

                logger.info(
                    f"  Epoch {epoch}/{total_epochs} "
                    f"[{steps_done}/{steps_in_epoch}] "
                    f"loss={avg_loss:.4f} "
                    f"lr={lr:.6f} "
                    f"step/s={steps_per_sec:.1f} "
                    f"ETA={self._format_time(eta_s)}"
                )

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
        """Run the full training loop, with proper resume support.

        Args:
            epochs: Total number of training epochs (not remaining).

        Returns:
            Dictionary with final training summary.
        """
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Starting training: epochs {self.start_epoch}→{epochs} on {self.device}")
        logger.info(f"Model parameters: {param_count:,}")
        logger.info(f"Train batches/epoch: {len(self.train_loader):,}")
        logger.info(f"Val batches/epoch: {len(self.val_loader):,}")
        logger.info(f"Logging every {self.log_every_n_steps} steps")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
        if self.start_epoch > 1:
            logger.info(
                f"Resuming: best_f1={self.best_val_f1:.4f}@{self.best_epoch}, "
                f"step={self.global_step}"
            )

        training_start = time.time()
        history = []

        for epoch in range(self.start_epoch, epochs + 1):
            epoch_start = time.time()

            # ── Train ──
            train_metrics = self.train_epoch(epoch, epochs)

            # ── Validate ──
            val_metrics = evaluate(
                self.model, self.val_loader, self.criterion,
                self.device, self.class_names,
            )

            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            # ── Scheduler step ──
            self.scheduler.step(val_metrics["f1_weighted"])

            # ── Build log entry ──
            log_entry = {
                "epoch": epoch,
                "global_step": self.global_step,
                "lr": lr,
                "epoch_time_s": round(epoch_time, 2),
                **train_metrics,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1_weighted": val_metrics["f1_weighted"],
                "val_f1_macro": val_metrics["f1_macro"],
                "best_val_f1": self.best_val_f1,
                "best_epoch": self.best_epoch,
            }
            history.append(log_entry)

            # ── Append to JSONL (never overwrites) ──
            with open(self.metrics_log, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # ── Epoch summary ──
            total_elapsed = time.time() - training_start
            remaining_epochs = epochs - epoch
            eta_total = (total_elapsed / max(epoch - self.start_epoch + 1, 1)) * remaining_epochs

            logger.info(
                f"Epoch {epoch:>3}/{epochs} | "
                f"train_loss={train_metrics['train_loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_f1={val_metrics['f1_weighted']:.4f} | "
                f"best_f1={self.best_val_f1:.4f}@{self.best_epoch} | "
                f"lr={lr:.6f} | "
                f"time={self._format_time(epoch_time)} | "
                f"ETA={self._format_time(eta_total)}"
            )

            # ── Save best checkpoint ──
            if val_metrics["f1_weighted"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1_weighted"]
                self.best_epoch = epoch
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.global_step,
                    val_metrics["f1_weighted"], self.best_val_f1, self.best_epoch,
                    self.checkpoint_dir / "best.pt",
                    self.config_hash,
                )
                logger.info(f"  ★ New best model saved (f1={self.best_val_f1:.4f})")

            # ── Save latest (always) ──
            save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                epoch, self.global_step,
                val_metrics["f1_weighted"], self.best_val_f1, self.best_epoch,
                self.checkpoint_dir / "latest.pt",
                self.config_hash,
            )

            # ── Periodic save ──
            if epoch % self.save_every_n_epochs == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.global_step,
                    val_metrics["f1_weighted"], self.best_val_f1, self.best_epoch,
                    self.checkpoint_dir / f"epoch_{epoch:04d}.pt",
                    self.config_hash,
                )

        total_time = time.time() - training_start
        logger.info(
            f"Training complete in {self._format_time(total_time)}. "
            f"Best val_f1={self.best_val_f1:.4f} at epoch {self.best_epoch}"
        )

        return {
            "best_val_f1": self.best_val_f1,
            "best_epoch": self.best_epoch,
            "total_epochs": epochs,
            "total_time_s": round(total_time, 2),
            "history": history,
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            return f"{h}h{m:02d}m"


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
