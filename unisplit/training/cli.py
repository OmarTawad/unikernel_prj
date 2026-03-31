"""Training CLI entry points.

Commands:
    train     — Full training run (supports --resume)
    validate  — Evaluate on validation set
    test      — Evaluate on test set
    dry-run   — Validate architecture with synthetic data
    export    — Export partitions from checkpoint
"""

from __future__ import annotations

import argparse
import sys

import torch

from unisplit.model.cnn import IoTCNN
from unisplit.shared.config import load_config
from unisplit.shared.constants import CLASS_NAMES
from unisplit.shared.logging import setup_logging


def cmd_train(args: argparse.Namespace) -> None:
    """Run full training with proper resume support."""
    from unisplit.training.dataloader import create_dataloader
    from unisplit.training.dataset import CICIoT2023Dataset
    from unisplit.training.trainer import Trainer, seed_everything

    logger = setup_logging(level="INFO", fmt="plain", name="unisplit.trainer")
    config = load_config(args.config)
    tc = config.training
    dc = config.dataset

    seed_everything(tc.seed)
    device = torch.device(tc.device)

    # Load datasets
    train_ds = CICIoT2023Dataset(
        dc.processed_dir, dc.metadata_dir,
        split_file=f"{dc.splits_dir}/train_indices.npy",
        normalize=dc.normalize,
    )
    val_ds = CICIoT2023Dataset(
        dc.processed_dir, dc.metadata_dir,
        split_file=f"{dc.splits_dir}/val_indices.npy",
        normalize=dc.normalize,
    )

    train_loader = create_dataloader(train_ds, config=tc, shuffle=True)
    val_loader = create_dataloader(val_ds, config=tc, shuffle=False)

    # Model
    model = IoTCNN(
        num_features=config.model.num_features,
        num_classes=config.model.num_classes,
    )

    # Class weights
    class_weights = train_ds.get_class_weights() if tc.use_class_weights else None

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=tc.learning_rate,
        weight_decay=tc.weight_decay,
        class_weights=class_weights,
        checkpoint_dir=tc.checkpoint_dir,
        metrics_log=tc.metrics_log,
        scheduler_patience=tc.scheduler_patience,
        scheduler_factor=tc.scheduler_factor,
        save_every_n_epochs=tc.save_every_n_epochs,
        log_every_n_steps=tc.log_every_n_steps,
        class_names=CLASS_NAMES,
        config=config,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.resume_from(args.resume)

    result = trainer.train(tc.epochs)
    print(f"\n✓ Training complete. Best F1: {result['best_val_f1']:.4f} at epoch {result['best_epoch']}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Run validation evaluation."""
    from unisplit.training.checkpoint import load_checkpoint
    from unisplit.training.dataloader import create_dataloader
    from unisplit.training.dataset import CICIoT2023Dataset
    from unisplit.training.evaluator import evaluate

    logger = setup_logging(level="INFO", fmt="plain", name="unisplit.evaluator")
    config = load_config(args.config)
    dc = config.dataset
    device = torch.device(config.training.device)

    model = IoTCNN(config.model.num_features, config.model.num_classes)
    ckpt_path = args.checkpoint or f"{config.training.checkpoint_dir}/best.pt"
    load_checkpoint(ckpt_path, model, device=config.training.device)

    val_ds = CICIoT2023Dataset(
        dc.processed_dir, dc.metadata_dir,
        split_file=f"{dc.splits_dir}/val_indices.npy",
        normalize=dc.normalize,
    )
    val_loader = create_dataloader(val_ds, config=config.training, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate(model, val_loader, criterion, device, CLASS_NAMES)

    print(f"\nValidation Results:")
    print(f"  Loss:        {metrics['loss']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(f"  F1 Macro:    {metrics['f1_macro']:.4f}")


def cmd_test(args: argparse.Namespace) -> None:
    """Run test evaluation."""
    from unisplit.training.checkpoint import load_checkpoint
    from unisplit.training.dataloader import create_dataloader
    from unisplit.training.dataset import CICIoT2023Dataset
    from unisplit.training.evaluator import evaluate

    logger = setup_logging(level="INFO", fmt="plain", name="unisplit.evaluator")
    config = load_config(args.config)
    dc = config.dataset
    device = torch.device(config.training.device)

    model = IoTCNN(config.model.num_features, config.model.num_classes)
    ckpt_path = args.checkpoint or f"{config.training.checkpoint_dir}/best.pt"
    load_checkpoint(ckpt_path, model, device=config.training.device)

    test_ds = CICIoT2023Dataset(
        dc.processed_dir, dc.metadata_dir,
        split_file=f"{dc.splits_dir}/test_indices.npy",
        normalize=dc.normalize,
    )
    test_loader = create_dataloader(test_ds, config=config.training, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, criterion, device, CLASS_NAMES)

    print(f"\nTest Results:")
    print(f"  Loss:        {metrics['loss']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    print(f"  F1 Macro:    {metrics['f1_macro']:.4f}")


def cmd_dry_run(args: argparse.Namespace) -> None:
    """Architecture validation dry-run."""
    from unisplit.training.trainer import dry_run

    setup_logging(level="INFO", fmt="plain", name="unisplit.trainer")

    model = IoTCNN()
    device = torch.device("cpu")
    success = dry_run(model, device)
    sys.exit(0 if success else 1)


def cmd_export(args: argparse.Namespace) -> None:
    """Export partitions from checkpoint."""
    from unisplit.model.export import main as export_main
    export_main()


def main() -> None:
    parser = argparse.ArgumentParser(description="UniSplit Training CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # train
    p_train = subparsers.add_parser("train", help="Train the model")
    p_train.add_argument("--config", default="configs/default.yaml")
    p_train.add_argument("--resume", default=None,
                        help="Resume from checkpoint (e.g., checkpoints/latest.pt)")

    # validate
    p_val = subparsers.add_parser("validate", help="Validate on val set")
    p_val.add_argument("--config", default="configs/default.yaml")
    p_val.add_argument("--checkpoint", default=None)

    # test
    p_test = subparsers.add_parser("test", help="Test on test set")
    p_test.add_argument("--config", default="configs/default.yaml")
    p_test.add_argument("--checkpoint", default=None)

    # dry-run
    p_dry = subparsers.add_parser("dry-run", help="Architecture validation")

    # export
    p_export = subparsers.add_parser("export", help="Export partitions")
    p_export.add_argument("--checkpoint", required=True)
    p_export.add_argument("--output-dir", default="partitions")

    args = parser.parse_args()

    commands = {
        "train": cmd_train,
        "validate": cmd_validate,
        "test": cmd_test,
        "dry-run": cmd_dry_run,
        "export": cmd_export,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
