"""CLI for exporting model partitions from a trained checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from unisplit.model.cnn import IoTCNN
from unisplit.model.partition import export_all_partitions
from unisplit.shared.config import load_config
from unisplit.shared.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export edge/cloud partitions from trained checkpoint"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint (.pt)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="partitions",
        help="Output directory for partition files"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--model-version", type=str, default="v0.1.0",
        help="Model version tag"
    )
    args = parser.parse_args()

    logger = setup_logging(level="INFO", fmt="plain", name="export")

    config = load_config(args.config)

    # Load model
    model = IoTCNN(
        num_features=config.model.num_features,
        num_classes=config.model.num_classes,
    )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle checkpoint format: either raw state_dict or wrapped
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    logger.info(f"Model loaded: {model.count_parameters()} parameters")
    logger.info(f"Exporting partitions to {args.output_dir}")

    result = export_all_partitions(
        model=model,
        output_dir=args.output_dir,
        model_version=args.model_version,
        source_checkpoint=str(checkpoint_path),
    )

    for split_id, paths in result.items():
        logger.info(f"  Split {split_id}: edge={paths['edge']}, cloud={paths['cloud']}")

    logger.info("✓ All partitions exported successfully")


if __name__ == "__main__":
    main()
