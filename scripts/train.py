#!/usr/bin/env python3
"""Training entry point."""

import argparse
import sys

from unisplit.training.cli import cmd_train


def main():
    parser = argparse.ArgumentParser(description="Train UniSplit IoTCNN model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    cmd_train(args)


if __name__ == "__main__":
    main()
