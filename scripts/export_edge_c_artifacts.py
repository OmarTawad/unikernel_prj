#!/usr/bin/env python3
"""Export edge split-7 artifacts in a C-friendly format."""

from __future__ import annotations

import argparse
from pathlib import Path

from unisplit.edge_native import export_edge_k7_to_c


def main() -> None:
    parser = argparse.ArgumentParser(description="Export split-7 edge artifacts for C runtime")
    parser.add_argument("--partitions-dir", default="partitions", help="Directory containing edge_k*/cloud_k* partitions")
    parser.add_argument("--out-dir", default="edge_native/artifacts/edge_k7_c", help="Output directory for C artifacts")
    parser.add_argument("--model-version", default="v0.1.0", help="Model version tag")
    parser.add_argument("--source-checkpoint", default="checkpoints/best.pt", help="Source checkpoint path for metadata")
    parser.add_argument("--eps", type=float, default=1e-5, help="BatchNorm epsilon for C runtime")
    parser.add_argument("--no-reference", action="store_true", help="Skip reference input/activation export")
    args = parser.parse_args()

    manifest_path = export_edge_k7_to_c(
        partitions_dir=args.partitions_dir,
        out_dir=args.out_dir,
        model_version=args.model_version,
        source_checkpoint=args.source_checkpoint,
        eps=args.eps,
        export_reference=not args.no_reference,
    )

    print(f"Export complete: {manifest_path}")


if __name__ == "__main__":
    main()
