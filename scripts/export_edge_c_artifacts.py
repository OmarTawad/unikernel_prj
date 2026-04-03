#!/usr/bin/env python3
"""Export edge artifacts in a C-friendly format."""

from __future__ import annotations

import argparse

from unisplit.edge_native import export_all_edge_splits_to_c, export_edge_split_to_c


def main() -> None:
    parser = argparse.ArgumentParser(description="Export edge artifacts for C runtime")
    parser.add_argument("--partitions-dir", default="partitions", help="Directory containing edge_k*/cloud_k* partitions")
    parser.add_argument("--split-id", type=int, default=7, help="Single split ID to export")
    parser.add_argument(
        "--out-dir",
        default="edge_native/artifacts/edge_k7_c",
        help="Output directory when exporting a single split",
    )
    parser.add_argument(
        "--out-root-dir",
        default="edge_native/artifacts/c_splits",
        help="Root output directory when exporting all splits",
    )
    parser.add_argument("--all", action="store_true", help="Export all supported edge splits")
    parser.add_argument("--model-version", default="v0.1.0", help="Model version tag")
    parser.add_argument("--source-checkpoint", default="checkpoints/best.pt", help="Source checkpoint path for metadata")
    parser.add_argument("--eps", type=float, default=1e-5, help="BatchNorm epsilon for C runtime")
    parser.add_argument("--no-reference", action="store_true", help="Skip reference input/activation export")
    args = parser.parse_args()

    if args.all:
        manifests = export_all_edge_splits_to_c(
            partitions_dir=args.partitions_dir,
            out_root_dir=args.out_root_dir,
            model_version=args.model_version,
            source_checkpoint=args.source_checkpoint,
            eps=args.eps,
            export_reference=not args.no_reference,
        )
        print(f"Exported {len(manifests)} splits under: {args.out_root_dir}")
        for split_id, path in manifests.items():
            print(f"  split {split_id}: {path}")
        return

    manifest_path = export_edge_split_to_c(
        partitions_dir=args.partitions_dir,
        split_id=args.split_id,
        out_dir=args.out_dir,
        model_version=args.model_version,
        source_checkpoint=args.source_checkpoint,
        eps=args.eps,
        export_reference=not args.no_reference,
    )
    print(f"Export complete (split {args.split_id}): {manifest_path}")


if __name__ == "__main__":
    main()
