"""Edge CLI entry points.

Commands:
    single   — Run single-sample inference
    replay   — Replay test set through edge→cloud pipeline
    benchmark — Run benchmark with timing
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from unisplit.shared.config import load_config
from unisplit.shared.logging import setup_logging


def cmd_replay(args: argparse.Namespace) -> None:
    """Replay test set through edge→cloud pipeline."""
    from unisplit.edge.benchmark import save_results
    from unisplit.edge.client import CloudClient
    from unisplit.edge.context import ContextExtractor
    from unisplit.edge.ingestion import ReplayFileSource
    from unisplit.edge.runner import EdgeRunner
    from unisplit.edge.simulator import EdgeSimulator
    from unisplit.policies import create_policy
    from unisplit.profiler.profile_store import load_profile

    logger = setup_logging(level="INFO", fmt="plain", name="unisplit.edge")
    config = load_config(args.config)
    ec = config.edge
    dc = config.dataset

    # Load feasibility profile
    profile = load_profile(ec.profile_path)
    feasible_ids = profile.feasible_split_ids
    logger.info(f"Feasible split IDs: {feasible_ids}")

    # Create components
    runner = EdgeRunner(ec.partition_dir)
    runner.load_partitions(feasible_ids)

    client = CloudClient(
        cloud_url=ec.cloud_url,
        timeout_seconds=ec.request_timeout_seconds,
    )

    if not client.health_check():
        logger.error(f"Cloud service at {ec.cloud_url} is not reachable")
        sys.exit(1)

    context = ContextExtractor(rtt_ewma_alpha=ec.rtt_ewma_alpha)

    policy = create_policy(ec.default_policy, feasible_ids)

    source = ReplayFileSource(
        features_path=f"{dc.processed_dir}/features.npy",
        labels_path=f"{dc.processed_dir}/labels.npy",
        indices_path=f"{dc.splits_dir}/test_indices.npy",
        max_samples=args.max_samples,
    )

    simulator = EdgeSimulator(
        runner=runner,
        client=client,
        context_extractor=context,
        policy=policy,
        feasible_split_ids=feasible_ids,
    )

    results = simulator.run(source, max_samples=args.max_samples)

    if args.output:
        save_results(results, args.output)

    client.close()


def cmd_single(args: argparse.Namespace) -> None:
    """Run single-sample inference for testing."""
    from unisplit.edge.client import CloudClient
    from unisplit.edge.runner import EdgeRunner

    logger = setup_logging(level="INFO", fmt="plain", name="unisplit.edge")
    config = load_config(args.config)
    ec = config.edge

    runner = EdgeRunner(ec.partition_dir)
    runner.load_partitions([args.split_id])

    # Generate synthetic sample
    x = np.random.randn(80).astype(np.float32)

    activation, edge_ms = runner.run(x, args.split_id)
    logger.info(f"Edge inference: {edge_ms:.3f}ms, activation shape={activation.shape}")

    if args.split_id < 9:
        client = CloudClient(cloud_url=ec.cloud_url)
        response, rtt = client.send_activation(activation, args.split_id)
        logger.info(
            f"Cloud response: class={response.predicted_label}, "
            f"cloud_ms={response.timing.total_ms:.3f}, rtt={rtt:.3f}ms"
        )
        client.close()
    else:
        logger.info(f"Local-only: predicted class={np.argmax(activation)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="UniSplit Edge CLI")
    subparsers = parser.add_subparsers(dest="command")

    # replay
    p_replay = subparsers.add_parser("replay", help="Replay test set")
    p_replay.add_argument("--config", default="configs/default.yaml")
    p_replay.add_argument("--max-samples", type=int, default=-1)
    p_replay.add_argument("--output", default=None, help="Output JSONL path")

    # single
    p_single = subparsers.add_parser("single", help="Single sample inference")
    p_single.add_argument("--config", default="configs/default.yaml")
    p_single.add_argument("--split-id", type=int, default=7)

    args = parser.parse_args()

    commands = {
        "replay": cmd_replay,
        "single": cmd_single,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
