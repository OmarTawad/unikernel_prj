"""Experiment replay runner — replays test set with a given config."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from unisplit.edge.benchmark import save_results
from unisplit.edge.client import CloudClient
from unisplit.edge.context import ContextExtractor
from unisplit.edge.ingestion import ReplayFileSource
from unisplit.edge.runner import EdgeRunner
from unisplit.edge.simulator import EdgeSimulator
from unisplit.experiments.metrics import compute_full_report
from unisplit.policies import create_policy
from unisplit.profiler.profile_store import load_profile
from unisplit.shared.config import load_config

logger = logging.getLogger("unisplit.experiments.replay")


def run_replay_experiment(
    config_path: str,
    output_dir: str | None = None,
    max_samples: int = -1,
) -> dict:
    """Run a replay experiment from a config file.

    Args:
        config_path: Path to experiment config YAML.
        output_dir: Override output directory.
        max_samples: Max samples to process.

    Returns:
        Full experiment metrics report.
    """
    config = load_config(config_path)
    ec = config.edge
    dc = config.dataset
    exp = config.experiment

    # Load feasibility profile
    profile = load_profile(ec.profile_path)
    feasible_ids = profile.feasible_split_ids
    logger.info(f"Experiment: {exp.name}")
    logger.info(f"Policy: {exp.policy}")
    logger.info(f"Feasible splits: {feasible_ids}")

    # Create components
    runner = EdgeRunner(ec.partition_dir)
    runner.load_partitions(feasible_ids)

    client = CloudClient(cloud_url=ec.cloud_url)

    context = ContextExtractor(rtt_ewma_alpha=ec.rtt_ewma_alpha)

    policy = create_policy(exp.policy, feasible_ids, **exp.policy_args)

    source = ReplayFileSource(
        features_path=f"{dc.processed_dir}/features.npy",
        labels_path=f"{dc.processed_dir}/labels.npy",
        indices_path=f"{dc.splits_dir}/test_indices.npy",
        max_samples=max_samples if max_samples > 0 else exp.num_samples,
    )

    simulator = EdgeSimulator(
        runner=runner,
        client=client,
        context_extractor=context,
        policy=policy,
        feasible_split_ids=feasible_ids,
    )

    # Run simulation
    results = simulator.run(source)

    # Compute metrics
    report = compute_full_report(results)
    report["experiment_name"] = exp.name
    report["policy"] = exp.policy

    # Save results
    out_dir = output_dir or exp.output_dir
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        save_results(results, f"{out_dir}/results.jsonl")
        with open(f"{out_dir}/report.json", "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {out_dir}/report.json")

    client.close()
    return report
