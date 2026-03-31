"""Experiment orchestrator — runs multiple experiments from configs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from unisplit.experiments.replay import run_replay_experiment

logger = logging.getLogger("unisplit.experiments.runner")


def run_experiment_suite(
    config_paths: list[str],
    output_base: str = "experiments/results",
    max_samples: int = -1,
) -> dict[str, dict]:
    """Run a suite of experiments.

    Args:
        config_paths: List of experiment config YAML paths.
        output_base: Base output directory.
        max_samples: Max samples per experiment.

    Returns:
        Dict mapping experiment name to metrics report.
    """
    all_reports = {}

    for config_path in config_paths:
        logger.info(f"Running experiment: {config_path}")
        try:
            report = run_replay_experiment(
                config_path=config_path,
                max_samples=max_samples,
            )
            all_reports[report.get("experiment_name", config_path)] = report
        except Exception as e:
            logger.error(f"Experiment {config_path} failed: {e}")
            all_reports[config_path] = {"error": str(e)}

    # Save combined report
    combined_path = Path(output_base) / "combined_report.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(all_reports, f, indent=2)

    logger.info(f"Combined report saved to {combined_path}")
    return all_reports
