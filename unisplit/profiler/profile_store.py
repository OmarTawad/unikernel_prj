"""Profile persistence — save and load feasibility reports as JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from unisplit.shared.schemas import FeasibilityReport


def save_profile(report: FeasibilityReport, path: str | Path) -> Path:
    """Save a feasibility report to JSON.

    Args:
        report: FeasibilityReport to persist.
        path: Output file path.

    Returns:
        Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)
    return path


def load_profile(path: str | Path) -> FeasibilityReport:
    """Load a feasibility report from JSON.

    Args:
        path: Path to JSON file.

    Returns:
        FeasibilityReport instance.

    Raises:
        FileNotFoundError: If profile file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return FeasibilityReport(**data)
