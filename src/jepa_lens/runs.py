"""Loading run logs and locating where a series leaves its baseline.

Shared by both renderers — the report figures and the interactive HTML page —
so they cannot disagree about which runs exist or which colour marks which
condition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .logging_utils import read_jsonl

CONDITION_ORDER = ["ema_stopgrad", "sigreg_stopgrad", "sigreg_nostopgrad", "none_nostopgrad"]
CONDITION_COLORS = {
    "ema_stopgrad": "#4C72B0",
    "sigreg_stopgrad": "#DD8452",
    "sigreg_nostopgrad": "#C44E52",
    "none_nostopgrad": "#8172B3",
}


def load_runs(experiments_dir: Path, tag: str) -> dict[str, list[dict]]:
    """Load every condition's records under `<experiments_dir>/<tag>/`."""
    runs: dict[str, list[dict]] = {}
    tag_dir = Path(experiments_dir) / tag
    for run_dir in sorted(tag_dir.iterdir()):
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.exists():
            runs[run_dir.name] = sorted(read_jsonl(metrics_path), key=lambda r: r["step"])
    return runs


def first_departure_step(
    steps: list[int],
    values: list[float],
    tolerance: float = 2.0,
) -> int | None:
    """First step where a series leaves its own early-training baseline.

    The baseline is the mean and standard deviation of the first quarter of the
    series. Departure is the first later point more than `tolerance` standard
    deviations away. Returns None if the series never departs.

    This is a deliberately simple detector. Part 2 compares departure steps
    across metrics, so what matters is that the same rule is applied to every
    series, not that the rule is sophisticated.
    """
    if len(values) < 4:
        return None
    baseline_count = max(2, len(values) // 4)
    baseline = np.array(values[:baseline_count], dtype=float)
    mean = baseline.mean()
    std = baseline.std()
    floor = abs(mean) * 0.01 + 1e-9
    if std < floor:
        std = floor
    for step, value in zip(steps[baseline_count:], values[baseline_count:], strict=True):
        if abs(value - mean) > tolerance * std:
            return int(step)
    return None
