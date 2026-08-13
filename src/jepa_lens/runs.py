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

    ASSUMES A FLAT BASELINE, and is wrong when that fails. On the 32k
    `ema_stopgrad` run the first quarter is the rapid-learning phase: unscaled
    probe accuracy sweeps 0.369 to 0.582 there, giving a baseline std of 0.065
    and a tolerance band of 0.396 to 0.654. The run's final value of 0.582 sits
    inside that band, so no departure is reported even though the probe had
    peaked at 0.603 and declined. For a series that learns, plateaus, then
    degrades, use `first_sustained_decline` instead.
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


def monotone_fraction(values: list[float]) -> float:
    """Fraction of consecutive steps moving in the series' dominant direction.

    Near 1.0 means the series barely reverses, so it has no meaningful turning
    point and any "decline from best" detector will fire at the first step and
    say nothing useful. Used to screen metrics out of a timing comparison.
    """
    series = np.asarray(values, dtype=float)
    if len(series) < 2:
        return 1.0
    deltas = np.diff(series)
    nonzero = deltas[deltas != 0]
    if nonzero.size == 0:
        return 1.0
    return float(max((nonzero > 0).mean(), (nonzero < 0).mean()))


def turning_point(steps: list[int], values: list[float], mode: str) -> int | None:
    """Step at which a series reaches its best value before reversing.

    `mode` is "max" for quantities where higher is better (probe accuracy) or
    "min" where lower is better (mean pairwise cosine). Returns None if the
    extremum is at either end, meaning the series never turned within the run.

    Preferred over `first_sustained_decline` for the timing comparison in Part
    2, because it makes no assumption about a metric having a fixed good
    direction — it only asks where the series reversed.
    """
    if len(values) < 3:
        return None
    series = np.asarray(values, dtype=float)
    index = int(series.argmax() if mode == "max" else series.argmin())
    if index in (0, len(series) - 1):
        return None
    return int(steps[index])


def first_sustained_decline(
    steps: list[int],
    values: list[float],
    higher_is_better: bool,
    margin: float | None = None,
    persistence: int = 2,
) -> int | None:
    """First step where a series turns away from its own running best and stays there.

    ONLY VALID FOR METRICS WITH A FIXED GOOD DIRECTION. Total variance is not
    one: on the 32k `ema_stopgrad` run it falls monotonically from 87.2 to 12.6
    across the whole run, including the phase where probe accuracy climbs from
    0.369 to 0.603. Falling variance there is healthy concentration, not
    degradation, and this detector reports a "decline" at step 1500 that means
    nothing. Screen candidates with `monotone_fraction` first, and prefer
    `turning_point` for timing comparisons.


    Written for the shape real runs actually have: improve, plateau, then
    degrade. `first_departure_step` cannot handle that, because it takes the
    early-training window as a stable baseline when in fact that window is where
    the metric moves most.

    Args:
        higher_is_better: True for quantities where degradation means falling
            (probe accuracy, total variance), False where it means rising
            (mean pairwise cosine).
        margin: how far past the running best counts as a turn. Defaults to 5%
            of the series' observed range, which auto-scales across metrics on
            very different scales. Pass an explicit value to use a known noise
            floor instead.
        persistence: how many consecutive checkpoints must stay past the margin.
            Guards against a single noisy point triggering a spurious detection;
            the returned step is where the streak began, not where it ended.

    Returns:
        The step at which sustained decline begins, or None if it never does.
    """
    if len(values) < 3:
        return None
    series = np.asarray(values, dtype=float)
    if margin is None:
        margin = 0.05 * float(series.max() - series.min())
    if margin <= 0:
        return None

    streak_start: int | None = None
    streak = 0
    best = series[0]
    for index in range(1, len(series)):
        best = max(best, series[index - 1]) if higher_is_better else min(best, series[index - 1])
        declined = (
            best - series[index] > margin
            if higher_is_better
            else series[index] - best > margin
        )
        if declined:
            if streak == 0:
                streak_start = int(steps[index])
            streak += 1
            if streak >= persistence:
                return streak_start
        else:
            streak = 0
            streak_start = None
    return None
