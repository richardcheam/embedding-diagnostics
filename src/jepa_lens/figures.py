"""Matplotlib figures for the LaTeX report, generated from run logs.

Figures are generated rather than hand-placed, so one cannot silently drift
from the run that produced it. The interactive HTML page consumes the same run
logs through a different renderer; neither renderer is the source of truth.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .runs import CONDITION_COLORS, CONDITION_ORDER, first_departure_step  # noqa: E402


def _plot_metric(runs: dict[str, list[dict]], key: str, ylabel: str, title: str, out_path: Path):
    figure, axis = plt.subplots(figsize=(6.0, 3.8))
    for condition in CONDITION_ORDER:
        records = runs.get(condition)
        if not records:
            continue
        # Tolerate logs written before this metric existed. Run logs are kept
        # in git across the life of the project, so a metric added later must
        # not make every earlier run unplottable.
        present = [r for r in records if key in r]
        if not present:
            continue
        steps = [r["step"] for r in present]
        values = [r[key] for r in present]
        axis.plot(
            steps,
            values,
            label=condition,
            color=CONDITION_COLORS[condition],
            linewidth=1.8,
        )
    axis.set_xlabel("training step")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_path)
    plt.close(figure)


def plot_probe_accuracy(runs: dict[str, list[dict]], out_dir: Path) -> None:
    _plot_metric(
        runs,
        "probe_accuracy",
        "linear probe accuracy",
        "Downstream probe accuracy vs training duration",
        out_dir / "probe_accuracy.pdf",
    )


def plot_effective_rank(runs: dict[str, list[dict]], out_dir: Path) -> None:
    _plot_metric(
        runs,
        "effective_rank",
        "effective rank (participation ratio)",
        "Embedding effective rank vs training duration",
        out_dir / "effective_rank.pdf",
    )


def plot_feature_std(runs: dict[str, list[dict]], out_dir: Path) -> None:
    _plot_metric(
        runs,
        "mean_feature_std",
        "mean feature std",
        "Embedding variance vs training duration",
        out_dir / "feature_std.pdf",
    )


def plot_departure_comparison(runs: dict[str, list[dict]], out_dir: Path) -> None:
    """Part 2 made visual: does a cheap metric depart before probe accuracy?"""
    # total_variance and probe_accuracy_unscaled are included because the pilot
    # showed effective_rank and standardized probe accuracy can move the wrong
    # way under collapse; a departure comparison built only on those two would
    # compare the timings of two blind instruments.
    metrics = [
        "effective_rank",
        "total_variance",
        "mean_feature_std",
        "mean_pairwise_cosine",
        "probe_accuracy",
        "probe_accuracy_unscaled",
    ]
    figure, axis = plt.subplots(figsize=(6.5, 4.2))

    labels: list[str] = []
    for offset, condition in enumerate(CONDITION_ORDER):
        records = runs.get(condition)
        if not records:
            continue
        steps = [r["step"] for r in records]
        for index, metric in enumerate(metrics):
            if metric not in records[0]:
                continue
            departure = first_departure_step(steps, [r[metric] for r in records])
            if departure is not None:
                axis.scatter(
                    departure,
                    index + offset * 0.12,
                    color=CONDITION_COLORS[condition],
                    label=condition if index == 0 else None,
                    s=40,
                )
        labels = metrics

    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel("step at which the series departs its baseline")
    axis.set_title("Departure step by metric")
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis="x")
    figure.tight_layout()
    figure.savefig(out_dir / "departure_comparison.pdf")
    plt.close(figure)


def plot_total_variance(runs: dict[str, list[dict]], out_dir: Path) -> None:
    """The scale companion to effective rank, which is scale-invariant."""
    _plot_metric(
        runs,
        "total_variance",
        "total variance (trace of covariance)",
        "Embedding total variance vs training duration",
        out_dir / "total_variance.pdf",
    )


def plot_probe_comparison(runs: dict[str, list[dict]], out_dir: Path) -> None:
    """Standardized against unstandardized probe accuracy.

    Standardizing rescales a collapsed encoder's numerical noise back to unit
    variance, so the standardized probe can rise while the representation dies.
    A gap opening between the two lines is itself a collapse signal.
    """
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    for condition in CONDITION_ORDER:
        records = runs.get(condition)
        if not records or "probe_accuracy_unscaled" not in records[0]:
            continue
        steps = [r["step"] for r in records]
        colour = CONDITION_COLORS[condition]
        axis.plot(
            steps,
            [r["probe_accuracy"] for r in records],
            color=colour,
            linewidth=1.8,
            label=f"{condition} (standardized)",
        )
        axis.plot(
            steps,
            [r["probe_accuracy_unscaled"] for r in records],
            color=colour,
            linewidth=1.4,
            linestyle="--",
            label=f"{condition} (unscaled)",
        )
    axis.set_xlabel("training step")
    axis.set_ylabel("linear probe accuracy")
    axis.set_title("Probe accuracy: standardized (solid) vs unscaled (dashed)")
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=6)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_dir / "probe_comparison.pdf")
    plt.close(figure)


def build_all_figures(runs: dict[str, list[dict]], out_dir: Path) -> list[Path]:
    """Write every report figure, returning the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_probe_accuracy(runs, out_dir)
    plot_effective_rank(runs, out_dir)
    plot_feature_std(runs, out_dir)
    plot_total_variance(runs, out_dir)
    plot_probe_comparison(runs, out_dir)
    plot_departure_comparison(runs, out_dir)
    return [
        out_dir / "probe_accuracy.pdf",
        out_dir / "effective_rank.pdf",
        out_dir / "feature_std.pdf",
        out_dir / "total_variance.pdf",
        out_dir / "probe_comparison.pdf",
        out_dir / "departure_comparison.pdf",
    ]
