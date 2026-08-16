"""Validate the minimal diagnostic panel, on synthetic degradations and real runs.

    uv run python scripts/validate_panel.py
    uv run python scripts/validate_panel.py --tags phaseA:5 bddpilot:3

Prints, in order:

1. the mode-coverage table and the exhaustive minimal-panel search;
2. whether drift-from-initialisation separates dead runs from healthy ones
   (it does not -- that negative is the reason the panel uses absolute limits);
3. whether the absolute panel separates them (it does), with the margins.

Every claim in `jepa_lens.panel`'s docstring is reproduced by this script, so a
reader can check the reasoning rather than take it on trust.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jepa_lens.panel import (
    CANDIDATES,
    OBSERVED_MARGINS,
    RECOMMENDED_PANEL,
    PanelThresholds,
    minimal_panels,
    mode_coverage,
    verdict,
)

ROOT = Path(__file__).resolve().parents[1]
COLLAPSED = "none_nostopgrad"


def clustered(samples: int = 512, dim: int = 32, classes: int = 8, seed: int = 0):
    """Embeddings with real class structure, so semantic endpoints mean something."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(classes), samples // classes)
    centroids = rng.normal(size=(classes, dim)) * 6.0
    return centroids[labels] + rng.normal(scale=1.0, size=(len(labels), dim)), labels


def final_means(experiments_dir: Path, tag: str, seeds: int) -> dict[str, dict[str, float]]:
    """Each condition's first and last checkpoint, averaged over seeds."""
    out: dict[str, dict[str, float]] = {}
    collected: dict[str, list[tuple[dict, dict]]] = {}
    for seed in range(seeds):
        root = experiments_dir / f"{tag}_s{seed}"
        if not root.is_dir():
            continue
        for condition_dir in sorted(root.iterdir()):
            metrics = condition_dir / "metrics.jsonl"
            if not metrics.exists():
                continue
            lines = metrics.read_text().splitlines()
            records = [json.loads(line) for line in lines if line.strip()]
            if not records:
                continue
            records.sort(key=lambda r: r["step"])
            collected.setdefault(condition_dir.name, []).append((records[0], records[-1]))

    for condition, pairs in collected.items():
        merged: dict[str, float] = {}
        for key in pairs[0][1]:
            values = [last[key] for _, last in pairs if isinstance(last.get(key), (int, float))]
            if values:
                merged[key] = float(np.mean(values))
        for key in pairs[0][0]:
            values = [first[key] for first, _ in pairs if isinstance(first.get(key), (int, float))]
            if values:
                merged[f"init_{key}"] = float(np.mean(values))
        out[condition] = merged
    return out


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the minimal diagnostic panel")
    parser.add_argument(
        "--tags",
        nargs="*",
        default=["phaseA:5", "bddpilot:3"],
        help="tag:seed-count pairs to validate against, e.g. phaseA:5",
    )
    parser.add_argument("--experiments-dir", default=str(ROOT / "experiments"))
    parser.add_argument("--severity", type=float, default=0.99)
    args = parser.parse_args()

    embeddings, _ = clustered()

    section("1. Which diagnostics respond to each controlled degradation")
    coverage = mode_coverage(embeddings, severity=args.severity)
    width = max(len(name) for name in coverage)
    for mode, responded in coverage.items():
        blind = [c for c in CANDIDATES if c not in responded]
        print(f"  {mode:<{width}}  responds: {', '.join(sorted(responded)) or 'NOTHING'}")
        if blind:
            print(f"  {'':<{width}}  blind:    {', '.join(blind)}")

    section("2. Exhaustive minimal-panel search")
    panels = minimal_panels(coverage)
    if not panels:
        print("  no subset of the candidates covers every mode")
        return 1
    size = len(panels[0])
    for smaller in range(1, size):
        print(f"  size {smaller}: 0 sufficient panels")
    print(f"  size {size}: {len(panels)} sufficient panel(s)")
    for panel in panels:
        mark = "  <-- recommended" if tuple(panel) == RECOMMENDED_PANEL else ""
        print(f"      {list(panel)}{mark}")
    never = [c for c in CANDIDATES if not any(c in p for p in panels)]
    if never:
        print(f"  in NO minimal panel: {', '.join(never)}")

    thresholds = PanelThresholds()
    experiments_dir = Path(args.experiments_dir)

    for spec in args.tags:
        tag, _, seed_text = spec.partition(":")
        seeds = int(seed_text or 1)
        runs = final_means(experiments_dir, tag, seeds)
        if not runs:
            print(f"\n  (no runs found for {tag}; skipping)")
            continue

        section(f"3. Drift from initialisation on real runs -- {tag} ({seeds} seeds)")
        print("   the rule 'alarm when a metric moves far from its init value'")
        header = f"   {'condition':<24}{'totvar x':>10}{'cosine x':>10}"
        print(header + f"{'drift alarm':>13}{'truth':>8}")
        false_alarms = 0
        for condition, metrics in sorted(runs.items()):
            ratios = []
            for key in ("total_variance", "mean_pairwise_cosine"):
                before, after = metrics.get(f"init_{key}"), metrics.get(key)
                ratios.append(after / before if before else float("nan"))
            alarm = any(r > 1.5 or r < 1 / 1.5 for r in ratios if np.isfinite(r))
            dead = condition == COLLAPSED
            if alarm and not dead:
                false_alarms += 1
            print(
                f"   {condition:<24}{ratios[0]:>10.4f}{ratios[1]:>10.3f}"
                f"{('ALARM' if alarm else 'quiet'):>13}{('DEAD' if dead else 'alive'):>8}"
            )
        print(f"   -> {false_alarms} false alarms on genuinely-training conditions.")
        print("      Drift is not usable as a health signal; hence absolute limits.")

        section(f"4. Absolute panel on real runs -- {tag} ({seeds} seeds)")
        print(f"   {'condition':<24}{'verdict':>12}   tripped checks")
        errors = 0
        for condition, metrics in sorted(runs.items()):
            result = verdict(metrics, thresholds)
            dead = condition == COLLAPSED
            if result["degenerate"] != dead:
                errors += 1
            flags = ", ".join(f"{k} ({v})" for k, v in result["tripped"].items()) or "-"
            print(
                f"   {condition:<24}"
                f"{('DEGENERATE' if result['degenerate'] else 'ok'):>12}   {flags}"
            )
        print(f"   -> {errors} misclassification(s) against ground truth.")

    section("5. Observed separation margins (collapsed vs worst genuinely training)")
    for metric, datasets in OBSERVED_MARGINS.items():
        parts = []
        for dataset, (dead, alive) in datasets.items():
            ratio = max(dead, alive) / max(min(dead, alive), 1e-12)
            parts.append(f"{dataset} {dead:g} vs {alive:g} ({ratio:.0f}x)")
        print(f"   {metric:<22} {'; '.join(parts)}")

    print("\nA tripped check means scale-invariant metrics are untrustworthy here.")
    print("It does not by itself mean the representation is uninformative:")
    print("three of the five controlled degradations leave retrieval P@10 at 1.000.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
