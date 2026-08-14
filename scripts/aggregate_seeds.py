"""Aggregate a multi-seed experiment family into mean +/- sd per condition.

    python scripts/aggregate_seeds.py --tag phaseA

Reads every `experiments/<tag>_s<seed>/<condition>/metrics.jsonl` written by
`run_all_conditions.py --seeds 0,1,2` and reports, per condition, the
across-seed mean and standard deviation of the pre-registered endpoints at the
FINAL checkpoint. This is the table results claims come from: a single-seed
difference is not a result, whatever its size.

The last line prints the decision rule from the Phase-2 spec: a difference is
claimed only if it clears 2x the seed-level sd AND the probe's binomial
standard error (~0.007 at n=5000 test samples).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from jepa_lens.logging_utils import read_jsonl
from jepa_lens.runs import CONDITION_ORDER

ROOT = Path(__file__).resolve().parents[1]

ENDPOINTS = [
    ("probe_accuracy_unscaled", "probe_uns"),
    ("probe_accuracy", "probe_std"),
    ("total_variance", "totvar"),
    ("mean_pairwise_cosine", "cosine"),
    ("participation_ratio", "PR"),
    ("rankme", "RankMe"),
]


def collect(experiments_dir: Path, tag: str) -> dict[str, dict[int, dict]]:
    """{condition: {seed: final_record}} for every seed directory of the family."""
    families: dict[str, dict[int, dict]] = {}
    pattern = re.compile(rf"^{re.escape(tag)}_s(\d+)$")
    for tag_dir in sorted(Path(experiments_dir).iterdir()):
        match = pattern.match(tag_dir.name)
        if not match:
            continue
        seed = int(match.group(1))
        for run_dir in sorted(tag_dir.iterdir()):
            metrics = run_dir / "metrics.jsonl"
            if not metrics.exists():
                continue
            records = sorted(read_jsonl(metrics), key=lambda r: r["step"])
            if records:
                families.setdefault(run_dir.name, {})[seed] = records[-1]
    return families


def aggregate(families: dict[str, dict[int, dict]]) -> list[dict]:
    """One row per condition: n, and mean/sd for each endpoint present."""
    rows = []
    for condition in sorted(families, key=lambda c: (CONDITION_ORDER + [c]).index(c)):
        by_seed = families[condition]
        row: dict = {"condition": condition, "n": len(by_seed), "seeds": sorted(by_seed)}
        for key, label in ENDPOINTS:
            values = [record[key] for record in by_seed.values() if key in record]
            if values:
                spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                row[label] = (float(np.mean(values)), spread)
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate seeds for an experiment family")
    parser.add_argument("--tag", required=True, help="family prefix; reads <tag>_s<seed>/")
    parser.add_argument("--experiments-dir", default=str(ROOT / "experiments"))
    args = parser.parse_args()

    families = collect(Path(args.experiments_dir), args.tag)
    if not families:
        print(f"no directories matching {args.tag}_s<seed> under {args.experiments_dir}")
        return 1

    rows = aggregate(families)
    header = f"{'condition':<24} {'n':>2} " + " ".join(f"{label:>16}" for _, label in ENDPOINTS)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for _, label in ENDPOINTS:
            if label in row:
                mean, sd = row[label]
                cells.append(f"{mean:>8.3f}±{sd:<6.3f}")
            else:
                cells.append(f"{'-':>16}")
        print(f"{row['condition']:<24} {row['n']:>2} " + " ".join(cells))

    print()
    print(
        "claim rule (pre-registered): a between-condition difference is real only if it "
        "clears 2x the seed sd shown here AND the probe binomial SE (~0.007 at n=5000)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
