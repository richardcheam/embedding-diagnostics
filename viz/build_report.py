"""Build the interactive HTML report from run logs.

Usage:
    python viz/build_report.py --tag phaseA_s0 --scorecard-tag phaseA --seeds 0,1,2,3,4

The curves come from a single seed and the scorecard from across-seed means.
Those are deliberately separate arguments: deriving the scorecard from the same
single seed would put a number on the page whose stated basis is wrong, which
is the specific failure this project exists to document.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_lens.report_html import build_html
from jepa_lens.runs import load_runs

ROOT = Path(__file__).resolve().parents[1]

COLLAPSED = "none_nostopgrad"
WEAKEST = "none_stopgrad"   # the weakest condition that still genuinely trains
HEALTHY = "ema_stopgrad"

# (metric key, display name, formatter, does a HIGHER value mean healthier?)
SCORECARD_ROWS = [
    ("total_variance", "total variance", "{:.4f}", True),
    ("mean_pairwise_cosine", "mean pairwise cosine", "{:.4f}", False),
    ("probe_accuracy_unscaled", "unscaled linear probe", "{:.3f}", True),
    ("rankme", "RankMe", "{:.2f}", True),
    ("retrieval_p10", "retrieval P@10 (cosine)", "{:.3f}", True),
    ("probe_accuracy", "standardized linear probe", "{:.3f}", True),
    ("participation_ratio", "participation ratio", "{:.2f}", True),
]


def final_records(experiments_dir: Path, tag: str, seeds: list[int]) -> dict[str, list[dict]]:
    """Each condition's final-checkpoint record, one per seed."""
    per_condition: dict[str, list[dict]] = {}
    for seed in seeds:
        for condition, records in load_runs(experiments_dir, f"{tag}_s{seed}").items():
            per_condition.setdefault(condition, []).append(max(records, key=lambda r: r["step"]))
    return per_condition


def _values(finals: dict[str, list[dict]], condition: str, key: str) -> list[float] | None:
    records = finals.get(condition)
    if not records or not all(isinstance(r.get(key), (int, float)) for r in records):
        return None
    return [float(r[key]) for r in records]


def build_scorecard(finals: dict[str, list[dict]]) -> list[dict]:
    """One row per diagnostic, verdicted against the pre-registered claim rule.

    The question a collapse diagnostic exists to answer is not "is the dead
    encoder worse than the best one" -- almost anything clears that -- but
    "does it rank the dead encoder below an encoder that is merely weak".
    So the comparison is against WEAKEST, the worst condition that still
    genuinely trains, and the margin must clear twice the across-seed standard
    deviation of the paired differences, which is the same rule the report
    applies to every other contrast.

    Three verdicts, because two would hide the interesting case: a metric can
    point the right way by a margin indistinguishable from noise, which is a
    different failure from pointing the wrong way.
    """
    rows = []
    for key, name, fmt, higher_is_healthier in SCORECARD_ROWS:
        dead = _values(finals, COLLAPSED, key)
        weak = _values(finals, WEAKEST, key)
        healthy = _values(finals, HEALTHY, key)
        if dead is None or weak is None or healthy is None or len(dead) != len(weak):
            continue

        dead_mean = sum(dead) / len(dead)
        weak_mean = sum(weak) / len(weak)
        points_right_way = (dead_mean < weak_mean) if higher_is_healthier else (
            dead_mean > weak_mean
        )

        diffs = [d - w for d, w in zip(dead, weak)]
        mean_diff = sum(diffs) / len(diffs)
        if len(diffs) > 1:
            var = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
            resolvable = abs(mean_diff) > 2 * (var ** 0.5)
        else:
            resolvable = False

        if not points_right_way:
            status, verdict = "blind", "BLIND — rates the dead encoder better"
        elif not resolvable:
            status, verdict = "unresolved", "cannot separate — margin inside seed noise"
        else:
            status, verdict = "correct", "separates them"

        rows.append(
            {
                "name": name,
                "collapsed": fmt.format(dead_mean),
                "weakest": fmt.format(weak_mean),
                "healthy": fmt.format(sum(healthy) / len(healthy)),
                "verdict": verdict,
                "status": status,
            }
        )
    rows.sort(key=lambda r: {"correct": 0, "unresolved": 1, "blind": 2}[r["status"]])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the interactive HTML report")
    parser.add_argument("--tag", default="phaseA_s0", help="tag supplying the curves (one seed)")
    parser.add_argument(
        "--scorecard-tag",
        default=None,
        help="tag prefix for the across-seed scorecard, e.g. phaseA; omit to skip the table",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4", help="seeds for --scorecard-tag")
    parser.add_argument("--experiments-dir", default=str(ROOT / "experiments"))
    parser.add_argument("--out", default=str(ROOT / "viz" / "dist" / "report.html"))
    parser.add_argument("--title", default="jepa-lens")
    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)
    runs = load_runs(experiments_dir, args.tag)
    if not runs:
        print(f"no runs found under {experiments_dir}/{args.tag}")
        return 1

    scorecard: list[dict] = []
    provenance = f"Curves: {args.tag}, one seed."
    if args.scorecard_tag:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        finals = final_records(experiments_dir, args.scorecard_tag, seeds)
        scorecard = build_scorecard(finals)
        if not scorecard:
            print(f"warning: no scorecard built — need {COLLAPSED}, {WEAKEST} and {HEALTHY}")
        else:
            provenance += (
                f" Scorecard: {args.scorecard_tag}, {len(seeds)} seeds"
                f" ({args.seeds}), mean of each condition's final checkpoint."
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        build_html(runs, args.title, scorecard=scorecard, provenance=provenance),
        encoding="utf-8",
    )
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path} ({size_kb:.0f} KB) from conditions: {', '.join(sorted(runs))}")
    if scorecard:
        for status in ("correct", "unresolved", "blind"):
            names = [r["name"] for r in scorecard if r["status"] == status]
            print(f"  {status:<11} {len(names)}: {', '.join(names) or '-'}")
    print(json.dumps({"curves": args.tag, "scorecard": args.scorecard_tag}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
