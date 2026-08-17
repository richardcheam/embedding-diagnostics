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


DATASET_LABELS = {"cifar10": "CIFAR-10 calibration bench", "bdd100k": "BDD100K driving scenarios"}


def dataset_label(experiments_dir: Path, tag: str) -> str:
    """Read the dataset from a run's own config rather than hardcoding it.

    The subtitle used to be a literal "CIFAR-10 calibration bench" and silently
    became false the first time the demo was rebuilt on BDD100K.
    """
    for config_path in sorted((experiments_dir / tag).glob("*/config.json")):
        name = json.loads(config_path.read_text())["data"].get("dataset", "")
        if name:
            return DATASET_LABELS.get(name, name)
    return "calibration bench"


def section1_note(finals: dict[str, list[dict]]) -> str:
    """State what the two default evaluations actually did, on THIS data.

    The original copy asserted the degenerate encoder was hard to pick out. That
    was measured on CIFAR-10, where it ranked second of seven on the
    standardized probe. On BDD100K it ranks last on both charts, so the same
    sentence would be false. This computes the ranking and the margin to the
    nearest genuinely-training condition, and says whether that margin clears
    twice the across-seed SD of the paired differences.
    """
    if COLLAPSED not in finals:
        return ""
    parts = []
    for key, label in (("probe_accuracy", "the standardized probe"),
                       ("retrieval_p10", "cosine retrieval")):
        dead = _values(finals, COLLAPSED, key)
        if dead is None:
            continue
        others = {c: _values(finals, c, key) for c in finals if c != COLLAPSED}
        others = {c: v for c, v in others.items() if v is not None}
        if not others:
            continue
        dead_mean = sum(dead) / len(dead)
        means = {c: sum(v) / len(v) for c, v in others.items()}
        rank = 1 + sum(1 for m in means.values() if m > dead_mean)
        nearest = min(means, key=lambda c: abs(means[c] - dead_mean))
        diffs = [d - w for d, w in zip(dead, others[nearest])]
        mean_diff = sum(diffs) / len(diffs)
        if len(diffs) > 1:
            sd = (sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)) ** 0.5
        else:
            sd = 0.0
        separable = abs(mean_diff) > 2 * sd
        parts.append(
            f"On <strong>{label}</strong> it ranks {rank} of {len(means) + 1} "
            f"({dead_mean:.4f}); its nearest neighbour is <code>{nearest}</code> at "
            f"{means[nearest]:.4f}, a gap of {abs(mean_diff):.4f} against a "
            f"2&times;SD spread of {2 * sd:.4f} &mdash; "
            + ("<strong>separable</strong>" if separable
               else "<strong>inside seed noise</strong>") + "."
        )
    return ('<span class="chip">ours</span>' + " ".join(parts)
            + " Whether these defaults <em>rank</em> a degenerate encoder correctly is the "
              "question; whether they flag it in absolute terms is a different one.")


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
    finals: dict[str, list[dict]] = {}
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
    note = section1_note(finals) if args.scorecard_tag else ""
    out_path.write_text(
        build_html(
            runs, args.title, scorecard=scorecard, provenance=provenance,
            dataset=dataset_label(experiments_dir, args.tag), section1_note=note,
        ),
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
