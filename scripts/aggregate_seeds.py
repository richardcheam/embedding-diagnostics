"""Per-condition descriptives and paired contrasts for a multi-seed family.

    python scripts/aggregate_seeds.py --tag phaseA

Reads `experiments/<tag>_s<seed>/<condition>/metrics.jsonl` and reports, at the
pre-registered final checkpoint:

  1. per condition  - seed count, seed IDs, per-seed values, mean, sample SD
  2. per contrast   - per-seed differences, mean, SD, standard error, and a
                      two-sided 95% Student-t confidence interval

Contrasts are the five pre-declared factorial comparisons in
`embedding_diagnostics.stats.CONTRASTS`. EMA appears only in the descriptives: it differs
from the shared-encoder conditions in two factors at once, so it cannot
estimate a stop-gradient-only effect.

Nothing here labels a result significant, real, learned or collapsed. The
interval covers training-seed variability conditional on the fixed evaluation
split (`data.eval_split_seed`) — not split choice, dataset or architecture.
Five contrasts times several endpoints is dozens of intervals with no
multiplicity correction applied; read them as estimates, not tests.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from embedding_diagnostics.logging_utils import read_jsonl
from embedding_diagnostics.runs import CONDITION_ORDER
from embedding_diagnostics.stats import CONTRASTS, paired_contrast, summarize_condition

ROOT = Path(__file__).resolve().parents[1]

ENDPOINTS = [
    ("probe_accuracy_unscaled", "probe_uns"),
    ("probe_balanced", "probe_bal"),
    ("retrieval_p10", "retr_p10"),
    ("total_variance", "totvar"),
    ("mean_pairwise_cosine", "cosine"),
    ("participation_ratio", "PR"),
    ("rankme", "RankMe"),
]


def collect(experiments_dir: Path, tag: str) -> dict[str, dict[int, dict]]:
    """{condition: {seed: final record}} across the <tag>_s<seed> family."""
    families: dict[str, dict[int, dict]] = {}
    pattern = re.compile(rf"^{re.escape(tag)}_s(\d+)$")
    for tag_dir in sorted(Path(experiments_dir).iterdir()):
        match = pattern.match(tag_dir.name)
        if not match or not tag_dir.is_dir():
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired across-seed analysis")
    parser.add_argument("--tag", required=True, help="family prefix; reads <tag>_s<seed>/")
    parser.add_argument("--experiments-dir", default=str(ROOT / "experiments"))
    parser.add_argument(
        "--endpoints",
        default=",".join(key for key, _ in ENDPOINTS),
        help="comma-separated endpoint keys to analyse",
    )
    args = parser.parse_args()

    runs = collect(Path(args.experiments_dir), args.tag)
    if not runs:
        print(f"no directories matching {args.tag}_s<seed> under {args.experiments_dir}")
        return 1

    labels = dict(ENDPOINTS)
    endpoints = [key for key in args.endpoints.split(",") if key.strip()]
    order = {name: index for index, name in enumerate(CONDITION_ORDER)}
    conditions = sorted(runs, key=lambda c: order.get(c, len(order)))

    print(f"family: {args.tag}   conditions: {len(conditions)}")
    steps = {
        record.get("step") for by_seed in runs.values() for record in by_seed.values()
    }
    print(f"final steps present: {sorted(s for s in steps if s is not None)}")

    print("\n=== per condition (final checkpoint) ===")
    header = f"{'condition':<24} {'endpoint':<12} {'n':>2} {'mean':>9} {'sd':>9}  per-seed"
    print(header)
    print("-" * len(header))
    for condition in conditions:
        for endpoint in endpoints:
            try:
                summary = summarize_condition(condition, endpoint, runs[condition])
            except KeyError:
                continue
            values = " ".join(f"{v:.4f}" for v in summary.values)
            print(
                f"{condition:<24} {labels.get(endpoint, endpoint):<12} {summary.n:>2} "
                f"{summary.mean:>9.4f} {summary.sd:>9.4f}  seeds{summary.seeds}: {values}"
            )

    print("\n=== paired contrasts, two-sided 95% Student-t ===")
    print("(interval covers training-seed variability at a fixed evaluation split;")
    print(" no multiplicity correction; no significance verdicts)")
    header = (
        f"{'contrast':<46} {'endpoint':<12} {'n':>2} {'mean diff':>10} "
        f"{'sd':>8} {'SE':>8} {'95% CI':>22}"
    )
    print(header)
    print("-" * len(header))
    for name_a, name_b, description in CONTRASTS:
        if name_a not in runs or name_b not in runs:
            print(f"{name_a} - {name_b:<24} SKIPPED: condition absent from this family")
            continue
        for endpoint in endpoints:
            try:
                result = paired_contrast(name_a, name_b, endpoint, runs)
            except (ValueError, KeyError) as error:
                print(f"{name_a} - {name_b} [{endpoint}] FAILED: {error}")
                continue
            interval = f"[{result.ci_low:+.4f}, {result.ci_high:+.4f}]"
            print(
                f"{result.contrast:<46} {labels.get(endpoint, endpoint):<12} "
                f"{result.n_pairs:>2} {result.mean_difference:>+10.4f} "
                f"{result.sd_difference:>8.4f} {result.standard_error:>8.4f} {interval:>22}"
            )
        print(f"    ^ {description}; per-seed differences above are the paired units")

    print("\nema_stopgrad appears only in the descriptives: it differs from the")
    print("shared-encoder conditions in two factors, so it is a reference baseline,")
    print("not a contrast arm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
