"""Summarize a SIGReg weight sweep, one row per (lambda, condition).

    python scripts/summarize_sweep.py --tag sweep

Reads every `experiments/<tag>_lam*/` directory. Endpoints are pre-registered
rather than selected: the primary number is the FINAL-checkpoint unscaled probe
accuracy (the schedule is fixed before the run, so the final checkpoint is
selection-free). The best-over-checkpoints value is still shown, explicitly
labelled as selection-biased: the expected maximum of 21 checkpoints of pure
noise (sd 0.0025) is already +0.005, so "best" flatters every run.

There are deliberately NO binary learned?/collapsed? verdicts. A single
threshold misclassified this project's best representation (the 32k EMA run
ended at cosine 0.53 and 7x variance shrinkage while gaining +0.21 probe
accuracy), so the table reports the per-phenomenon numbers and leaves the call
to the seed-aggregated analysis:

    scale    : total-variance ratio (final / init); << 1 means contraction
    angular  : final mean pairwise cosine; -> 1 means concentration
    dimension: participation ratio and RankMe at the final step
    semantic : unscaled probe delta (final - init)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from jepa_lens.logging_utils import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
KERNEL_NOISE_FLOOR = 0.005  # same-seed rerun spread, measured (development log)
BINOMIAL_SE_5000 = 0.007  # sqrt(0.5*0.5/5000), the probe's own sampling error


def summarize_run(records: list[dict]) -> dict[str, float] | None:
    """Fixed-endpoint summary of one run's sorted records."""
    if not records or "probe_accuracy_unscaled" not in records[0]:
        return None
    probe = [r["probe_accuracy_unscaled"] for r in records]
    return {
        "steps": records[-1]["step"],
        "probe_init": probe[0],
        "probe_final": probe[-1],
        "probe_delta": probe[-1] - probe[0],
        "probe_best_biased": max(probe),
        "cos_final": records[-1]["mean_pairwise_cosine"],
        "totvar_ratio": records[-1]["total_variance"] / max(records[0]["total_variance"], 1e-12),
        "pr_final": records[-1].get("participation_ratio", records[-1].get("effective_rank")),
        "rankme_final": records[-1].get("rankme", float("nan")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a SIGReg weight sweep")
    parser.add_argument("--tag", default="sweep")
    parser.add_argument("--experiments-dir", default=str(ROOT / "experiments"))
    args = parser.parse_args()

    root = Path(args.experiments_dir)
    tag_dirs = sorted(
        (path for path in root.glob(f"{args.tag}_lam*") if path.is_dir()),
        key=lambda p: float(p.name.rsplit("lam", 1)[1]),
    )
    if not tag_dirs:
        print(f"no sweep directories matching {args.tag}_lam* under {root}")
        return 1

    header = (
        f"{'lambda':>7} {'condition':<24} {'probe i->f':>13} {'delta':>7} {'best*':>7} "
        f"{'cos_f':>7} {'var f/i':>8} {'PR_f':>7} {'RankMe':>7}"
    )
    print(header)
    print("-" * len(header))
    for tag_dir in tag_dirs:
        lam = tag_dir.name.rsplit("lam", 1)[1]
        for run_dir in sorted(tag_dir.iterdir()):
            metrics = run_dir / "metrics.jsonl"
            if not metrics.exists():
                continue
            row = summarize_run(sorted(read_jsonl(metrics), key=lambda r: r["step"]))
            if row is None:
                print(f"{lam:>7} {run_dir.name:<24}  (log predates the unscaled probe)")
                continue
            print(
                f"{lam:>7} {run_dir.name:<24} "
                f"{row['probe_init']:>6.3f}->{row['probe_final']:>5.3f} "
                f"{row['probe_delta']:>+7.3f} {row['probe_best_biased']:>7.3f} "
                f"{row['cos_final']:>7.3f} {row['totvar_ratio']:>8.3f} "
                f"{row['pr_final']:>7.2f} {row['rankme_final']:>7.2f}"
            )

    print()
    print("* best-over-checkpoints is selection-biased; the endpoint is the final checkpoint.")
    print(
        f"noise: kernel nondeterminism ~{KERNEL_NOISE_FLOOR}, probe binomial SE "
        f"~{BINOMIAL_SE_5000} at n=5000; single-seed deltas inside either are not results."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
