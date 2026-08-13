"""Summarize a SIGReg weight sweep into one table.

    python scripts/summarize_sweep.py --tag sweep

Reads every `experiments/<tag>_lam*/` directory and reports, per lambda and
condition, whether the representation learned anything and whether it collapsed.

The two questions the sweep exists to answer:

1. Does any lambda make a SIGReg condition beat its own random initialisation?
   Compared on `probe_accuracy_unscaled`, since the standardized probe cannot
   see scale collapse.
2. Does stop-gradient still matter at that lambda? Compared on final mean
   pairwise cosine and total variance, which register collapse directly.

Probe differences below about 0.005 are within the measured run-to-run noise
floor and should not be read as real. See docs/development-log.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from jepa_lens.logging_utils import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
NOISE_FLOOR = 0.005


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
        f"{'lambda':>7} {'condition':<20} {'init':>7} {'final':>7} {'peak':>7} "
        f"{'learned?':>9} {'cos':>7} {'totvar':>9} {'collapsed?':>11}"
    )
    print(header)
    print("-" * len(header))

    for tag_dir in tag_dirs:
        lam = tag_dir.name.rsplit("lam", 1)[1]
        for run_dir in sorted(tag_dir.iterdir()):
            metrics = run_dir / "metrics.jsonl"
            if not metrics.exists():
                continue
            records = sorted(read_jsonl(metrics), key=lambda r: r["step"])
            if not records:
                continue

            key = "probe_accuracy_unscaled"
            if key not in records[0]:
                print(f"{lam:>7} {run_dir.name:<20}  (log predates {key})")
                continue

            init = records[0][key]
            final = records[-1][key]
            peak = max(r[key] for r in records)
            learned = "yes" if peak > init + NOISE_FLOOR else "no"

            cosine = records[-1]["mean_pairwise_cosine"]
            variance = records[-1]["total_variance"]
            # Collapse judged on direction and scale together, since either alone
            # can mislead: cosine near 1 means every input maps the same way, and
            # a large drop in variance means the representation shrank to a point.
            shrank = variance < records[0]["total_variance"] / 4
            collapsed = "yes" if cosine > 0.5 or shrank else "no"

            print(
                f"{lam:>7} {run_dir.name:<20} {init:>7.3f} {final:>7.3f} {peak:>7.3f} "
                f"{learned:>9} {cosine:>7.3f} {variance:>9.2f} {collapsed:>11}"
            )

    print()
    print(f"'learned?' requires peak to beat init by more than {NOISE_FLOOR} (the noise floor).")
    print("'collapsed?' is cosine > 0.5 or total variance below a quarter of its initial value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
