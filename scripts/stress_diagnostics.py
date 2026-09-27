"""Run the controlled degradation stress test and print a diagnostic-response table.

    python scripts/stress_diagnostics.py                     # synthetic clustered data
    python scripts/stress_diagnostics.py --tag phaseA_s0 --condition ema_stopgrad

Applies each known transformation at increasing severity and reports how every
diagnostic responds. The purpose is to expose each metric's invariances: where
a column does not move, that metric is blind to that degradation mode.

This is a controlled stress test on feature vectors, NOT ground truth for
natural training collapse — real degeneration mixes modes and interacts with
the optimiser. It bounds what a metric CAN detect, not what it will detect in
a run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from embedding_diagnostics.degradation import DEFAULT_SEVERITIES, TRANSFORMS, diagnose_transform

ROOT = Path(__file__).resolve().parents[1]

COLUMNS = [
    ("total_variance", "totvar", "{:>10.3e}"),
    ("mean_feature_std", "featstd", "{:>9.4f}"),
    ("mean_pairwise_cosine", "cosine", "{:>8.3f}"),
    ("participation_ratio", "PR", "{:>7.2f}"),
    ("rankme", "RankMe", "{:>7.2f}"),
    ("probe_balanced", "probe_bal", "{:>10.3f}"),
    ("retrieval_p10", "retr_p10", "{:>9.3f}"),
]


def synthetic(samples: int = 512, dim: int = 32, classes: int = 8, seed: int = 0):
    """Clustered embeddings with a known class structure to degrade."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(classes), samples // classes)
    centroids = rng.normal(size=(classes, dim)) * 6.0
    return centroids[labels] + rng.normal(scale=1.0, size=(len(labels), dim)), labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Controlled diagnostic stress test")
    parser.add_argument("--tag", help="use a real run's logged 2D projection instead of synthetic")
    parser.add_argument("--condition", default="ema_stopgrad")
    parser.add_argument("--experiments-dir", default=str(ROOT / "experiments"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="write the sweep to this JSON path")
    args = parser.parse_args()

    if args.tag:
        # Run logs store a 2D PCA projection, not full embeddings; enough to
        # show the transformations behaving, too few dimensions for the rank
        # columns to be interesting.
        metrics = Path(args.experiments_dir) / args.tag / args.condition / "metrics.jsonl"
        records = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        embeddings = np.array(records[-1]["projection"], dtype=np.float64)
        labels = None
        print(f"source: {metrics} (final checkpoint 2D projection, {embeddings.shape})")
        print("note: only 2 dimensions, so the rank columns are not informative here")
    else:
        embeddings, labels = synthetic(seed=args.seed)
        print(f"source: synthetic clustered embeddings {embeddings.shape}, 8 classes")

    active = [c for c in COLUMNS if labels is not None or c[0] not in
              {"probe_balanced", "retrieval_p10"}]
    header = f"{'transform':<20} {'sev':>5} " + " ".join(f"{label:>10}" for _, label, _ in active)
    rows_out = []

    for name in TRANSFORMS:
        print()
        print(header)
        print("-" * len(header))
        for severity in DEFAULT_SEVERITIES:
            record = diagnose_transform(
                embeddings, name, severity, labels=labels, seed=args.seed
            )
            record["transform"] = name
            rows_out.append(record)
            cells = []
            for key, _, fmt in active:
                value = record.get(key, float("nan"))
                cells.append(fmt.format(value).rjust(10))
            print(f"{name:<20} {severity:>5.2f} " + " ".join(cells))

    if args.out:
        Path(args.out).write_text(json.dumps(rows_out, indent=2))
        print(f"\nwrote {len(rows_out)} records to {args.out}")

    print("\nRead this as an invariance map: a column that does not move under a")
    print("transformation is blind to that degradation mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
