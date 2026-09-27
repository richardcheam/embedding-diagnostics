"""Build report figures from run logs.

Usage:
    python scripts/make_figures.py --tag main
"""

from __future__ import annotations

import argparse
from pathlib import Path

from embedding_diagnostics.figures import build_all_figures
from embedding_diagnostics.runs import load_runs

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build report figures from run logs")
    parser.add_argument("--tag", default="default")
    parser.add_argument("--experiments-dir", default=str(ROOT / "experiments"))
    parser.add_argument("--out-dir", default=str(ROOT / "report" / "figures"))
    args = parser.parse_args()

    runs = load_runs(Path(args.experiments_dir), args.tag)
    if not runs:
        print(f"no runs found under {args.experiments_dir}/{args.tag}")
        return 1

    written = build_all_figures(runs, Path(args.out_dir))
    print(f"wrote {len(written)} figures to {args.out_dir} from: {', '.join(sorted(runs))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
