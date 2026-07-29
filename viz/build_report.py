"""Build the interactive HTML report from run logs.

Usage:
    python viz/build_report.py --tag main
"""

from __future__ import annotations

import argparse
from pathlib import Path

from jepa_lens.report_html import build_html
from jepa_lens.runs import load_runs

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the interactive HTML report")
    parser.add_argument("--tag", default="default")
    parser.add_argument("--experiments-dir", default=str(ROOT / "experiments"))
    parser.add_argument("--out", default=str(ROOT / "viz" / "dist" / "report.html"))
    parser.add_argument("--title", default="jepa-lens")
    args = parser.parse_args()

    runs = load_runs(Path(args.experiments_dir), args.tag)
    if not runs:
        print(f"no runs found under {args.experiments_dir}/{args.tag}")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(runs, args.title), encoding="utf-8")
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path} ({size_kb:.0f} KB) from conditions: {', '.join(sorted(runs))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
