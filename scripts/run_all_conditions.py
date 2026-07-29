"""Run all four conditions in sequence.

Usage:
    python scripts/run_all_conditions.py --device cuda --tag main
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["ema_stopgrad", "sigreg_stopgrad", "sigreg_nostopgrad", "none_nostopgrad"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full condition grid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--tag", default="default")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for condition in CONDITIONS:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--condition",
            condition,
            "--device",
            args.device,
            "--tag",
            args.tag,
        ]
        if args.total_steps is not None:
            command += ["--total-steps", str(args.total_steps)]
        if args.checkpoint_every is not None:
            command += ["--checkpoint-every", str(args.checkpoint_every)]

        print(f"\n=== {condition} ===", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"condition {condition} failed with code {result.returncode}", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
