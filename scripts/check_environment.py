"""Report the training environment and flag anything that would break a run.

Run this first on a new machine:

    python scripts/check_environment.py --device cuda

Exits non-zero if the requested device cannot actually be used, so it can gate
a job script.
"""

from __future__ import annotations

import argparse

from jepa_lens.hardware import diagnose, format_report, probe_environment


def main() -> int:
    parser = argparse.ArgumentParser(description="Report and check the training environment")
    parser.add_argument(
        "--device",
        default="cuda",
        help="device the run intends to use; checks are CUDA-specific when this starts with cuda",
    )
    args = parser.parse_args()

    info = probe_environment()
    problems = diagnose(info, want_cuda=str(args.device).startswith("cuda"))
    print("environment:")
    print(format_report(info, problems))

    if problems:
        print(f"\n{len(problems)} problem(s) would prevent a run on {args.device!r}.")
        return 1
    print(f"\nOK for device {args.device!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
