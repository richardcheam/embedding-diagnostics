"""Sweep the SIGReg loss weight across both SIGReg conditions, one job per GPU.

    python scripts/sweep_sigreg_weight.py --device cuda --lambdas 0.01,0.02,0.05,0.1 \
        --total-steps 2000 --checkpoint-every 100 --tag sweep

Lambdas are given in the reference LeJEPA parametrisation, where the loss is
`sigreg * lambda + other * (1 - lambda)`. This project uses the additive form
`prediction + weight * sigreg`, so `weight = lambda / (1 - lambda)`. Quoting
lambda keeps the numbers comparable with the reference, whose ImageNet-10
launcher sweeps 0.01, 0.02, 0.05 and 0.1.

Both SIGReg conditions are run at every lambda, because the question is not
only "does SIGReg help" but "does stop-gradient still matter at this lambda".
Results land in `experiments/<tag>_lam<lambda>/<condition>/`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from jepa_lens.hardware import plan_gpu_waves

ROOT = Path(__file__).resolve().parents[1]
SIGREG_CONDITIONS = ["sigreg_stopgrad", "sigreg_nostopgrad"]


def lambda_to_weight(lam: float) -> float:
    """Convert the reference's convex-combination lambda to our additive weight."""
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lambda must be in (0, 1), got {lam}")
    return lam / (1.0 - lam)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep the SIGReg loss weight")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lambdas", default="0.01,0.02,0.05,0.1")
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--tag", default="sweep")
    parser.add_argument("--gpus", default=None, help="comma-separated ids (default: all)")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument(
        "--conditions",
        default=",".join(SIGREG_CONDITIONS),
        help="which conditions to sweep",
    )
    return parser.parse_args()


def resolve_gpus(raw: str | None) -> list[int]:
    if raw:
        return [int(part) for part in raw.split(",") if part.strip()]
    import torch

    count = torch.cuda.device_count()
    if count == 0:
        raise SystemExit(
            "this sweep needs at least one visible GPU, but torch reports none. "
            "Run scripts/check_environment.py --device cuda to see why."
        )
    return list(range(count))


def main() -> int:
    args = parse_args()
    lambdas = [float(part) for part in args.lambdas.split(",") if part.strip()]
    conditions = [part.strip() for part in args.conditions.split(",") if part.strip()]
    gpus = resolve_gpus(args.gpus)

    jobs = [(condition, lam) for lam in lambdas for condition in conditions]
    labels = [f"{condition}@lam{lam}" for condition, lam in jobs]
    waves = plan_gpu_waves(labels, gpus, jobs_per_gpu=args.jobs_per_gpu)
    lookup = dict(zip(labels, jobs, strict=True))

    print(f"{len(jobs)} runs over GPUs {gpus} in {len(waves)} wave(s)")
    for lam in lambdas:
        print(f"  lambda {lam} -> sigreg_weight {lambda_to_weight(lam):.4f}")

    failures: list[tuple[str, int]] = []
    for index, wave in enumerate(waves, start=1):
        running = []
        for label, gpu in wave:
            condition, lam = lookup[label]
            tag = f"{args.tag}_lam{lam}"
            run_dir = ROOT / "experiments" / tag / condition
            run_dir.mkdir(parents=True, exist_ok=True)

            command = [
                sys.executable,
                str(ROOT / "scripts" / "train.py"),
                "--condition", condition,
                "--device", args.device,
                "--tag", tag,
                "--total-steps", str(args.total_steps),
                "--checkpoint-every", str(args.checkpoint_every),
                "--sigreg-weight", f"{lambda_to_weight(lam):.6f}",
            ]
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)

            handle = (run_dir / "train.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                command, env=environment, stdout=handle, stderr=subprocess.STDOUT
            )
            running.append((label, gpu, process, handle))
            print(f"  wave {index}: {label} -> GPU {gpu}", flush=True)

        for label, gpu, process, handle in running:
            code = process.wait()
            handle.close()
            status = "ok" if code == 0 else f"FAILED ({code})"
            print(f"  wave {index}: {label} {status}", flush=True)
            if code != 0:
                failures.append((label, code))

    if failures:
        print("", file=sys.stderr)
        for label, code in failures:
            print(f"{label} failed with code {code}", file=sys.stderr)
        return 1

    print(f"\ndone. compare with: uv run python scripts/summarize_sweep.py --tag {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
