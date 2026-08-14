"""Run every condition, sequentially or one job per GPU, across one or more seeds.

    python scripts/run_all_conditions.py --device cuda --tag main
    python scripts/run_all_conditions.py --device cuda --tag main --parallel

`--parallel` gives each condition its own GPU via CUDA_VISIBLE_DEVICES, so each
training process sees exactly one device and the training code runs unchanged.
There is deliberately no data-parallel sharding of a single condition: SIGReg
and the collapse diagnostics are batch-level statistics, so splitting a batch
across ranks would change what they measure. See docs/development-log.md.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from jepa_lens.hardware import plan_gpu_waves

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = [
    "ema_stopgrad",
    "none_stopgrad",
    "sigreg_stopgrad",
    "sigreg_nostopgrad",
    "proj_sigreg_stopgrad",
    "proj_sigreg_nostopgrad",
    "none_nostopgrad",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full condition grid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--tag", default="default")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="run conditions concurrently, one per GPU, instead of one after another",
    )
    parser.add_argument(
        "--base-config",
        default="base.yaml",
        help="base config under configs/ (base.yaml = CIFAR-10, bdd.yaml = BDD100K)",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="comma-separated run seeds; each seed writes to <tag>_s<seed> so "
        "load_runs and the figures keep working unchanged per seed",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="comma-separated GPU ids for --parallel (default: every visible GPU)",
    )
    return parser.parse_args()


def build_command(
    condition: str, seed: int, tag: str, args: argparse.Namespace
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--condition",
        condition,
        "--device",
        args.device,
        "--tag",
        tag,
        "--seed",
        str(seed),
        "--base-config",
        args.base_config,
    ]
    if args.total_steps is not None:
        command += ["--total-steps", str(args.total_steps)]
    if args.checkpoint_every is not None:
        command += ["--checkpoint-every", str(args.checkpoint_every)]
    return command


def build_jobs(args: argparse.Namespace) -> list[tuple[str, int, str]]:
    """(condition, seed, tag) triples for the whole grid.

    A single seed keeps the flat `<tag>/<condition>/` layout every existing
    tool reads. Multiple seeds write to `<tag>_s<seed>/<condition>/`, so
    `load_runs` and the figures keep working unchanged per seed and
    `aggregate_seeds.py` can glob the family.
    """
    seeds = [int(part) for part in str(args.seeds).split(",") if part.strip()]
    if len(seeds) == 1:
        return [(condition, seeds[0], args.tag) for condition in CONDITIONS]
    return [
        (condition, seed, f"{args.tag}_s{seed}")
        for seed in seeds
        for condition in CONDITIONS
    ]


def resolve_gpus(raw: str | None) -> list[int]:
    if raw:
        return [int(part) for part in raw.split(",") if part.strip()]
    import torch

    count = torch.cuda.device_count()
    if count == 0:
        raise SystemExit(
            "--parallel needs at least one visible GPU, but torch reports none. "
            "Run scripts/check_environment.py --device cuda to see why."
        )
    return list(range(count))


def run_sequential(args: argparse.Namespace) -> int:
    for condition, seed, tag in build_jobs(args):
        print(f"\n=== {condition} (seed {seed} -> {tag}) ===", flush=True)
        result = subprocess.run(build_command(condition, seed, tag, args), check=False)
        if result.returncode != 0:
            print(f"condition {condition} failed with code {result.returncode}", file=sys.stderr)
            return result.returncode
    return 0


def run_parallel(args: argparse.Namespace) -> int:
    gpus = resolve_gpus(args.gpus)
    jobs = build_jobs(args)
    labels = [f"{condition}@s{seed}" for condition, seed, _ in jobs]
    lookup = dict(zip(labels, jobs, strict=True))
    waves = plan_gpu_waves(labels, gpus)
    print(f"{len(jobs)} jobs across GPUs {gpus} in {len(waves)} wave(s)")

    failures: list[tuple[str, int]] = []
    for index, wave in enumerate(waves, start=1):
        running = []
        for label, gpu in wave:
            condition, seed, tag = lookup[label]
            run_dir = ROOT / "experiments" / tag / condition
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "train.log"

            environment = dict(os.environ)
            # The child sees exactly one device, so --device cuda inside means
            # this GPU and the training code needs no notion of rank.
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)

            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                build_command(condition, seed, tag, args),
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running.append((label, tag, condition, gpu, process, handle))
            print(f"  wave {index}: {label} -> GPU {gpu}  (log: {log_path})", flush=True)

        for label, tag, condition, gpu, process, handle in running:
            code = process.wait()
            handle.close()
            status = "ok" if code == 0 else f"FAILED ({code})"
            print(f"  wave {index}: {label} on GPU {gpu} {status}", flush=True)
            if code != 0:
                failures.append((label, code))

    if failures:
        print("", file=sys.stderr)
        for label, code in failures:
            print(f"{label} failed with code {code}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    args = parse_args()
    if args.parallel:
        if not str(args.device).startswith("cuda"):
            print(
                f"--parallel assigns one GPU per condition, but --device is {args.device!r}. "
                "Use --device cuda, or drop --parallel to run sequentially.",
                file=sys.stderr,
            )
            return 2
        return run_parallel(args)
    return run_sequential(args)


if __name__ == "__main__":
    raise SystemExit(main())
