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
from jepa_lens.planning import build_jobs, parse_seeds, partition_jobs, preflight

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
        "--jobs-per-gpu",
        type=int,
        default=1,
        help="concurrent jobs per GPU. These models use ~1GB of a 24GB card, so "
        "raising this shortens a seeded matrix; the CPU saturates before the GPU does",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="discard existing results in the target directories (refused by default)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip jobs whose log already reaches --total-steps and rerun only the rest. "
        "For recovering a matrix that lost some jobs to a crash (OOM, preemption) "
        "without discarding the runs that finished",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="comma-separated GPU ids for --parallel (default: every visible GPU)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="override data.root for every job. Dataset location is a property of "
        "the machine, not the experiment — passing it here keeps machine-specific "
        "absolute paths out of the committed configs",
    )
    return parser.parse_args()


def build_command(job, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--condition",
        job.condition,
        "--device",
        args.device,
        "--tag",
        job.tag,
        "--seed",
        str(job.seed),
        "--base-config",
        args.base_config,
    ]
    if args.total_steps is not None:
        command += ["--total-steps", str(args.total_steps)]
    if args.checkpoint_every is not None:
        command += ["--checkpoint-every", str(args.checkpoint_every)]
    if args.data_root is not None:
        command += ["--data-root", args.data_root]
    # A resumed job is by definition replacing a partial log from a crashed
    # attempt; without this the child's anti-clobber guard rejects it.
    if args.resume or args.overwrite:
        command += ["--replace"]
    return command


def plan(args: argparse.Namespace):
    """Build the job list and refuse to proceed if any target is occupied."""
    jobs = build_jobs(CONDITIONS, parse_seeds(args.seeds), args.tag)

    if args.resume:
        if args.total_steps is None:
            raise SystemExit(
                "--resume needs --total-steps: whether a run finished is decided by "
                "whether its log reaches that step, and the runner cannot infer it."
            )
        done, jobs = partition_jobs(jobs, ROOT / "experiments", args.total_steps)
        print(f"resume: {len(done)} already at step {args.total_steps}, {len(jobs)} to run")
        for job in jobs:
            print(f"  rerun {job.label}")
        if not jobs:
            print("nothing to do; every job is complete")
            raise SystemExit(0)
        # Partial logs from the crashed attempts are what we are replacing.
        return jobs

    problems = preflight(jobs, ROOT / "experiments", overwrite=args.overwrite)
    if problems:
        print("refusing to start; nothing has been written:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        raise SystemExit(2)
    return jobs


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


def run_sequential(args: argparse.Namespace, jobs) -> int:
    # --gpus is honoured here too. It used to apply only under --parallel, so a
    # sequential run with --gpus 2 silently trained on GPU 0 -- the wrong device,
    # with no error, possibly colliding with someone else's job.
    environment = dict(os.environ)
    if args.gpus:
        gpus = resolve_gpus(args.gpus)
        if len(gpus) > 1:
            print(
                f"--gpus lists {len(gpus)} devices but this is a sequential run, which "
                "uses one. Pass --parallel to spread jobs across them, or name a single "
                "GPU.",
                file=sys.stderr,
            )
            return 2
        environment["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        print(f"pinning every job to GPU {gpus[0]}")

    for job in jobs:
        print(f"\n=== {job.condition} (seed {job.seed} -> {job.tag}) ===", flush=True)
        # Write train.log here too. Sequential runs used to leave it untouched,
        # so a stale log from an earlier crashed parallel attempt survived beside
        # fresh metrics -- two runs of the BDD pilot looked like OOM failures
        # when they had actually succeeded on the rerun.
        run_dir = job.run_dir(ROOT / "experiments")
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "train.log").open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                build_command(job, args),
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            print(f"{job.label} failed with code {result.returncode}", file=sys.stderr)
            return result.returncode
    return 0


def run_parallel(args: argparse.Namespace, jobs) -> int:
    gpus = resolve_gpus(args.gpus)
    labels = [job.label for job in jobs]
    lookup = dict(zip(labels, jobs, strict=True))
    waves = plan_gpu_waves(labels, gpus, jobs_per_gpu=args.jobs_per_gpu)
    print(
        f"{len(jobs)} jobs across GPUs {gpus} at {args.jobs_per_gpu}/GPU "
        f"in {len(waves)} wave(s)"
    )

    failures: list[tuple[str, int]] = []
    for index, wave in enumerate(waves, start=1):
        running = []
        for label, gpu in wave:
            job = lookup[label]
            run_dir = job.run_dir(ROOT / "experiments")
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "train.log"

            environment = dict(os.environ)
            # The child sees exactly one device, so --device cuda inside means
            # this GPU and the training code needs no notion of rank.
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)

            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                build_command(job, args),
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running.append((label, gpu, process, handle))
            print(f"  wave {index}: {label} -> GPU {gpu}  (log: {log_path})", flush=True)

        for label, gpu, process, handle in running:
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
    jobs = plan(args)
    if args.parallel:
        if not str(args.device).startswith("cuda"):
            print(
                f"--parallel assigns one GPU per condition, but --device is {args.device!r}. "
                "Use --device cuda, or drop --parallel to run sequentially.",
                file=sys.stderr,
            )
            return 2
        return run_parallel(args, jobs)
    return run_sequential(args, jobs)


if __name__ == "__main__":
    raise SystemExit(main())
