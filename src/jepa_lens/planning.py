"""Job planning and preflight validation for multi-run experiments.

Separated from the runner script so it is importable and testable, and so the
whole plan can be validated *before* any process starts or any file is opened.

The bug this exists to prevent: the runner used to open each job's `train.log`
with mode "w" before spawning the child, while the guard against clobbering an
existing run lives inside the child's `RunLogger`. A re-run into an occupied
tag therefore destroyed the previous `train.log` and only then failed. Nothing
should touch the filesystem until every planned job is known to be safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Job:
    """One planned training run."""

    condition: str
    seed: int
    tag: str

    @property
    def label(self) -> str:
        return f"{self.condition}@s{self.seed}"

    def run_dir(self, experiments_dir: Path) -> Path:
        return Path(experiments_dir) / self.tag / self.condition


def parse_seeds(raw: str) -> list[int]:
    """Parse and validate a comma-separated seed list.

    Rejects empties, duplicates and negatives. Negatives are refused because
    the directory convention is `<tag>_s<seed>` and the aggregation regex reads
    `_s(\\d+)$`, so a negative seed would produce a directory the analysis
    silently cannot find.
    """
    seeds = [int(part) for part in str(raw).split(",") if part.strip()]
    if not seeds:
        raise ValueError("no seeds given")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds in {seeds}")
    if any(seed < 0 for seed in seeds):
        raise ValueError(
            f"negative seeds are not supported: {seeds}. Run directories are "
            "named <tag>_s<seed> and aggregation matches _s(\\d+)$."
        )
    return seeds


def build_jobs(conditions: list[str], seeds: list[int], tag: str) -> list[Job]:
    """Every (condition, seed) pair, always in seed-explicit directories.

    Phase 2 uses `<tag>_s<seed>/<condition>/` unconditionally, including for a
    single seed. The previous behaviour — a flat `<tag>/` for one seed and
    seed-suffixed directories for several — meant the output layout depended on
    how a run was invoked, so an analysis script could not tell from a path
    whether it was looking at a one-seed or many-seed experiment. Readers still
    handle the older flat layout, which is where all Phase-1 evidence lives.
    """
    if not conditions:
        raise ValueError("no conditions given")
    jobs = [Job(condition, seed, f"{tag}_s{seed}") for seed in seeds for condition in conditions]
    labels = [job.label for job in jobs]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate job labels; conditions must be unique")
    return jobs


def preflight(
    jobs: list[Job],
    experiments_dir: Path,
    overwrite: bool = False,
) -> list[str]:
    """Return the reasons this plan is unsafe to run. Empty means go.

    Checks every job before any of them starts, so a collision in the last job
    of the last wave cannot be discovered after the first waves have already
    written results.

    A run directory is a blocker when it holds a non-empty `metrics.jsonl`:
    that is a completed or partial previous run, and continuing would either
    interleave two runs into one log or destroy the log of the first. An empty
    or absent directory is fine.
    """
    problems: list[str] = []
    for job in jobs:
        metrics = job.run_dir(experiments_dir) / "metrics.jsonl"
        if metrics.exists() and metrics.stat().st_size > 0 and not overwrite:
            problems.append(
                f"{job.label}: {metrics} already holds results. Choose a different "
                "--tag, delete that directory, or pass --overwrite to discard it."
            )
    return problems
