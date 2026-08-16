"""Environment probing and preflight checks for the training hardware.

A run on the wrong PyTorch build fails with a message that names a driver
version and nothing else, partway into whatever you were doing. Two mismatches
account for almost all of it, and both are knowable before training starts:

1. The CUDA *major* version PyTorch was built against exceeds what the driver
   supports. CUDA is minor-version compatible within a major version, so a 12.8
   driver runs any 12.x build, but cannot run a 13.x build at all.
2. The GPU's compute capability is absent from the arch list PyTorch was
   compiled for. Newer wheels drop older architectures, so a card that worked
   last year can stop working after a routine upgrade.

The diagnosis logic here is deliberately pure so it can be tested without a
GPU; only `probe_environment` touches torch's CUDA state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvironmentInfo:
    """What we could learn about the environment. Fields are None if unknown."""

    torch_version: str | None = None
    build_cuda: str | None = None
    driver_cuda: tuple[int, int] | None = None
    cuda_available: bool = False
    device_name: str | None = None
    capability: tuple[int, int] | None = None
    arch_list: list[str] = field(default_factory=list)
    device_count: int = 0
    init_error: str | None = None


def plan_gpu_waves(
    conditions: list[str], gpus: list[int], jobs_per_gpu: int = 1
) -> list[list[tuple[str, int]]]:
    """Schedule conditions onto GPU slots, `jobs_per_gpu` concurrent jobs each.

    Returns a list of waves; each wave is a list of (condition, gpu) pairs that
    run concurrently. Slots are interleaved across GPUs, so a partially filled
    final wave spreads over the devices rather than piling onto the first.

    `jobs_per_gpu` exists because these models are small: a CIFAR-10 condition
    occupies roughly 1 GB of a 24 GB card, so one-job-per-GPU leaves the device
    almost idle. Packing several jobs per GPU shortens the wall clock of a
    seeded matrix roughly linearly until something else saturates — and what
    saturates is the CPU, not the GPU: each job runs dataloader workers AND
    fits two sklearn linear probes at every checkpoint. Raise this until
    steps/sec stops improving, then stop.

    Still deliberately one condition per job rather than sharding a condition
    across GPUs: SIGReg and the collapse diagnostics are batch-level
    statistics, so splitting a batch across ranks changes what they measure.
    See docs/development-log.md.
    """
    if not gpus:
        raise ValueError("no GPUs to schedule onto")
    if jobs_per_gpu < 1:
        raise ValueError(f"jobs_per_gpu must be at least 1, got {jobs_per_gpu}")

    # Interleaved: [0, 1, 2, 3, 0, 1, 2, 3, ...] rather than [0, 0, 1, 1, ...].
    slots = [gpus[index % len(gpus)] for index in range(len(gpus) * jobs_per_gpu)]
    return [
        [(condition, slots[offset]) for offset, condition in enumerate(chunk)]
        for chunk in (
            conditions[start : start + len(slots)]
            for start in range(0, len(conditions), len(slots))
        )
    ]


# DataLoader workers hand batches back through POSIX shared memory. Containers
# ship a 64 MB /dev/shm by default, which is less than a single BDD batch, and
# the failure is a bare "unable to allocate shared memory(shm)" from inside a
# worker -- it names neither /dev/shm nor the batch size, so it reads like a
# torch bug rather than a container setting.
SHM_PATH = "/dev/shm"
PREFETCH_PER_WORKER = 2  # torch's default prefetch_factor


def shared_memory_limit(path: str = SHM_PATH) -> int | None:
    """Bytes available in the shared-memory filesystem, or None if absent.

    None means "not applicable here" (macOS has no /dev/shm and does not use it
    this way), not "unlimited" -- callers must not treat it as a pass.
    """
    try:
        stats = os.statvfs(path)
    except (OSError, ValueError):
        return None
    return stats.f_bsize * stats.f_blocks


def dataloader_shm_bytes(
    batch_size: int,
    image_size: int,
    num_workers: int,
    channels: int = 3,
    dtype_bytes: int = 4,
    prefetch: int = PREFETCH_PER_WORKER,
) -> int:
    """Peak shared memory a DataLoader needs, in bytes.

    Each worker keeps `prefetch` batches in flight, and every batch crosses the
    process boundary through /dev/shm. Zero workers means loading happens in the
    main process, so nothing is shared.
    """
    if num_workers <= 0:
        return 0
    per_batch = batch_size * channels * image_size * image_size * dtype_bytes
    return per_batch * num_workers * prefetch


def diagnose_shared_memory(
    batch_size: int,
    image_size: int,
    num_workers: int,
    limit: int | None = None,
    path: str = SHM_PATH,
) -> list[str]:
    """Problems that would make DataLoader workers fail on this machine.

    Pure apart from the optional probe, so it is testable without a container.
    """
    if num_workers <= 0:
        return []
    available = shared_memory_limit(path) if limit is None else limit
    if available is None:
        return []

    needed = dataloader_shm_bytes(batch_size, image_size, num_workers)
    if needed <= available:
        return []

    fits = available // dataloader_shm_bytes(batch_size, image_size, 1)
    if fits >= 1:
        worker_advice = f"pass --num-workers {fits} or fewer"
    else:
        # Not even one worker's prefetch fits, so no positive count helps.
        worker_advice = (
            "pass --num-workers 0 to load in the main process (no shared memory "
            "at all, but no prefetching either)"
        )
    return [
        f"DataLoader needs about {needed / 2**20:.0f} MiB of shared memory "
        f"({num_workers} workers x {PREFETCH_PER_WORKER} batches of "
        f"{batch_size}x{image_size}x{image_size}), but {path} holds "
        f"{available / 2**20:.0f} MiB. Workers will die with 'unable to allocate "
        f"shared memory(shm)'. Fix by any of: restart the container with "
        f"--shm-size=8g (or --ipc=host); {worker_advice}; or let the run fall "
        f"back to file-system sharing, which is automatic but slower."
    ]


def parse_driver_version(raw: int) -> tuple[int, int]:
    """Convert torch's packed driver version to (major, minor).

    Torch reports the driver as a single integer — 12080 means CUDA 12.8, which
    is the number that appears in the "driver is too old" error.
    """
    return raw // 1000, (raw % 1000) // 10


def _major(version: str | None) -> int | None:
    if not version:
        return None
    try:
        return int(version.split(".")[0])
    except ValueError:
        return None


def diagnose(info: EnvironmentInfo, want_cuda: bool) -> list[str]:
    """Return human-readable problems that would break a CUDA run.

    An empty list means nothing detectable is wrong. Pure function: it reads
    only `info`, so it is testable without a GPU.
    """
    problems: list[str] = []
    if not want_cuda:
        return problems

    if info.build_cuda is None:
        problems.append(
            "This PyTorch is a CPU-only build (torch.version.cuda is None), so --device cuda "
            "cannot work. Install a CUDA build matching your driver."
        )
        return problems

    build_major = _major(info.build_cuda)
    if info.driver_cuda is not None and build_major is not None:
        driver_major, driver_minor = info.driver_cuda
        if build_major > driver_major:
            problems.append(
                f"PyTorch was built against CUDA {info.build_cuda} but the driver supports at "
                f"most CUDA {driver_major}.{driver_minor}. CUDA is only minor-version "
                f"compatible, so a CUDA {build_major}.x build cannot run on a "
                f"{driver_major}.x driver. Install a torch built for CUDA {driver_major}.x "
                f"(e.g. --index-url https://download.pytorch.org/whl/cu{driver_major}"
                f"{driver_minor}) or update the driver."
            )

    if info.capability is not None and info.arch_list:
        tag = f"sm_{info.capability[0]}{info.capability[1]}"
        if not any(entry.startswith(tag) for entry in info.arch_list):
            problems.append(
                f"The GPU reports compute capability {tag}, which is not in this PyTorch's "
                f"compiled arch list ({', '.join(info.arch_list)}). Kernels for this card were "
                "not built. Install an older torch that still ships that architecture."
            )

    if not info.cuda_available and not problems:
        detail = f" torch reported: {info.init_error}" if info.init_error else ""
        problems.append(f"torch.cuda.is_available() is False for an unrecognised reason.{detail}")

    return problems


def probe_environment() -> EnvironmentInfo:
    """Collect what torch can tell us, tolerating a CUDA stack that fails to init."""
    import torch

    info = EnvironmentInfo(torch_version=torch.__version__, build_cuda=torch.version.cuda)

    getter = getattr(torch._C, "_cuda_getDriverVersion", None)
    if getter is not None:
        try:
            raw = getter()
            if raw:
                info.driver_cuda = parse_driver_version(int(raw))
        except Exception:  # noqa: BLE001 - probing must never be fatal
            pass

    try:
        info.arch_list = list(torch.cuda.get_arch_list())
    except Exception:  # noqa: BLE001
        info.arch_list = []

    try:
        info.cuda_available = torch.cuda.is_available()
    except Exception as error:  # noqa: BLE001
        info.init_error = str(error).strip().splitlines()[0] if str(error) else None

    if info.cuda_available:
        try:
            info.device_count = torch.cuda.device_count()
            info.device_name = torch.cuda.get_device_name(0)
            info.capability = torch.cuda.get_device_capability(0)
        except Exception as error:  # noqa: BLE001
            info.init_error = str(error).strip().splitlines()[0] if str(error) else None
    elif info.init_error is None:
        # is_available() swallows the underlying error, so provoke it to get a
        # message worth printing.
        try:
            torch.cuda.init()
        except Exception as error:  # noqa: BLE001
            info.init_error = str(error).strip().splitlines()[0] if str(error) else None

    return info


def format_report(info: EnvironmentInfo, problems: list[str]) -> str:
    """Render the probe as a short block suitable for a run log."""
    driver = f"{info.driver_cuda[0]}.{info.driver_cuda[1]}" if info.driver_cuda else "?"
    rows: list[tuple[str, Any]] = [
        ("torch", info.torch_version),
        ("built for CUDA", info.build_cuda or "cpu-only build"),
        ("driver CUDA", driver),
        ("cuda available", info.cuda_available),
        ("device", info.device_name or "-"),
        ("device count", info.device_count),
        (
            "capability",
            f"sm_{info.capability[0]}{info.capability[1]}" if info.capability else "-",
        ),
        ("arch list", ", ".join(info.arch_list) if info.arch_list else "-"),
    ]
    width = max(len(label) for label, _ in rows)
    lines = [f"  {label.ljust(width)} : {value}" for label, value in rows]
    if info.init_error:
        lines.append(f"  {'init error'.ljust(width)} : {info.init_error}")
    if problems:
        lines.append("")
        for problem in problems:
            lines.append(f"  PROBLEM: {problem}")
    return "\n".join(lines)


def require_device(device: str) -> None:
    """Fail fast, with an explanation, if `device` cannot actually be used.

    Called at the start of a run so a mismatched build costs seconds rather
    than surfacing partway through training.
    """
    want_cuda = str(device).startswith("cuda")
    info = probe_environment()
    problems = diagnose(info, want_cuda=want_cuda)
    print("environment:")
    print(format_report(info, problems))
    if problems:
        raise SystemExit(
            f"\nRefusing to start on device {device!r}: the environment cannot run it. "
            "See the problems listed above."
        )
