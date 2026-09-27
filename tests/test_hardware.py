import pytest

from embedding_diagnostics.hardware import (
    EnvironmentInfo,
    dataloader_shm_bytes,
    diagnose,
    diagnose_shared_memory,
    format_report,
    parse_driver_version,
    plan_gpu_waves,
    shared_memory_limit,
)

TURING = (7, 5)
CU128_ARCHES = ["sm_75", "sm_80", "sm_86", "sm_90"]


def test_parse_driver_version_matches_the_number_torch_reports():
    """12080 is what appears in torch's 'driver is too old' error."""
    assert parse_driver_version(12080) == (12, 8)
    assert parse_driver_version(13000) == (13, 0)
    assert parse_driver_version(12040) == (12, 4)


def test_cuda_major_ahead_of_driver_is_reported():
    """The exact failure seen on the RTX 6000 box: cu130 build, 12.8 driver.

    CUDA is only minor-version compatible, so this can never work regardless of
    which GPU is installed.
    """
    info = EnvironmentInfo(
        torch_version="2.13.0+cu130",
        build_cuda="13.0",
        driver_cuda=(12, 8),
        cuda_available=False,
    )
    problems = diagnose(info, want_cuda=True)
    assert len(problems) == 1
    assert "13.0" in problems[0]
    assert "12.8" in problems[0]
    assert "cu128" in problems[0]


def test_matching_major_versions_are_accepted():
    """A 12.8 driver runs any 12.x build — that is the whole point of minor compat."""
    info = EnvironmentInfo(
        torch_version="2.9.0+cu124",
        build_cuda="12.4",
        driver_cuda=(12, 8),
        cuda_available=True,
        device_name="Quadro RTX 6000",
        capability=TURING,
        arch_list=CU128_ARCHES,
        device_count=4,
    )
    assert diagnose(info, want_cuda=True) == []


def test_missing_architecture_is_reported():
    """A wheel that dropped Turing leaves no kernels for the card."""
    info = EnvironmentInfo(
        torch_version="2.13.0+cu128",
        build_cuda="12.8",
        driver_cuda=(12, 8),
        cuda_available=True,
        device_name="Quadro RTX 6000",
        capability=TURING,
        arch_list=["sm_80", "sm_90", "sm_100"],
    )
    problems = diagnose(info, want_cuda=True)
    assert len(problems) == 1
    assert "sm_75" in problems[0]


def test_cpu_only_build_is_reported_when_cuda_requested():
    info = EnvironmentInfo(torch_version="2.9.0", build_cuda=None, cuda_available=False)
    problems = diagnose(info, want_cuda=True)
    assert len(problems) == 1
    assert "CPU-only" in problems[0]


def test_cpu_run_never_complains():
    """Requesting CPU must not be blocked by anything CUDA-related."""
    info = EnvironmentInfo(
        torch_version="2.13.0+cu130",
        build_cuda="13.0",
        driver_cuda=(12, 8),
        cuda_available=False,
    )
    assert diagnose(info, want_cuda=False) == []


def test_unavailable_cuda_surfaces_the_underlying_error():
    info = EnvironmentInfo(
        torch_version="2.9.0+cu124",
        build_cuda="12.4",
        driver_cuda=(12, 4),
        cuda_available=False,
        init_error="no CUDA-capable device is detected",
    )
    problems = diagnose(info, want_cuda=True)
    assert len(problems) == 1
    assert "no CUDA-capable device is detected" in problems[0]


def test_arch_list_suffixes_still_match():
    """torch reports entries like 'sm_90a'; the base architecture still counts."""
    info = EnvironmentInfo(
        torch_version="2.9.0+cu124",
        build_cuda="12.4",
        driver_cuda=(12, 4),
        cuda_available=True,
        capability=(9, 0),
        arch_list=["sm_80", "sm_90a"],
    )
    assert diagnose(info, want_cuda=True) == []


def test_report_renders_without_a_gpu():
    info = EnvironmentInfo(torch_version="2.9.0", build_cuda=None)
    text = format_report(info, ["something is wrong"])
    assert "torch" in text
    assert "cpu-only build" in text
    assert "PROBLEM: something is wrong" in text


def test_four_conditions_on_four_gpus_is_one_wave():
    conditions = ["ema_stopgrad", "sigreg_stopgrad", "sigreg_nostopgrad", "none_nostopgrad"]
    waves = plan_gpu_waves(conditions, [0, 1, 2, 3])
    assert len(waves) == 1
    assert waves[0] == [
        ("ema_stopgrad", 0),
        ("sigreg_stopgrad", 1),
        ("sigreg_nostopgrad", 2),
        ("none_nostopgrad", 3),
    ]


def test_fewer_gpus_than_conditions_splits_into_waves():
    conditions = ["a", "b", "c", "d", "e"]
    waves = plan_gpu_waves(conditions, [0, 1])
    assert [len(w) for w in waves] == [2, 2, 1]
    assert [c for wave in waves for c, _ in wave] == conditions


def test_no_gpu_exceeds_its_capacity_within_a_wave():
    """At the default of one job per GPU, no device is double-booked."""
    waves = plan_gpu_waves([f"c{i}" for i in range(9)], [0, 1, 2, 3])
    for wave in waves:
        assigned = [gpu for _, gpu in wave]
        assert len(assigned) == len(set(assigned))


def test_jobs_per_gpu_packs_that_many_and_no_more():
    """These models use ~1GB of 24GB, so packing is the point — but the
    scheduler must still respect the stated capacity exactly."""
    from collections import Counter

    waves = plan_gpu_waves([f"c{i}" for i in range(24)], [0, 1, 2, 3], jobs_per_gpu=3)
    assert len(waves) == 2  # 24 jobs / (4 GPUs x 3) = 2 waves
    for wave in waves:
        counts = Counter(gpu for _, gpu in wave)
        assert max(counts.values()) <= 3


def test_packing_reduces_the_wave_count_proportionally():
    jobs = [f"c{i}" for i in range(35)]  # 7 conditions x 5 seeds
    assert len(plan_gpu_waves(jobs, [0, 1, 2, 3], jobs_per_gpu=1)) == 9
    assert len(plan_gpu_waves(jobs, [0, 1, 2, 3], jobs_per_gpu=3)) == 3


def test_a_partial_final_wave_spreads_across_gpus():
    """Interleaved slots, so leftovers do not all pile onto GPU 0."""
    waves = plan_gpu_waves([f"c{i}" for i in range(6)], [0, 1, 2, 3], jobs_per_gpu=2)
    assert {gpu for _, gpu in waves[0]} == {0, 1, 2, 3}


def test_jobs_per_gpu_below_one_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="at least 1"):
        plan_gpu_waves(["a"], [0], jobs_per_gpu=0)


def test_every_condition_is_scheduled_exactly_once():
    conditions = [f"c{i}" for i in range(7)]
    scheduled = [c for wave in plan_gpu_waves(conditions, [0, 1, 2]) for c, _ in wave]
    assert sorted(scheduled) == sorted(conditions)


def test_single_gpu_runs_everything_sequentially():
    waves = plan_gpu_waves(["a", "b", "c"], [0])
    assert [len(w) for w in waves] == [1, 1, 1]


def test_empty_gpu_list_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="no GPUs"):
        plan_gpu_waves(["a"], [])


def test_lambda_to_weight_matches_the_reference_parametrisation():
    """LeJEPA uses sigreg*lam + other*(1-lam); we use prediction + weight*sigreg.

    Dividing the reference form through by (1-lam) gives weight = lam/(1-lam),
    so quoting lambda keeps our numbers comparable with the reference's sweep.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from sweep_sigreg_weight import lambda_to_weight

    assert abs(lambda_to_weight(0.05) - 0.052631) < 1e-5
    assert abs(lambda_to_weight(0.5) - 1.0) < 1e-12
    # The reference's swept range maps to roughly 0.01 - 0.11.
    assert 0.010 < lambda_to_weight(0.01) < 0.011
    assert 0.111 < lambda_to_weight(0.1) < 0.112


def test_lambda_outside_the_unit_interval_is_rejected():
    import sys
    from pathlib import Path

    import pytest

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from sweep_sigreg_weight import lambda_to_weight

    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="lambda must be in"):
            lambda_to_weight(bad)


# --- running one pinned build across differently-aged machines ----------


def gh200(build_cuda="12.8", arch_list=None):
    """A GH200: Hopper sm_90, Grace ARM host, CUDA 13.0 driver."""
    return EnvironmentInfo(
        torch_version=f"2.11.0+cu{build_cuda.replace('.', '')}",
        build_cuda=build_cuda,
        driver_cuda=(13, 0),
        cuda_available=True,
        device_name="NVIDIA GH200 144GB HBM3e",
        capability=(9, 0),
        arch_list=arch_list or ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"],
        device_count=2,
    )


def quadro(build_cuda="12.8", arch_list=None):
    """The other machine: Turing sm_75, CUDA 12.8 driver."""
    return EnvironmentInfo(
        torch_version=f"2.11.0+cu{build_cuda.replace('.', '')}",
        build_cuda=build_cuda,
        driver_cuda=(12, 8),
        cuda_available=True,
        device_name="Quadro RTX 6000",
        capability=(7, 5),
        arch_list=arch_list or ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"],
        device_count=4,
    )


def test_a_newer_driver_runs_an_older_build():
    """CUDA drivers are backward compatible: a 13.0 driver runs a cu128 build.
    Only the reverse fails. Without this the project would need a second pin
    for every newer machine."""
    assert diagnose(gh200(), want_cuda=True) == []


def test_the_pinned_build_runs_on_both_machines():
    """The whole reason for one pin: results must be comparable across the two
    boxes, which needs the same build on both."""
    assert diagnose(gh200(), want_cuda=True) == []
    assert diagnose(quadro(), want_cuda=True) == []


def test_upgrading_the_pin_to_cu130_would_break_the_older_machine():
    """The trap. cu130 looks like the natural choice for a CUDA 13.0 box, but a
    13.x build cannot run on the 12.8 driver -- the exact failure this project
    already hit once. cu128 is the only pin that serves both."""
    problems = diagnose(quadro(build_cuda="13.0"), want_cuda=True)
    assert problems and "cannot run" in problems[0]
    assert diagnose(gh200(build_cuda="13.0"), want_cuda=True) == []


def test_hopper_kernels_must_actually_be_compiled_in():
    """Driver compatibility is not enough: the build also has to carry sm_90.
    A wheel that dropped it would fail at the first kernel launch, not import."""
    without_hopper = ["sm_75", "sm_80", "sm_86"]
    problems = diagnose(gh200(arch_list=without_hopper), want_cuda=True)
    assert problems and "sm_90" in problems[0]


# --- shared memory, the container failure mode --------------------------

MIB = 2**20


def test_zero_workers_needs_no_shared_memory():
    """Loading in the main process never crosses a process boundary."""
    assert dataloader_shm_bytes(256, 128, num_workers=0) == 0
    assert diagnose_shared_memory(256, 128, num_workers=0, limit=1) == []


def test_the_estimate_matches_the_batch_arithmetic():
    """One BDD batch is 50 MiB, which alone exceeds a default 64 MB /dev/shm."""
    one_batch = dataloader_shm_bytes(256, 128, num_workers=1, prefetch=1)
    assert one_batch == 256 * 3 * 128 * 128 * 4
    assert one_batch / MIB == pytest.approx(48.0, abs=0.1)


def test_the_estimate_scales_with_workers_and_prefetch():
    base = dataloader_shm_bytes(256, 128, num_workers=1)
    assert dataloader_shm_bytes(256, 128, num_workers=8) == 8 * base


def test_the_container_default_is_diagnosed():
    """The exact configuration that failed on the GH200 box: 8 workers, batch
    256, 128px, against Docker's 64 MB default."""
    problems = diagnose_shared_memory(256, 128, num_workers=8, limit=64 * 10**6)
    assert len(problems) == 1
    message = problems[0]
    assert "shm-size" in message and "--num-workers" in message


def test_the_advice_names_a_worker_count_that_would_actually_fit():
    """Advice that still overflows would send the user round the loop twice."""
    limit = 200 * MIB
    problems = diagnose_shared_memory(256, 128, num_workers=8, limit=limit)
    suggested = int(problems[0].split("--num-workers ")[1].split()[0])
    assert suggested >= 1
    assert dataloader_shm_bytes(256, 128, num_workers=suggested) <= limit
    assert dataloader_shm_bytes(256, 128, num_workers=suggested + 1) > limit


def test_advice_when_not_even_one_worker_fits_says_zero_not_a_negative():
    """The GH200 case: 61 MiB of /dev/shm against a 96 MiB single-worker need.
    '--num-workers 0 or fewer' is not advice anyone can act on."""
    problems = diagnose_shared_memory(256, 128, num_workers=8, limit=64 * 10**6)
    assert "--num-workers 0 " in problems[0]
    assert "or fewer" not in problems[0].split("--num-workers")[1][:40]


def test_ample_shared_memory_is_silent():
    assert diagnose_shared_memory(256, 128, num_workers=8, limit=16 * 2**30) == []


def test_an_unmeasurable_filesystem_is_not_treated_as_a_pass_or_a_failure():
    """macOS has no /dev/shm and does not use it this way. Unknown must mean
    'no opinion', never 'fine' by omission of the check nor a false alarm."""
    assert shared_memory_limit("/definitely/not/a/path") is None
    assert diagnose_shared_memory(256, 128, num_workers=8, path="/definitely/not/a/path") == []


def test_smaller_images_need_proportionally_less():
    """CIFAR at 32px is 16x smaller per batch, which is why this never bit
    before BDD."""
    cifar = dataloader_shm_bytes(256, 32, num_workers=8)
    bdd = dataloader_shm_bytes(256, 128, num_workers=8)
    assert bdd == 16 * cifar
    assert diagnose_shared_memory(256, 32, num_workers=8, limit=64 * 10**6) == []
