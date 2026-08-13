from jepa_lens.hardware import (
    EnvironmentInfo,
    diagnose,
    format_report,
    parse_driver_version,
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
