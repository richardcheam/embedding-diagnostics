"""The runner's job is to hand each child the flags it was given.

A flag that silently fails to thread through does not raise: the child falls
back to its config default and the run completes, wrong. For --data-root that
means a whole seeded matrix trained on whichever dataset happened to sit at the
default path. These tests exist because that failure is invisible in the logs.
"""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from jepa_lens.planning import Job  # noqa: E402
from run_all_conditions import build_command  # noqa: E402


def make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        device="cuda",
        total_steps=None,
        checkpoint_every=None,
        base_config="base.yaml",
        data_root=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def job():
    return Job(condition="ema_stopgrad", seed=3, tag="phaseB_s3")


def test_the_job_identity_reaches_the_child(job):
    command = build_command(job, make_args())
    assert command[command.index("--condition") + 1] == "ema_stopgrad"
    assert command[command.index("--seed") + 1] == "3"
    assert command[command.index("--tag") + 1] == "phaseB_s3"


def test_base_config_reaches_the_child(job):
    """Phase B differs from Phase A only by this flag, so losing it would run
    the BDD matrix on CIFAR-10 and look entirely normal."""
    command = build_command(job, make_args(base_config="bdd.yaml"))
    assert command[command.index("--base-config") + 1] == "bdd.yaml"


def test_data_root_reaches_the_child_when_given(job):
    command = build_command(job, make_args(data_root="/data/bdd/100k"))
    assert command[command.index("--data-root") + 1] == "/data/bdd/100k"


def test_data_root_is_absent_when_not_given(job):
    """Omitted rather than passed as None, so the config default governs."""
    assert "--data-root" not in build_command(job, make_args())


def test_optional_overrides_are_omitted_unless_set(job):
    command = build_command(job, make_args())
    assert "--total-steps" not in command
    assert "--checkpoint-every" not in command


def test_optional_overrides_are_passed_when_set(job):
    command = build_command(job, make_args(total_steps=8000, checkpoint_every=400))
    assert command[command.index("--total-steps") + 1] == "8000"
    assert command[command.index("--checkpoint-every") + 1] == "400"


def test_the_command_invokes_train_py_with_the_running_interpreter(job):
    """uv's venv is selected by sys.executable; hardcoding "python" would
    resolve to whatever is on PATH and miss the project's pinned torch."""
    command = build_command(job, make_args())
    assert command[0] == sys.executable
    assert command[1].endswith("train.py")


def test_train_py_accepts_every_flag_the_runner_emits(job):
    """The two scripts are separate argument parsers, so a flag can be added to
    one and not the other. This catches that directly rather than at run time."""
    import train

    command = build_command(
        job,
        make_args(data_root="/data/bdd/100k", total_steps=100, checkpoint_every=50),
    )
    original = sys.argv
    try:
        sys.argv = ["train.py"] + command[2:]
        args = train.parse_args()
    finally:
        sys.argv = original

    assert args.condition == "ema_stopgrad"
    assert args.seed == 3
    assert args.data_root == "/data/bdd/100k"
    assert args.total_steps == 100
