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

from embedding_diagnostics.planning import Job  # noqa: E402
from run_all_conditions import build_command  # noqa: E402


def make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        device="cuda",
        total_steps=None,
        checkpoint_every=None,
        base_config="base.yaml",
        data_root=None,
        num_workers=None,
        resume=False,
        overwrite=False,
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


def test_resume_tells_the_child_to_replace_its_partial_log(job):
    """A resumed job is replacing a crashed attempt's log. Without --replace the
    child's anti-clobber guard rejects it and the resume silently does nothing."""
    assert "--replace" in build_command(job, make_args(resume=True))


def test_overwrite_also_replaces(job):
    assert "--replace" in build_command(job, make_args(overwrite=True))


def test_a_normal_run_does_not_replace_anything(job):
    """--replace deletes files, so it must never appear by default."""
    assert "--replace" not in build_command(job, make_args())


# --- device pinning -----------------------------------------------------


def test_parallel_pins_each_child_to_its_assigned_gpu():
    """CUDA_VISIBLE_DEVICES is how a job is confined to one card. The child then
    sees it as device 0, so --device cuda inside means the assigned GPU."""
    from embedding_diagnostics.hardware import plan_gpu_waves

    waves = plan_gpu_waves(["a", "b", "c"], [2], jobs_per_gpu=2)
    assert waves == [[("a", 2), ("b", 2)], [("c", 2)]]


def test_sequential_writes_a_train_log_beside_the_metrics(tmp_path, monkeypatch, job):
    """Regression: sequential runs left train.log untouched, so a stale log from
    an earlier crashed attempt survived beside fresh metrics. Two BDD pilot runs
    read as OOM failures when the rerun had actually succeeded."""
    import subprocess

    import run_all_conditions as runner

    monkeypatch.setattr(runner, "ROOT", tmp_path)

    def fake_run(command, env=None, stdout=None, stderr=None, check=False):
        stdout.write("child output\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    args = make_args()
    args.gpus = None
    assert runner.run_sequential(args, [job]) == 0

    log = tmp_path / "experiments" / job.tag / job.condition / "train.log"
    assert log.read_text() == "child output\n"


def test_sequential_truncates_a_stale_log_rather_than_appending(tmp_path, monkeypatch, job):
    import subprocess

    import run_all_conditions as runner

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    run_dir = tmp_path / "experiments" / job.tag / job.condition
    run_dir.mkdir(parents=True)
    (run_dir / "train.log").write_text("OLD CRASH: torch.OutOfMemoryError\n")

    def fake_run(command, env=None, stdout=None, stderr=None, check=False):
        stdout.write("fresh run\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    args = make_args()
    args.gpus = None
    runner.run_sequential(args, [job])
    assert "OutOfMemoryError" not in (run_dir / "train.log").read_text()


def test_replace_does_not_delete_train_log():
    """The parallel runner holds an open handle on train.log and passes it as the
    child's stdout. Unlinking it in the child would leave the parent writing to a
    deleted inode, losing the log of the run that is happening right now."""
    source = (Path(__file__).resolve().parents[1] / "scripts" / "train.py").read_text()
    replace_block = source.split("if args.replace:")[1].split("with RunLogger")[0]
    assert "train.log" not in replace_block.replace("# NOT train.log", "")


def test_sequential_pins_to_a_single_named_gpu(tmp_path, monkeypatch, job):
    """Regression: --gpus used to apply only under --parallel, so a sequential
    run with --gpus 2 trained on GPU 0 with no error at all."""
    import subprocess

    import run_all_conditions as runner

    seen = {}

    monkeypatch.setattr(runner, "ROOT", tmp_path)

    def fake_run(command, env=None, stdout=None, stderr=None, check=False):
        seen["device"] = (env or {}).get("CUDA_VISIBLE_DEVICES")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    args = make_args()
    args.gpus = "2"
    assert runner.run_sequential(args, [job]) == 0
    assert seen["device"] == "2"


def test_sequential_refuses_several_gpus_rather_than_picking_one(monkeypatch, job):
    """Silently using the first of four would be the same class of bug."""
    import run_all_conditions as runner

    monkeypatch.setattr(
        runner.subprocess, "run", lambda *a, **k: pytest.fail("a process was started")
    )
    args = make_args()
    args.gpus = "0,1,2,3"
    assert runner.run_sequential(args, [job]) == 2


def test_sequential_without_gpus_leaves_the_environment_alone(tmp_path, monkeypatch, job):
    """No --gpus means inherit whatever the shell set, including an existing
    CUDA_VISIBLE_DEVICES the user exported themselves."""
    import subprocess

    import run_all_conditions as runner

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    seen = {}

    def fake_run(command, env=None, stdout=None, stderr=None, check=False):
        seen["device"] = (env or {}).get("CUDA_VISIBLE_DEVICES")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    args = make_args()
    args.gpus = None
    runner.run_sequential(args, [job])
    assert seen["device"] == "3"
