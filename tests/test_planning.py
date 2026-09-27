"""Preflight safety: nothing may be written until the whole plan is safe."""

import json
from pathlib import Path

import pytest

from embedding_diagnostics.planning import (
    Job,
    build_jobs,
    completed_step,
    parse_seeds,
    partition_jobs,
    preflight,
)

CONDITIONS = ["ema_stopgrad", "none_stopgrad"]


def test_parse_seeds_accepts_a_plain_list():
    assert parse_seeds("0,1,2,3,4") == [0, 1, 2, 3, 4]


def test_duplicate_seeds_are_rejected():
    with pytest.raises(ValueError, match="duplicate seeds"):
        parse_seeds("0,1,1")


def test_empty_seed_list_is_rejected():
    with pytest.raises(ValueError, match="no seeds"):
        parse_seeds("")
    with pytest.raises(ValueError, match="no seeds"):
        parse_seeds(" , ")


def test_negative_seeds_are_rejected_because_naming_cannot_carry_them():
    """<tag>_s<seed> with aggregation matching _s(\\d+)$ would silently miss them."""
    with pytest.raises(ValueError, match="negative seeds"):
        parse_seeds("0,-1")


def test_jobs_are_always_seed_explicit_even_for_one_seed():
    """A path must reveal its seed regardless of how the run was invoked."""
    jobs = build_jobs(CONDITIONS, [7], "phaseA")
    assert {job.tag for job in jobs} == {"phaseA_s7"}


def test_every_condition_seed_pair_is_planned_once():
    jobs = build_jobs(CONDITIONS, [0, 1, 2], "phaseA")
    assert len(jobs) == 6
    assert len({job.label for job in jobs}) == 6


def test_duplicate_conditions_are_rejected():
    with pytest.raises(ValueError, match="duplicate job labels"):
        build_jobs(["a", "a"], [0], "t")


def test_empty_condition_list_is_rejected():
    with pytest.raises(ValueError, match="no conditions"):
        build_jobs([], [0], "t")


def test_preflight_passes_on_a_clean_tree(tmp_path):
    assert preflight(build_jobs(CONDITIONS, [0], "phaseA"), tmp_path) == []


def test_preflight_blocks_an_occupied_directory(tmp_path):
    job = Job("ema_stopgrad", 0, "phaseA_s0")
    run_dir = job.run_dir(tmp_path)
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 0}) + "\n")

    problems = preflight([job], tmp_path)
    assert len(problems) == 1
    assert "already holds results" in problems[0]


def test_preflight_allows_an_empty_leftover_directory(tmp_path):
    job = Job("ema_stopgrad", 0, "phaseA_s0")
    job.run_dir(tmp_path).mkdir(parents=True)
    assert preflight([job], tmp_path) == []


def test_overwrite_opt_in_clears_the_block(tmp_path):
    job = Job("ema_stopgrad", 0, "phaseA_s0")
    run_dir = job.run_dir(tmp_path)
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text("{}\n")
    assert preflight([job], tmp_path, overwrite=True) == []


def test_preflight_reports_every_collision_not_just_the_first(tmp_path):
    """A collision in the last wave must surface before the first wave runs."""
    jobs = build_jobs(CONDITIONS, [0, 1], "phaseA")
    for job in jobs:
        run_dir = job.run_dir(tmp_path)
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.jsonl").write_text("{}\n")
    assert len(preflight(jobs, tmp_path)) == len(jobs)


def test_a_rejected_rerun_leaves_previous_artifacts_untouched(tmp_path, monkeypatch):
    """The regression this module exists for: train.log used to be truncated
    before the child's RunLogger refused to append."""
    import argparse
    import subprocess
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_all_conditions as runner

    run_dir = tmp_path / "experiments" / "phaseA_s0" / "ema_stopgrad"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text('{"step": 0, "probe_accuracy": 0.5}\n')
    (run_dir / "config.json").write_text('{"seed": 0}')
    (run_dir / "train.log").write_text("original log contents\n")
    before = {path.name: path.read_text() for path in run_dir.iterdir() if path.is_file()}

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "CONDITIONS", ["ema_stopgrad"])
    # Any spawn at all would mean the preflight contract was violated.
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: pytest.fail("a process was started")
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("a process was started")
    )

    args = argparse.Namespace(
        device="cpu", total_steps=None, checkpoint_every=None, tag="phaseA",
        resume=False,
        parallel=False, seeds="0", gpus=None, overwrite=False, base_config="base.yaml",
    )
    with pytest.raises(SystemExit) as excinfo:
        runner.plan(args)
    assert excinfo.value.code == 2

    after = {path.name: path.read_text() for path in run_dir.iterdir() if path.is_file()}
    assert after == before, "a rejected rerun modified existing artifacts"


# --- resume after a partial failure ------------------------------------


def write_log(run_dir: Path, steps: list[int]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps({"step": s, "probe_accuracy": 0.5}) for s in steps) + "\n"
    )


def test_completed_step_reads_the_highest_step(tmp_path):
    write_log(tmp_path, [0, 200, 400])
    assert completed_step(tmp_path) == 400


def test_completed_step_is_none_when_nothing_was_logged(tmp_path):
    assert completed_step(tmp_path) is None
    (tmp_path / "metrics.jsonl").write_text("")
    assert completed_step(tmp_path) is None


def test_completed_step_survives_a_truncated_final_line(tmp_path):
    """A process killed mid-write leaves half a JSON object. Refusing to parse
    the file would misreport a nearly-complete run as never started."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "metrics.jsonl").write_text(
        '{"step": 0}\n{"step": 200}\n{"step": 400, "probe_ac'
    )
    assert completed_step(tmp_path) == 200


def test_completed_step_ignores_step_order_in_the_file(tmp_path):
    write_log(tmp_path, [400, 0, 200])
    assert completed_step(tmp_path) == 400


def test_partition_splits_finished_from_crashed(tmp_path):
    jobs = build_jobs(["a", "b", "c"], [0], "t")
    write_log(tmp_path / "t_s0" / "a", [0, 2000, 4000])  # finished
    write_log(tmp_path / "t_s0" / "b", [0])              # crashed at the first checkpoint
    # c never wrote anything at all

    done, todo = partition_jobs(jobs, tmp_path, total_steps=4000)
    assert [j.condition for j in done] == ["a"]
    assert [j.condition for j in todo] == ["b", "c"]


def test_a_run_past_the_target_counts_as_finished(tmp_path):
    """A log from a longer budget must not be rerun just because the number
    does not match exactly."""
    jobs = build_jobs(["a"], [0], "t")
    write_log(tmp_path / "t_s0" / "a", [0, 4000, 8000])
    done, todo = partition_jobs(jobs, tmp_path, total_steps=4000)
    assert len(done) == 1 and not todo


def test_partition_does_not_touch_the_filesystem(tmp_path):
    """Planning must stay read-only; deletion happens in the child, per job."""
    jobs = build_jobs(["a"], [0], "t")
    run_dir = tmp_path / "t_s0" / "a"
    write_log(run_dir, [0])
    before = (run_dir / "metrics.jsonl").read_text()
    partition_jobs(jobs, tmp_path, total_steps=4000)
    assert (run_dir / "metrics.jsonl").read_text() == before
