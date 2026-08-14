"""Preflight safety: nothing may be written until the whole plan is safe."""

import json

import pytest

from jepa_lens.planning import Job, build_jobs, parse_seeds, preflight

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
        parallel=False, seeds="0", gpus=None, overwrite=False, base_config="base.yaml",
    )
    with pytest.raises(SystemExit) as excinfo:
        runner.plan(args)
    assert excinfo.value.code == 2

    after = {path.name: path.read_text() for path in run_dir.iterdir() if path.is_file()}
    assert after == before, "a rejected rerun modified existing artifacts"
