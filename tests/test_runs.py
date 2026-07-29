import json
from pathlib import Path

from jepa_lens.runs import first_departure_step, load_runs


def write_run(root: Path, tag: str, condition: str, records: list[dict]) -> None:
    run_dir = root / tag / condition
    run_dir.mkdir(parents=True)
    with (run_dir / "metrics.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_load_runs_keys_by_condition(tmp_path):
    write_run(tmp_path, "main", "ema_stopgrad", [{"step": 0, "probe_accuracy": 0.1}])
    write_run(tmp_path, "main", "sigreg_stopgrad", [{"step": 0, "probe_accuracy": 0.2}])

    runs = load_runs(tmp_path, "main")
    assert set(runs) == {"ema_stopgrad", "sigreg_stopgrad"}
    assert runs["sigreg_stopgrad"][0]["probe_accuracy"] == 0.2


def test_load_runs_sorts_records_by_step(tmp_path):
    write_run(tmp_path, "main", "ema_stopgrad", [{"step": 500}, {"step": 0}, {"step": 1000}])
    runs = load_runs(tmp_path, "main")
    assert [r["step"] for r in runs["ema_stopgrad"]] == [0, 500, 1000]


def test_first_departure_detects_a_drop():
    """The departure step is what Part 2 compares across metrics."""
    steps = [0, 100, 200, 300, 400, 500]
    values = [1.0, 1.0, 1.0, 1.0, 0.2, 0.1]
    assert first_departure_step(steps, values) == 400


def test_first_departure_returns_none_for_flat_series():
    steps = [0, 100, 200, 300]
    values = [1.0, 1.01, 0.99, 1.0]
    assert first_departure_step(steps, values) is None


def test_first_departure_handles_short_series():
    assert first_departure_step([0], [1.0]) is None
