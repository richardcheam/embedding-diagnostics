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


def test_figures_tolerate_logs_missing_a_newer_metric(tmp_path):
    """A metric added later must not make earlier run logs unplottable.

    Run logs are kept in git for the life of the project, so the figure builder
    has to cope with records written before a diagnostic existed.
    """
    from jepa_lens.figures import build_all_figures

    old = [{"step": s, "probe_accuracy": 0.3, "effective_rank": 5.0} for s in (0, 100, 200, 300)]
    written = build_all_figures({"ema_stopgrad": old}, tmp_path)

    assert written, "figure builder returned nothing"
    assert all(path.exists() for path in written)


def test_sustained_decline_finds_the_turn_after_a_rise():
    """The shape first_departure_step cannot handle: learn, plateau, degrade."""
    from jepa_lens.runs import first_sustained_decline

    steps = list(range(0, 1100, 100))
    values = [0.3, 0.5, 0.6, 0.65, 0.66, 0.66, 0.60, 0.55, 0.50, 0.45, 0.40]
    turn = first_sustained_decline(steps, values, higher_is_better=True, margin=0.02)
    assert turn == 600


def test_sustained_decline_handles_lower_is_better():
    """Cosine similarity degrades by RISING, so direction must be respected."""
    from jepa_lens.runs import first_sustained_decline

    steps = list(range(0, 900, 100))
    values = [0.9, 0.6, 0.4, 0.3, 0.29, 0.45, 0.60, 0.75]
    turn = first_sustained_decline(steps, values, higher_is_better=False, margin=0.02)
    assert turn == 500


def test_a_single_noisy_point_does_not_trigger_a_detection():
    """Persistence guards against one bad checkpoint being read as degradation."""
    from jepa_lens.runs import first_sustained_decline

    steps = list(range(0, 900, 100))
    values = [0.3, 0.5, 0.6, 0.65, 0.40, 0.66, 0.67, 0.68]  # one dip at step 400
    assert first_sustained_decline(steps, values, higher_is_better=True, margin=0.02) is None


def test_monotone_improvement_never_declines():
    from jepa_lens.runs import first_sustained_decline

    steps = list(range(0, 800, 100))
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert first_sustained_decline(steps, values, higher_is_better=True, margin=0.01) is None


def test_flat_series_never_declines():
    from jepa_lens.runs import first_sustained_decline

    steps = list(range(0, 800, 100))
    assert first_sustained_decline(steps, [0.5] * 7, higher_is_better=True) is None


def test_returns_the_start_of_the_streak_not_its_end():
    """The step of interest is when degradation began."""
    from jepa_lens.runs import first_sustained_decline

    steps = [0, 100, 200, 300, 400, 500]
    values = [0.5, 0.6, 0.61, 0.50, 0.45, 0.40]
    turn = first_sustained_decline(
        steps, values, higher_is_better=True, margin=0.02, persistence=3
    )
    assert turn == 300


def test_monotone_fraction_flags_a_series_with_no_turn():
    """Total variance falls throughout a healthy run; it has no turning point."""
    from jepa_lens.runs import monotone_fraction

    assert monotone_fraction([10.0, 8.0, 6.0, 4.0, 2.0]) == 1.0
    assert monotone_fraction([1.0, 2.0, 3.0, 4.0]) == 1.0
    assert monotone_fraction([1.0, 2.0, 3.0, 2.0, 1.0]) < 0.75


def test_turning_point_finds_a_peak_and_a_trough():
    from jepa_lens.runs import turning_point

    steps = [0, 100, 200, 300, 400]
    assert turning_point(steps, [0.3, 0.5, 0.6, 0.55, 0.5], mode="max") == 200
    assert turning_point(steps, [0.9, 0.6, 0.3, 0.5, 0.8], mode="min") == 200


def test_turning_point_returns_none_when_the_series_never_turns():
    """An extremum at either end means the run ended before reversing."""
    from jepa_lens.runs import turning_point

    steps = [0, 100, 200, 300]
    assert turning_point(steps, [0.1, 0.2, 0.3, 0.4], mode="max") is None
    assert turning_point(steps, [0.4, 0.3, 0.2, 0.1], mode="max") is None


def _script(name):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    return __import__(name)


def _write_run(root, tag, condition, records):
    import json

    run_dir = root / tag / condition
    run_dir.mkdir(parents=True)
    with (run_dir / "metrics.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_summarize_run_uses_the_final_checkpoint_not_the_max():
    """The endpoint is pre-registered: last checkpoint, never best-of-run."""
    summarize = _script("summarize_sweep")
    records = [
        {"step": s, "probe_accuracy_unscaled": p, "mean_pairwise_cosine": 0.3,
         "total_variance": 80.0, "participation_ratio": 5.0, "rankme": 4.0}
        for s, p in [(0, 0.37), (100, 0.60), (200, 0.55)]
    ]
    row = summarize.summarize_run(records)
    assert row["probe_final"] == 0.55
    assert row["probe_best_biased"] == 0.60
    assert abs(row["probe_delta"] - 0.18) < 1e-12


