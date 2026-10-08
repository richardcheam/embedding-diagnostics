"""Synthetic wiring tests; no model, source-image or research-claim validation."""

import copy

import numpy as np
import pytest

from embedding_diagnostics import source_sensitivity as m
from embedding_diagnostics.phase_c_pilot import support


def fixture():
    rows = [
        dict(
            image_id=f"id{i:02}",
            split="train" if i < 8 else "val",
            weather=i % 2,
            scene=i % 2,
            timeofday=i % 2,
        )
        for i in range(24)
    ]
    sample = {
        "selected_rows": rows,
        "support": support(rows),
        "eligible_probe_class_codes": {a: [0, 1] for a in m.ATTRIBUTES},
        "eligibility_rule": {"training_support": 20, "validation_support": 10},
    }
    matrix = np.random.default_rng(1).normal(size=(24, 4)).astype(np.float32)
    return matrix, sample


def test_exclusion_is_exact_preserves_training_and_order():
    x, sample = fixture()
    original = copy.deepcopy(sample)
    train, val, reduced, keep = m.exclude_rows(x, sample, ["id09", "id17"])
    np.testing.assert_array_equal(train, x[:8])
    np.testing.assert_array_equal(val, x[8:][keep])
    assert train.tobytes() == x[:8].tobytes()
    assert [r["image_id"] for r in reduced["selected_rows"]] == [
        r["image_id"] for r in sample["selected_rows"] if r["image_id"] not in {"id09", "id17"}
    ]
    assert reduced["eligible_probe_class_codes"] == sample["eligible_probe_class_codes"]
    assert sample == original
    assert len(val) == 14


@pytest.mark.parametrize("ids", [["unknown"], ["id00"], ["id09", "id09"]])
def test_missing_training_or_duplicate_exclusions_fail(ids):
    with pytest.raises(ValueError):
        m.exclude_rows(*fixture(), ids)


def test_row_alignment_rejects_duplicate_ids_or_matrix_size():
    x, sample = fixture()
    with pytest.raises(ValueError):
        m.exclude_rows(x[:-1], sample, ["id09"])
    sample["selected_rows"][-1]["image_id"] = "id00"
    with pytest.raises(ValueError):
        m.exclude_rows(x, sample, ["id09"])


def test_zero_support_is_unscorable_without_changing_eligibility():
    rows = m.class_scores(np.array([0, 0]), np.array([0, 1]), 3, (0, 1))
    assert rows["1"]["support"] == 0 and rows["1"]["recall"] is None
    assert rows["1"]["eligible"] and not rows["1"]["scorable"]
    assert rows["2"]["support"] == 0 and not rows["2"]["eligible"]
    assert rows["0"]["recall"] == 0.5


def test_rectangular_gallery_self_exclusion_and_denominators():
    x = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    rows = [
        dict(image_id=name, weather=label, scene=label, timeofday=label)
        for name, label in zip(["a", "b", "c", "d"], [0, 0, 1, 1], strict=True)
    ]
    result = m.gallery_scores(x, rows, np.array([0, 2]), k=1)
    for query, neighbours in zip(result["query_ids"], result["neighbour_ids"], strict=True):
        assert query not in neighbours
    assert result["gallery_count"] == 4 and result["query_count"] == 2
    assert result["attributes"]["weather"]["chance_self_excluded"] == pytest.approx(1 / 3)
    reduced = m.gallery_scores(x[:3], rows[:3], np.array([0, 2]), k=1)
    assert reduced["attributes"]["weather"]["chance_self_excluded"] == 0.25
    assert result["tie_rule"] == "inherited NumPy argpartition; no new tie ordering"


def test_rectangular_query_alignment_fails():
    with pytest.raises(ValueError):
        m.gallery_scores(np.eye(3), [{"image_id": "a"}], np.array([0]), k=1)


def test_historical_hash_protection_detects_changes(tmp_path):
    p = tmp_path / "historic.json"
    p.write_text("original")
    baseline = {str(p): m.digest(p)}
    m.verify_hashes(baseline)
    p.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        m.verify_hashes(baseline)


def test_refit_reuses_training_selection_and_existing_solver(monkeypatch):
    from embedding_diagnostics.diagnostics import probe
    from embedding_diagnostics.diagnostics.probe import ProbeConfig

    x = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    original = {
        "selected_C": 1.0,
        "inner_validation_accuracy": 0.75,
        "selection_fits": [{"selected_C": 1.0, "validation_accuracy": 0.75}],
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("train-only selection already exists; do not retune")

    monkeypatch.setattr(probe, "_select_c", forbidden)
    result = m.refit_probe(
        x, y, x[:2], y[:2], original, config=ProbeConfig(), eligible_classes=(0, 1)
    )
    assert result["selected_C"] == 1.0 and result["selection_fits"] == original["selection_fits"]
    assert result["selection_mode"] == "inner_validation"
    assert result["accuracy"] == 1.0 and result["converged"] == 1.0
    assert result["validation_per_class"]["1"]["support"] == 0
    assert result["validation_per_class"]["1"]["recall"] is None
    assert result["refit_provenance"]["training_selection_reused"]


def test_completed_resume_calls_no_endpoints_and_preserves_bytes(tmp_path):
    from embedding_diagnostics.phase_c_numerics import run_conditions

    path = tmp_path / "results.json"

    def evaluate(name):
        return {"condition": name, "value": 1}

    _, calls = run_conditions(path, {"frozen": "a"}, ("one", "two"), evaluate)
    assert calls == 2
    before = path.read_bytes()

    def forbidden(name):
        raise AssertionError("completed endpoint was recomputed")

    _, calls = run_conditions(path, {"frozen": "a"}, ("one", "two"), forbidden)
    assert calls == 0 and path.read_bytes() == before
    with pytest.raises(ValueError, match="provenance"):
        run_conditions(path, {"frozen": "b"}, ("one", "two"), forbidden)


def test_supplied_labels_and_fixed_eligibility_reach_refit(monkeypatch):
    from embedding_diagnostics.diagnostics import probe

    original = {"selected_C": 1.0, "inner_validation_accuracy": 0.5, "selection_fits": []}
    x = np.arange(24).reshape(12, 2).astype(float)
    y = np.arange(12) % 2
    result = m.refit_probe(
        x, y, x[:3], np.zeros(3, dtype=int), original, eligible_classes=(0, 1), class_count=3
    )
    assert result["fixed_eligible_class_codes"] == [0, 1]
    assert result["eligible_zero_support_classes"] == [1]
    assert result["scored_classes"] == 1
    assert result["validation_support"] == {"0": 3, "1": 0, "2": 0}
    assert result["selection_fits"] == []
    assert result["selection_mode"] == "fallback"
    assert probe._select_c.__name__ == "_select_c"  # no persistent monkeypatch


def test_source_context_reports_all_zero_support_categories():
    _, sample = fixture()
    sample["selected_rows"] = [r for r in sample["selected_rows"] if r["weather"] == 0]
    sample["support"] = support(sample["selected_rows"])
    record = {
        "attributes": {a: {"per_class": {"0": {"support": 8, "p10": 1}}} for a in m.ATTRIBUTES}
    }
    m.add_support_context(record, sample)
    assert record["attributes"]["weather"]["per_class"]["1"]["p10"] is None
    assert record["attributes"]["weather"]["class_support"]["foggy"]["val"] == 0
    assert record["attributes"]["weather"]["class_support"]["foggy"]["balanced_probe_eligible"]


def test_comparison_separates_absolute_change_from_effect_change():
    import runpy
    from pathlib import Path

    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/report_source_sensitivity.py"))
    result = module["comparison"](0.5, 0.7, 0.4, 0.6)
    assert result["absolute_change"] == pytest.approx(0.2)
    assert result["historical_effect"] == pytest.approx(0.1)
    assert result["reduced_effect"] == pytest.approx(0.1)
    assert result["effect_change"] == pytest.approx(0.0)


def test_runner_freeze_rejects_uncommitted_or_modified_file(monkeypatch, tmp_path):
    import runpy
    from pathlib import Path

    runner = runpy.run_path(str(Path(__file__).parents[1] / "scripts/run_source_sensitivity.py"))
    root = tmp_path / "sensitivity"
    root.mkdir()
    (root / "freeze.json").write_text("{}")
    runner["checked_freeze"].__globals__["ROOT"] = root
    runner["checked_freeze"].__globals__["committed"] = lambda path: b"committed bytes"
    with pytest.raises(ValueError, match="committed unchanged"):
        runner["checked_freeze"]()


def test_c3_numerical_comparison_handles_inapplicable_bounds():
    import runpy
    from pathlib import Path

    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/report_source_sensitivity.py"))
    old = {
        "condition": "c3_native_ref",
        "source_condition": "native_ref",
        "campaign": "C3",
        "summary": {
            "mean_overlap": 1.0,
            "identity_changed_queries": 0,
            "prospective_sufficient_queries": None,
        },
        "query_ids": ["a", "b"],
        "attributes": {
            a: {
                "p10": 0.5,
                "macro_p10_support10": 0.5,
                "chance_sum_p_squared": 0.5,
                "chance_self_excluded": 0.4,
            }
            for a in m.ATTRIBUTES
        },
    }
    new = copy.deepcopy(old)
    new["query_ids"] = ["a"]
    result = module["build_comparisons"]([new], {"c3_native_ref": old})
    summary = result["c3_native_ref"]["numerical_summary"]
    assert summary["prospective_sufficient_queries"]["absolute_change"] is None
    assert summary["mean_overlap"]["absolute_change"] == 0
    assert result["c3_native_ref"]["query_counts"] == {"historical": 2, "reduced": 1}


def test_cached_fit_rebinds_each_historical_comparison_without_mutating_it():
    cached = {
        "train_accuracy": 0.9,
        "refit_provenance": {
            "historical_train_accuracy": 0.7,
            "historical_iterations": 20,
            "historical_converged": False,
        },
    }
    prior = {"train_accuracy": 0.8, "n_iter": 30, "converged": True}
    result = m.rebind_fit_history(cached, prior, reused=True)
    assert result["refit_provenance"]["historical_train_accuracy"] == 0.8
    assert result["refit_provenance"]["train_accuracy_difference"] == pytest.approx(0.1)
    assert result["refit_provenance"]["historical_iterations"] == 30
    assert result["refit_provenance"]["reused_identical_fit_score"]
    assert cached["refit_provenance"]["historical_train_accuracy"] == 0.7
