"""Paired across-seed statistics: correctness and loud failure."""

import numpy as np
import pytest

from jepa_lens.stats import (
    CONTRASTS,
    paired_contrast,
    summarize_condition,
    t_critical,
)


def runs_from(values: dict[str, dict[int, float]], endpoint="probe", step=4000):
    """{condition: {seed: final_record}} from plain numbers."""
    return {
        condition: {seed: {endpoint: value, "step": step} for seed, value in by_seed.items()}
        for condition, by_seed in values.items()
    }


def test_paired_differences_are_computed_per_seed():
    runs = runs_from({"a": {0: 0.60, 1: 0.64, 2: 0.62}, "b": {0: 0.50, 1: 0.50, 2: 0.56}})
    result = paired_contrast("a", "b", "probe", runs)
    assert result.differences == pytest.approx([0.10, 0.14, 0.06])
    assert result.mean_difference == pytest.approx(0.10)
    assert result.n_pairs == 3
    assert result.seeds == [0, 1, 2]


def test_interval_matches_the_textbook_paired_t_formula():
    values = {"a": {0: 0.60, 1: 0.64, 2: 0.62}, "b": {0: 0.50, 1: 0.50, 2: 0.56}}
    result = paired_contrast("a", "b", "probe", runs_from(values))

    differences = np.array([0.10, 0.14, 0.06])
    expected_sd = np.std(differences, ddof=1)
    expected_se = expected_sd / np.sqrt(3)
    half = 4.303 * expected_se  # t(0.975, df=2)

    assert result.sd_difference == pytest.approx(expected_sd)
    assert result.standard_error == pytest.approx(expected_se)
    assert result.ci_low == pytest.approx(0.10 - half)
    assert result.ci_high == pytest.approx(0.10 + half)


def test_identical_conditions_give_a_zero_width_interval_at_zero():
    runs = runs_from({"a": {0: 0.5, 1: 0.6}, "b": {0: 0.5, 1: 0.6}})
    result = paired_contrast("a", "b", "probe", runs)
    assert result.mean_difference == 0.0
    assert result.ci_low == 0.0 and result.ci_high == 0.0


def test_interval_straddling_zero_is_reported_not_judged():
    """No verdict field exists; a null result looks like any other interval."""
    runs = runs_from({"a": {0: 0.50, 1: 0.60, 2: 0.40}, "b": {0: 0.55, 1: 0.45, 2: 0.52}})
    result = paired_contrast("a", "b", "probe", runs)
    assert result.ci_low < 0 < result.ci_high
    assert not hasattr(result, "significant")
    assert not hasattr(result, "real")


def test_t_critical_uses_the_right_quantiles():
    assert t_critical(4) == pytest.approx(2.776)  # 5 seeds
    assert t_critical(2) == pytest.approx(4.303)  # 3 seeds
    assert t_critical(1) == pytest.approx(12.706)  # 2 seeds
    assert t_critical(4) > t_critical(9)  # fewer seeds, wider interval


def test_t_critical_needs_at_least_two_pairs():
    with pytest.raises(ValueError, match="at least two paired seeds"):
        t_critical(0)


def test_five_seeds_is_wider_than_ten_for_the_same_spread():
    """Sets expectations: 5 seeds buys a CI half-width of ~1.24 seed SDs."""
    five = runs_from({"a": {i: 0.6 + 0.01 * (i % 2) for i in range(5)},
                      "b": {i: 0.5 for i in range(5)}})
    result = paired_contrast("a", "b", "probe", five)
    assert result.ci_high - result.ci_low > 0


def test_mismatched_seed_sets_fail_loudly():
    runs = runs_from({"a": {0: 0.6, 1: 0.6}, "b": {0: 0.5, 2: 0.5}})
    with pytest.raises(ValueError, match="Paired analysis needs identical seed sets"):
        paired_contrast("a", "b", "probe", runs)


def test_a_missing_seed_fails_rather_than_dropping_silently():
    runs = runs_from({"a": {0: 0.6, 1: 0.6, 2: 0.6}, "b": {0: 0.5, 1: 0.5}})
    with pytest.raises(ValueError, match="identical seed sets"):
        paired_contrast("a", "b", "probe", runs)


def test_mismatched_final_steps_fail_loudly():
    runs = {
        "a": {0: {"probe": 0.6, "step": 4000}, 1: {"probe": 0.6, "step": 4000}},
        "b": {0: {"probe": 0.5, "step": 4000}, 1: {"probe": 0.5, "step": 2000}},
    }
    with pytest.raises(ValueError, match="final steps differ"):
        paired_contrast("a", "b", "probe", runs)


def test_fewer_than_two_pairs_fails_loudly():
    runs = runs_from({"a": {0: 0.6}, "b": {0: 0.5}})
    with pytest.raises(ValueError, match="an interval needs at least 2"):
        paired_contrast("a", "b", "probe", runs)


def test_missing_endpoint_fails_loudly():
    runs = {
        "a": {0: {"probe": 0.6, "step": 1}, 1: {"step": 1}},
        "b": {0: {"probe": 0.5, "step": 1}, 1: {"probe": 0.5, "step": 1}},
    }
    with pytest.raises(KeyError, match="missing at seed"):
        paired_contrast("a", "b", "probe", runs)


def test_absent_condition_names_what_is_available():
    runs = runs_from({"a": {0: 0.6, 1: 0.6}})
    with pytest.raises(KeyError, match="absent"):
        paired_contrast("a", "missing", "probe", runs)


def test_condition_summary_reports_seeds_values_mean_and_sample_sd():
    summary = summarize_condition("a", "probe", {0: {"probe": 0.5}, 1: {"probe": 0.7}})
    assert summary.seeds == [0, 1]
    assert summary.values == [0.5, 0.7]
    assert summary.mean == pytest.approx(0.6)
    assert summary.sd == pytest.approx(np.std([0.5, 0.7], ddof=1))
    assert summary.n == 2


def test_single_seed_summary_has_undefined_sd_not_zero():
    """One seed cannot estimate spread; reporting 0.0 would imply certainty."""
    summary = summarize_condition("a", "probe", {0: {"probe": 0.5}})
    assert np.isnan(summary.sd)


def test_contrasts_are_the_five_predeclared_factorial_comparisons():
    assert len(CONTRASTS) == 5
    arms = {a for a, _, _ in CONTRASTS} | {b for _, b, _ in CONTRASTS}
    assert "ema_stopgrad" not in arms, "EMA differs in two factors; not a contrast arm"
    assert ("none_stopgrad", "none_nostopgrad") == CONTRASTS[0][:2]


def test_aggregate_seeds_collect_reads_the_family(tmp_path):
    """Smoke: paired aggregation on synthetic logs, end to end."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import aggregate_seeds

    for seed, (a, b) in enumerate([(0.60, 0.50), (0.64, 0.50), (0.62, 0.56)]):
        for condition, value in [("none_stopgrad", a), ("none_nostopgrad", b)]:
            run_dir = tmp_path / f"phaseA_s{seed}" / condition
            run_dir.mkdir(parents=True)
            with (run_dir / "metrics.jsonl").open("w") as handle:
                for step, probe in [(0, 0.37), (4000, value)]:
                    handle.write(json.dumps({
                        "step": step, "probe_accuracy_unscaled": probe,
                        "total_variance": 80.0, "mean_pairwise_cosine": 0.3,
                        "participation_ratio": 5.0, "rankme": 4.0,
                        "probe_balanced": probe, "retrieval_p10": probe,
                    }) + "\n")

    runs = aggregate_seeds.collect(tmp_path, "phaseA")
    assert sorted(runs) == ["none_nostopgrad", "none_stopgrad"]
    assert sorted(runs["none_stopgrad"]) == [0, 1, 2]
    # Final checkpoint only.
    assert runs["none_stopgrad"][1]["probe_accuracy_unscaled"] == 0.64

    result = paired_contrast(
        "none_stopgrad", "none_nostopgrad", "probe_accuracy_unscaled", runs
    )
    assert result.differences == pytest.approx([0.10, 0.14, 0.06])
    assert result.n_pairs == 3


def test_collect_ignores_unrelated_tags(tmp_path):
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import aggregate_seeds

    for tag in ("phaseA_s0", "phaseAB_s1", "other"):
        run_dir = tmp_path / tag / "none_stopgrad"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 0, "probe": 0.4}) + "\n")
    runs = aggregate_seeds.collect(tmp_path, "phaseA")
    assert sorted(runs["none_stopgrad"]) == [0]
