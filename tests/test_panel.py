"""The panel's claims, pinned as tests.

The module asserts three things that would quietly become false if the
diagnostics changed: that no single diagnostic covers every collapse mode, that
the participation ratio earns no place in a minimal panel, and that the
absolute thresholds separate this project's collapsed control from every
condition that genuinely trains. Each is checked here against the same
controlled degradations and the same recorded readings the claims came from.
"""

import numpy as np
import pytest

from jepa_lens.panel import (
    CANDIDATES,
    RECOMMENDED_PANEL,
    PanelThresholds,
    minimal_panels,
    mode_coverage,
    verdict,
)

# Final-checkpoint means actually recorded by this project, so the thresholds
# are tested against the numbers they were calibrated against rather than
# against invented ones.
CIFAR_DEAD = {
    "rankme": 1.12,
    "mean_pairwise_cosine": 1.0000,
    "mean_feature_std": 0.00106,
    "total_variance": 0.0002,
    "probe_accuracy_unscaled": 0.107,
}
CIFAR_ALIVE = {
    "ema_stopgrad": dict(rankme=85.72, mean_pairwise_cosine=0.3857,
                         mean_feature_std=0.66797, total_variance=87.7488),
    "none_stopgrad": dict(rankme=9.67, mean_pairwise_cosine=0.8398,
                          mean_feature_std=0.29140, total_variance=21.5723),
    "sigreg_nostopgrad": dict(rankme=39.60, mean_pairwise_cosine=0.8497,
                              mean_feature_std=0.05697, total_variance=0.6933),
    "proj_sigreg_nostopgrad": dict(rankme=11.28, mean_pairwise_cosine=0.6943,
                                   mean_feature_std=0.41139, total_variance=34.6536),
}
BDD_DEAD = {
    "rankme": 1.06,
    "mean_pairwise_cosine": 1.0000,
    "mean_feature_std": 0.00058,
    "total_variance": 0.0001,
}
BDD_ALIVE = {
    "ema_stopgrad": dict(rankme=78.93, mean_pairwise_cosine=0.2562,
                         mean_feature_std=0.73458, total_variance=108.6064),
    "none_stopgrad": dict(rankme=9.72, mean_pairwise_cosine=0.5490,
                          mean_feature_std=0.43740, total_variance=49.1232),
    "sigreg_nostopgrad": dict(rankme=23.73, mean_pairwise_cosine=0.5723,
                              mean_feature_std=0.09743, total_variance=4.1716),
}


@pytest.fixture
def clustered():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(8), 64)
    centroids = rng.normal(size=(8, 32)) * 6.0
    return centroids[labels] + rng.normal(scale=1.0, size=(512, 32))


# --- coverage and minimality -------------------------------------------


def test_no_single_diagnostic_covers_every_mode(clustered):
    """The project's central negative result, stated as a search over subsets."""
    panels = minimal_panels(mode_coverage(clustered))
    assert panels, "no subset covered every mode"
    assert len(panels[0]) >= 2


def test_two_diagnostics_suffice(clustered):
    panels = minimal_panels(mode_coverage(clustered))
    assert len(panels[0]) == 2


def test_the_recommended_panel_is_one_of_the_minimal_ones(clustered):
    panels = {tuple(sorted(p)) for p in minimal_panels(mode_coverage(clustered))}
    assert tuple(sorted(RECOMMENDED_PANEL)) in panels


def test_participation_ratio_earns_no_place_in_any_minimal_panel(clustered):
    """It is offered to the search and rejected on the evidence. On both real
    datasets it also reads higher on the dead encoder than the healthy one."""
    panels = minimal_panels(mode_coverage(clustered))
    assert all("participation_ratio" not in panel for panel in panels)


def test_scale_contraction_is_invisible_to_every_scale_invariant_metric(clustered):
    """The invariance that makes a one-metric panel impossible."""
    coverage = mode_coverage(clustered)
    for metric in ("mean_pairwise_cosine", "rankme", "participation_ratio"):
        assert metric not in coverage["scale_contraction"]


def test_mean_injection_is_invisible_to_every_centered_metric(clustered):
    """The complementary invariance: shifting the whole cloud leaves the
    centered covariance untouched, so total variance cannot see it. This is why
    the minimal panel needs one metric from each family."""
    coverage = mode_coverage(clustered)
    assert "total_variance" not in coverage["mean_injection"]
    assert "participation_ratio" not in coverage["mean_injection"]
    assert coverage["mean_injection"], "something must detect it"


def test_every_mode_is_detected_by_something(clustered):
    coverage = mode_coverage(clustered)
    for mode, responded in coverage.items():
        assert responded, f"{mode} went entirely undetected"


def test_minimal_panels_is_empty_when_a_mode_is_undetectable():
    assert minimal_panels({"a": {"x"}, "b": set()}) == []


def test_coverage_only_reports_requested_candidates(clustered):
    coverage = mode_coverage(clustered, candidates=("rankme",))
    assert all(responded <= {"rankme"} for responded in coverage.values())


# --- absolute thresholds on real readings ------------------------------


@pytest.mark.parametrize("readings", [CIFAR_DEAD, BDD_DEAD], ids=["cifar10", "bdd100k"])
def test_the_collapsed_control_is_flagged_on_both_datasets(readings):
    result = verdict(readings)
    assert result["degenerate"]
    # Not a lucky single trip: every check fires on total collapse.
    assert set(result["tripped"]) == {
        "rankme", "mean_pairwise_cosine", "mean_feature_std", "total_variance"
    }


@pytest.mark.parametrize("readings", list(CIFAR_ALIVE.values()), ids=list(CIFAR_ALIVE))
def test_no_false_alarm_on_cifar_conditions_that_train(readings):
    assert not verdict(readings)["degenerate"]


@pytest.mark.parametrize("readings", list(BDD_ALIVE.values()), ids=list(BDD_ALIVE))
def test_no_false_alarm_on_bdd_conditions_that_train(readings):
    assert not verdict(readings)["degenerate"]


def test_a_merely_untrained_encoder_is_not_called_degenerate():
    """The panel detects degeneracy, not failure to learn. `sigreg_nostopgrad`
    never beat its own initialisation but is not collapsed, and calling it
    degenerate would be a false positive with a real cost."""
    assert not verdict(CIFAR_ALIVE["sigreg_nostopgrad"])["degenerate"]


def test_missing_metrics_are_reported_rather_than_passing_silently():
    """A partial log must not produce a clean bill of health."""
    result = verdict({"rankme": 50.0})
    assert not result["degenerate"]
    assert set(result["missing"]) == {
        "mean_pairwise_cosine", "mean_feature_std", "total_variance"
    }


def test_an_empty_record_reports_everything_missing():
    result = verdict({})
    assert result["missing"] and not result["tripped"]


def test_the_unstandardized_probe_is_surfaced_as_the_semantic_reading():
    """The standardized probe rates a dead encoder second of seven, so the
    unstandardized one is the only probe the panel will quote."""
    assert verdict(CIFAR_DEAD)["semantic"] == 0.107
    assert verdict(BDD_DEAD)["semantic"] is None


def test_the_verdict_refuses_to_claim_semantic_loss():
    """Three of five controlled degradations leave retrieval at 1.000, so a
    geometric alarm cannot be reported as 'the representation is broken'."""
    note = verdict(CIFAR_DEAD)["note"].lower()
    assert "does not" in note and "retrieval" in note


def test_thresholds_are_overridable():
    strict = PanelThresholds(rankme_floor=100.0)
    assert verdict(CIFAR_ALIVE["ema_stopgrad"], strict)["degenerate"]


def test_candidates_and_recommendation_stay_consistent():
    assert set(RECOMMENDED_PANEL) <= set(CANDIDATES)
