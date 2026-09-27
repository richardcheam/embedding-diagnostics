"""Mathematical invariants of the controlled degradations.

These are the properties that make the stress test interpretable: if a
transformation does not do what its name says, every conclusion drawn from it
is void.
"""

import numpy as np
import pytest

from embedding_diagnostics.degradation import (
    DEFAULT_SEVERITIES,
    TRANSFORMS,
    diagnose_transform,
    isotropic_noise,
    mean_injection,
    mean_interpolation,
    rank_truncation,
    scale_contraction,
    sweep,
)
from embedding_diagnostics.diagnostics.metrics import collapse_metrics, participation_ratio, rankme


@pytest.fixture
def embeddings():
    rng = np.random.default_rng(0)
    return rng.normal(size=(256, 16))


def test_severity_zero_is_the_identity_for_every_transform(embeddings):
    for name, transform in TRANSFORMS.items():
        result = transform(embeddings, 0.0, 0)
        assert np.allclose(result, embeddings, atol=1e-9), name


def test_every_transform_is_deterministic_for_a_fixed_seed(embeddings):
    for name, transform in TRANSFORMS.items():
        a = transform(embeddings, 0.5, 7)
        b = transform(embeddings, 0.5, 7)
        assert np.array_equal(a, b), name


def test_scale_contraction_scales_total_variance_quadratically(embeddings):
    """Variance is quadratic in the scale factor."""
    before = collapse_metrics(embeddings)["total_variance"]
    after = collapse_metrics(scale_contraction(embeddings, 0.5))["total_variance"]
    assert after == pytest.approx(before * 0.25, rel=1e-9)


def test_positive_rescaling_leaves_the_scale_invariant_metrics_alone(embeddings):
    """Cosine, participation ratio and RankMe are all scale-invariant — which
    is exactly why none of them can detect scale collapse alone."""
    before = collapse_metrics(embeddings)
    after = collapse_metrics(scale_contraction(embeddings, 0.9))  # 10x smaller
    assert after["mean_pairwise_cosine"] == pytest.approx(
        before["mean_pairwise_cosine"], abs=1e-9
    )
    assert after["participation_ratio"] == pytest.approx(before["participation_ratio"], rel=1e-9)
    assert after["rankme"] == pytest.approx(before["rankme"], rel=1e-6)
    assert after["total_variance"] < before["total_variance"] / 50


def test_rank_truncation_cannot_exceed_the_requested_rank(embeddings):
    for severity in (0.25, 0.5, 0.75):
        truncated = rank_truncation(embeddings, severity)
        requested = max(1, int(round(min(embeddings.shape) * (1.0 - severity))))
        assert np.linalg.matrix_rank(truncated, tol=1e-8) <= requested


def test_rank_truncation_lowers_the_participation_ratio(embeddings):
    full = collapse_metrics(embeddings)["participation_ratio"]
    truncated = collapse_metrics(rank_truncation(embeddings, 0.75))["participation_ratio"]
    assert truncated < full


def test_mean_injection_leaves_the_centered_covariance_untouched(embeddings):
    """Adding a constant shifts every sample identically, so the centered
    covariance — and the participation ratio built from it — cannot move."""
    before = collapse_metrics(embeddings)
    after = collapse_metrics(mean_injection(embeddings, 0.9))
    assert after["participation_ratio"] == pytest.approx(before["participation_ratio"], rel=1e-6)
    assert after["total_variance"] == pytest.approx(before["total_variance"], rel=1e-6)


def test_mean_injection_concentrates_angles_and_collapses_rankme(embeddings):
    """The uncentered view sees it clearly: cosine rises toward 1 and RankMe,
    which works on raw singular values, falls as the mean dominates."""
    before = collapse_metrics(embeddings)
    after = collapse_metrics(mean_injection(embeddings, 0.95))
    assert after["mean_pairwise_cosine"] > 0.9
    assert after["mean_pairwise_cosine"] > before["mean_pairwise_cosine"]
    assert after["rankme"] < before["rankme"] / 2


def test_mean_injection_is_where_participation_ratio_and_rankme_disagree(embeddings):
    """The clearest demonstration that the two rank measures are not
    interchangeable: one is blind to this transformation, the other is not."""
    shifted = mean_injection(embeddings, 0.95)
    centered = shifted - shifted.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / (len(shifted) - 1)
    assert participation_ratio(np.linalg.eigvalsh(covariance)) > 12.0
    assert rankme(shifted) < 3.0


def test_mean_interpolation_ends_at_a_single_non_zero_point(embeddings):
    """Severity 1 is complete collapse to the mean — not to the origin."""
    collapsed = mean_interpolation(embeddings, 1.0)
    assert np.allclose(collapsed, collapsed[0])
    assert np.linalg.norm(collapsed[0]) > 0  # distinct from scale contraction


def test_mean_interpolation_drives_variance_down_and_cosine_up(embeddings):
    before = collapse_metrics(embeddings)
    after = collapse_metrics(mean_interpolation(embeddings, 0.99))
    assert after["total_variance"] < before["total_variance"] / 1000
    assert after["mean_pairwise_cosine"] > before["mean_pairwise_cosine"]


def test_isotropic_noise_raises_the_participation_ratio(embeddings):
    """The regime that makes rank misleading: noise flattens the spectrum, so
    a rank measure RISES while the representation carries less signal."""
    anisotropic = embeddings @ np.diag([8.0] + [0.05] * 15)
    before = collapse_metrics(anisotropic)["participation_ratio"]
    after = collapse_metrics(isotropic_noise(anisotropic, 0.9))["participation_ratio"]
    assert after > before


def test_diagnose_transform_without_labels_omits_semantic_endpoints(embeddings):
    record = diagnose_transform(embeddings, "scale_contraction", 0.5)
    assert "total_variance" in record
    assert "probe_accuracy" not in record
    assert "retrieval_p10" not in record


def test_diagnose_transform_with_labels_adds_semantic_endpoints(embeddings):
    labels = np.repeat(np.arange(4), 64)
    record = diagnose_transform(embeddings, "mean_interpolation", 0.5, labels=labels)
    for key in ("probe_accuracy", "probe_balanced", "retrieval_p10", "retrieval_chance"):
        assert key in record


def test_semantic_endpoints_fall_under_complete_collapse():
    """When every sample becomes the same vector, the label information carried
    by direction is gone, so retrieval falls to its own chance level."""
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(4), 64)
    # Well-separated clusters in DISTINCT directions, so cosine retrieval can
    # actually resolve them before the transformation is applied.
    centroids = np.eye(4, 16) * 10.0
    clustered = centroids[labels] + rng.normal(scale=0.1, size=(256, 16))

    healthy = diagnose_transform(clustered, "scale_contraction", 0.0, labels=labels)
    collapsed = diagnose_transform(clustered, "mean_interpolation", 1.0, labels=labels)

    assert healthy["retrieval_p10"] > 0.9
    assert collapsed["retrieval_p10"] == pytest.approx(
        collapsed["retrieval_chance"], abs=0.05
    )


def test_scale_contraction_alone_does_not_destroy_semantic_content():
    """Shrinking everything by 1000x leaves directions — and therefore cosine
    retrieval — untouched. Magnitude loss and information loss are different
    failures, and a diagnostic suite must be able to tell them apart."""
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(4), 64)
    centroids = np.eye(4, 16) * 10.0
    clustered = centroids[labels] + rng.normal(scale=0.1, size=(256, 16))

    before = diagnose_transform(clustered, "scale_contraction", 0.0, labels=labels)
    after = diagnose_transform(clustered, "scale_contraction", 0.999, labels=labels)

    assert after["total_variance"] < before["total_variance"] / 1000
    assert after["retrieval_p10"] == pytest.approx(before["retrieval_p10"], abs=1e-9)


def test_unknown_transform_is_rejected(embeddings):
    with pytest.raises(ValueError, match="unknown transform"):
        diagnose_transform(embeddings, "not_a_transform", 0.5)


def test_sweep_covers_every_transform_and_severity(embeddings):
    rows = sweep(embeddings, severities=(0.0, 0.5))
    assert len(rows) == len(TRANSFORMS) * 2
    assert {row["transform"] for row in rows} == set(TRANSFORMS)


def test_default_severities_start_at_identity_and_end_near_total():
    assert DEFAULT_SEVERITIES[0] == 0.0
    assert DEFAULT_SEVERITIES[-1] >= 0.99
