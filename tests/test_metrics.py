import numpy as np

from jepa_lens.diagnostics.metrics import collapse_metrics, effective_rank


def test_constant_embeddings_have_effective_rank_near_one():
    """A fully collapsed representation puts all variance in no direction."""
    embeddings = np.ones((128, 16), dtype=np.float64)
    metrics = collapse_metrics(embeddings)
    assert metrics["mean_feature_std"] < 1e-8
    assert metrics["effective_rank"] >= 1.0
    assert metrics["effective_rank"] < 1.5


def test_isotropic_embeddings_have_high_effective_rank():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(2048, 16))
    metrics = collapse_metrics(embeddings)
    assert metrics["effective_rank"] > 12.0


def test_effective_rank_of_single_dominant_direction_is_near_one():
    """One eigenvalue dominating means effective rank near 1."""
    eigenvalues = np.array([100.0, 0.01, 0.01, 0.01])
    assert effective_rank(eigenvalues) < 1.1


def test_effective_rank_of_all_zero_spectrum_is_one():
    """All-zero spectrum represents maximal collapse; should be at floor of [1, n]."""
    eigenvalues = np.array([0.0, 0.0, 0.0, 0.0])
    assert effective_rank(eigenvalues) == 1.0


def test_effective_rank_of_equal_eigenvalues_equals_count():
    eigenvalues = np.array([2.0, 2.0, 2.0, 2.0])
    assert abs(effective_rank(eigenvalues) - 4.0) < 1e-9


def test_collapsed_embeddings_have_cosine_near_one():
    base = np.ones((64, 8))
    metrics = collapse_metrics(base + 1e-6)
    assert metrics["mean_pairwise_cosine"] > 0.99


def test_all_expected_keys_present():
    rng = np.random.default_rng(1)
    metrics = collapse_metrics(rng.normal(size=(64, 8)))
    expected = {
        "mean_feature_std",
        "min_feature_std",
        "total_variance",
        "effective_rank",
        "mean_pairwise_cosine",
        "std_pairwise_cosine",
    }
    assert set(metrics) == expected
    assert all(isinstance(value, float) for value in metrics.values())


def test_feature_std_uses_the_same_ddof_as_the_covariance():
    """Both spread measures must follow one convention.

    They are read side by side on the same plot, so reporting one as a
    population statistic and the other as a sample statistic invites
    misreading. The covariance uses n-1, so the per-feature std must too.
    """
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(9, 3))
    metrics = collapse_metrics(embeddings)
    expected = float(embeddings.std(axis=0, ddof=1).mean())
    assert abs(metrics["mean_feature_std"] - expected) < 1e-12


def test_single_sample_does_not_produce_nan():
    """One sample has no sample-variance; it must not leak NaN into the log."""
    metrics = collapse_metrics(np.ones((1, 4)))
    assert all(v == v for v in metrics.values())


def test_total_variance_falls_as_a_representation_collapses():
    """The scale signal effective_rank cannot provide.

    effective_rank is a ratio of eigenvalue sums and so is scale-invariant;
    total_variance is what distinguishes a healthy spread from a collapsed one.
    """
    rng = np.random.default_rng(0)
    healthy = rng.normal(size=(256, 16))
    collapsed = healthy * 1e-3

    assert collapse_metrics(collapsed)["total_variance"] < collapse_metrics(healthy)[
        "total_variance"
    ]


def test_effective_rank_rises_under_noise_dominated_collapse():
    """Pins the failure mode the pilot exposed, so it cannot silently change.

    A representation whose signal has collapsed into isotropic numerical noise
    has HIGH effective rank, not low. This test documents that effective rank
    must never be read as a standalone collapse detector — total_variance is
    what actually falls.
    """
    rng = np.random.default_rng(0)
    dominated = rng.normal(size=(512, 16)) @ np.diag([10.0] + [0.001] * 15)
    noise_only = rng.normal(size=(512, 16)) * 1e-8

    assert collapse_metrics(dominated)["effective_rank"] < 2.0
    assert collapse_metrics(noise_only)["effective_rank"] > 10.0
    # ...while the scale metric orders them the way a reader expects.
    assert collapse_metrics(noise_only)["total_variance"] < collapse_metrics(dominated)[
        "total_variance"
    ]
