import numpy as np
import pytest
import torch

from embedding_diagnostics.diagnostics.probe import (
    ProbeConfig,
    linear_probe_accuracy,
    linear_probe_scores,
)


def test_separable_features_score_high():
    rng = np.random.default_rng(0)
    train_features = np.vstack([rng.normal(-5, 0.5, (100, 4)), rng.normal(5, 0.5, (100, 4))])
    train_labels = np.array([0] * 100 + [1] * 100)
    test_features = np.vstack([rng.normal(-5, 0.5, (50, 4)), rng.normal(5, 0.5, (50, 4))])
    test_labels = np.array([0] * 50 + [1] * 50)

    accuracy = linear_probe_accuracy(train_features, train_labels, test_features, test_labels)
    assert accuracy > 0.95


def test_collapsed_features_score_near_chance():
    """A collapsed encoder gives the probe nothing to work with."""
    rng = np.random.default_rng(1)
    train_features = np.zeros((200, 4))
    train_labels = rng.integers(0, 2, 200)
    test_features = np.zeros((100, 4))
    test_labels = rng.integers(0, 2, 100)

    accuracy = linear_probe_accuracy(train_features, train_labels, test_features, test_labels)
    assert 0.3 < accuracy < 0.7


def test_returns_plain_float_in_unit_range():
    rng = np.random.default_rng(2)
    accuracy = linear_probe_accuracy(
        rng.normal(size=(80, 4)),
        rng.integers(0, 3, 80),
        rng.normal(size=(40, 4)),
        rng.integers(0, 3, 40),
    )
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0


def test_a_scale_contracted_but_informative_representation_reads_the_same_either_way():
    """WITHDRAWN CLAIM, inverted and pinned.

    This test previously asserted that the standardized probe is "fooled" by
    scale contraction while the unstandardized one is "the more honest of the
    two". That was wrong, and the test was holding the error in place.

    The construction below is a large constant plus a tiny input-dependent
    residue that carries the label perfectly. There is no information loss by
    construction. Under matched optimisation both probes recover it, so a gap
    between them would have measured the optimiser, not the representation.

    Under the old protocol (fixed C=1, tol=1e-4, max_iter=1000) the
    unstandardized probe scored at chance here with n_iter near zero: lbfgs
    stops when the gradient norm drops below tol, the gradient scales with
    feature magnitude, and on a 1e-3 scale the criterion is met at the initial
    all-zero point -- with no ConvergenceWarning.
    """
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, 600)
    signal = np.eye(10)[labels] @ rng.normal(size=(10, 32))
    contracted = np.ones((600, 32)) + 0.0019 * signal / signal.std()

    train = slice(0, 450)
    test = slice(450, 600)
    args = (contracted[train], labels[train], contracted[test], labels[test])

    standardized = linear_probe_scores(*args, standardize=True)
    unscaled = linear_probe_scores(*args, standardize=False)

    assert standardized["accuracy"] > 0.9
    assert unscaled["accuracy"] > 0.9, (
        "with matched optimisation the unstandardized probe must also recover "
        "information that is present by construction"
    )
    assert not unscaled["underfit_train"]


def test_the_old_protocol_would_have_reported_chance_on_that_same_data():
    """Direct evidence for the paragraph above, so the claim is not just
    asserted in prose. Reproduces the old settings explicitly."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, 600)
    signal = np.eye(10)[labels] @ rng.normal(size=(10, 32))
    contracted = np.ones((600, 32)) + 0.0019 * signal / signal.std()

    old = ProbeConfig(c_grid=(1.0,), tol=1e-4, max_iter=1000)
    scores = linear_probe_scores(
        contracted[:450], labels[:450], contracted[450:], labels[450:],
        standardize=False, config=old,
    )
    assert scores["accuracy"] < 0.3, "the old protocol's failure must be reproducible"
    # The tell is that it does not even fit TRAIN: this is gross underfitting
    # from a stopping rule met almost immediately, not overfitting or a hard
    # problem. (Iteration count itself varies with the data -- 13 here, 2 at
    # the 192-dimension scale of the real runs -- so train accuracy is the
    # stable signature.)
    assert scores["train_accuracy"] < 0.3
    # And it did so silently: sklearn raised no ConvergenceWarning at all.
    assert scores["converged"] == 1.0


def test_standardize_flag_defaults_to_true():
    """The default stays the cross-checkpoint-comparable variant."""
    rng = np.random.default_rng(1)
    features = rng.normal(size=(80, 4))
    labels = rng.integers(0, 2, 80)
    explicit = linear_probe_accuracy(features, labels, features, labels, standardize=True)
    default = linear_probe_accuracy(features, labels, features, labels)
    assert explicit == default


def test_majority_rate_is_the_real_floor_on_skewed_labels():
    """BDD's `scene` is 71% one class; accuracy must be read against this."""
    from embedding_diagnostics.diagnostics.probe import majority_rate

    labels = np.array([0] * 71 + [1] * 20 + [2] * 9)
    assert abs(majority_rate(labels) - 0.71) < 1e-12


def test_retrieval_chance_is_class_self_match_not_one_over_k():
    """With two dominant classes, random retrieval already scores ~0.6."""
    from embedding_diagnostics.diagnostics.probe import retrieval_chance

    labels = np.array([0] * 71 + [1] * 28 + [2] * 1)
    expected = 0.71**2 + 0.28**2 + 0.01**2
    assert abs(retrieval_chance(labels) - expected) < 1e-9
    assert retrieval_chance(labels) > 1 / 3  # far above the naive 1/num_classes


def test_balanced_accuracy_exposes_a_majority_only_probe():
    """A probe that has learned nothing but the prior scores high raw accuracy
    and chance-level balanced accuracy. That gap is the point."""
    from embedding_diagnostics.diagnostics.probe import linear_probe_scores

    rng = np.random.default_rng(0)
    # Features carry no class information at all.
    features = rng.normal(size=(400, 6))
    labels = np.array([0] * 340 + [1] * 30 + [2] * 30)
    scores = linear_probe_scores(features, labels, features, labels, min_support=10)

    assert scores["accuracy"] >= scores["majority"] - 0.05  # tracks the prior
    assert scores["balanced_accuracy"] < 0.55  # but is near chance per class


def test_classes_below_min_support_are_dropped_and_counted():
    """BDD has 27 'gas stations' images in 70k; such classes cannot be scored,
    and the count of dropped ones is reported rather than hidden."""
    from embedding_diagnostics.diagnostics.probe import linear_probe_scores

    rng = np.random.default_rng(0)
    features = rng.normal(size=(120, 4))
    labels = np.array([0] * 60 + [1] * 57 + [2] * 3)  # class 2 has 3 samples
    scores = linear_probe_scores(features, labels, features, labels, min_support=10)
    assert scores["scored_classes"] == 2
    assert scores["dropped_classes"] == 1


def test_evaluate_logs_a_floor_for_every_headline_number():
    from embedding_diagnostics.training.trainer import Trainer
    from test_phase2_conditions import tiny_config

    trainer = Trainer(tiny_config("none_stopgrad"))
    rng = np.random.default_rng(0)
    skewed = np.array([0] * 9 + [1])
    record = trainer.evaluate(
        (torch.randn(12, 3, 32, 32), rng.integers(0, 2, 12)),
        (torch.randn(10, 3, 32, 32), skewed),
    )
    for key in ("probe_majority", "retrieval_chance", "probe_over_majority",
                "retrieval_over_chance", "probe_balanced"):
        assert key in record, key
    assert abs(record["probe_majority"] - 0.9) < 1e-12
    assert abs(
        record["probe_over_majority"] - (record["probe_accuracy"] - record["probe_majority"])
    ) < 1e-12


def test_chance_adjustment_rescales_so_zero_is_chance():
    """On BDD's scene, chance retrieval is ~0.46; raw 0.50 is 0.07 adjusted,
    not the '3x chance' a reader would infer from 1/K."""
    from embedding_diagnostics.diagnostics.probe import chance_adjusted

    assert chance_adjusted(0.46, 0.46) == pytest.approx(0.0)
    assert chance_adjusted(1.0, 0.46) == pytest.approx(1.0)
    assert chance_adjusted(0.50, 0.46) == pytest.approx(0.074, abs=0.001)
    assert chance_adjusted(0.30, 0.46) < 0  # below chance is negative, not clipped


def test_chance_adjustment_handles_a_degenerate_denominator():
    from embedding_diagnostics.diagnostics.probe import chance_adjusted

    assert np.isnan(chance_adjusted(1.0, 1.0))  # single-class split
    assert np.isnan(chance_adjusted(0.5, float("nan")))


def test_macro_f1_penalises_a_probe_that_over_predicts_the_majority():
    """Balanced accuracy is recall-only; macro-F1 also charges for the
    false positives a majority-latching probe generates."""
    from embedding_diagnostics.diagnostics.probe import linear_probe_scores

    rng = np.random.default_rng(0)
    features = rng.normal(size=(400, 6))  # carries no class information
    labels = np.array([0] * 340 + [1] * 30 + [2] * 30)
    scores = linear_probe_scores(features, labels, features, labels, min_support=10)
    assert scores["macro_f1"] < scores["accuracy"]


def test_macro_retrieval_weights_rare_classes_equally():
    """Micro P@10 is dominated by the majority class; macro is not."""
    from embedding_diagnostics.diagnostics.retrieval import (
        retrieval_macro_precision_at_k,
        retrieval_precision_at_k,
    )

    rng = np.random.default_rng(0)
    # A large, tight majority cluster and a small, scattered minority.
    majority = rng.normal(scale=0.05, size=(180, 8)) + np.array([9.0] + [0.0] * 7)
    minority = rng.normal(scale=4.0, size=(20, 8))
    features = np.vstack([majority, minority])
    labels = np.array([0] * 180 + [1] * 20)

    micro = retrieval_precision_at_k(features, labels, k=10)
    macro, per_class = retrieval_macro_precision_at_k(features, labels, k=10)
    assert micro > macro, "micro is flattered by the easy majority class"
    assert per_class[0] > per_class[1]


def test_macro_retrieval_drops_classes_below_support():
    from embedding_diagnostics.diagnostics.retrieval import retrieval_macro_precision_at_k

    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 4))
    labels = np.array([0] * 30 + [1] * 27 + [2] * 3)
    _, per_class = retrieval_macro_precision_at_k(features, labels, k=5, min_support=10)
    assert set(per_class) == {0, 1}


def test_evaluate_logs_macro_f1_and_adjusted_retrieval_per_attribute():
    from embedding_diagnostics.training.trainer import Trainer
    from test_phase2_conditions import tiny_config

    trainer = Trainer(tiny_config("none_stopgrad"))
    rng = np.random.default_rng(0)
    labels_train = {"weather": rng.integers(0, 3, 12), "scene": rng.integers(0, 2, 12)}
    labels_test = {"weather": rng.integers(0, 3, 10), "scene": rng.integers(0, 2, 10)}
    record = trainer.evaluate(
        (torch.randn(12, 3, 32, 32), labels_train),
        (torch.randn(10, 3, 32, 32), labels_test),
    )
    for attribute in ("weather", "scene"):
        for key in ("probe_macro_f1", "retrieval_macro_p10", "retrieval_adjusted"):
            assert f"{key}_{attribute}" in record, f"{key}_{attribute}"
    # The across-attribute aggregate uses the adjusted form, never raw P@10.
    assert "retrieval_adjusted" in record
    assert "probe_macro_f1" in record


# --- exact-constant negative control ------------------------------------
#
# An exactly constant encoder E(x) = c carries no input-dependent structure at
# all. No preprocessing can manufacture label information from it. These pin
# that floor, and separate it from the near-constant case above, which is a
# different thing entirely and must not be described with the same words.


def constant_features(samples=600, dim=32, value=1.0):
    return np.full((samples, dim), value)


def test_an_exactly_constant_encoder_cannot_be_probed_above_chance():
    """The genuine floor. Standardization cannot create information here:
    StandardScaler maps a zero-variance column to zeros."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, 600)
    features = constant_features()

    for standardize in (True, False):
        scores = linear_probe_scores(
            features[:450], labels[:450], features[450:], labels[450:],
            standardize=standardize,
        )
        # Only the majority class can ever be predicted, so accuracy equals the
        # majority rate rather than 1/k.
        assert scores["accuracy"] <= scores["majority"] + 1e-9


def test_standardizing_an_exact_constant_yields_no_signal():
    """The mechanism: zero variance in, zeros out. Contrast with the
    near-constant case, where a real residue survives division."""
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(constant_features())
    assert np.allclose(scaled, 0.0)


def test_an_exact_constant_is_distinguishable_from_a_contracted_representation():
    """These are different failures and the project must not conflate them:
    one has no recoverable information, the other has it at a small scale."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, 600)
    signal = np.eye(10)[labels] @ rng.normal(size=(10, 32))
    contracted = np.ones((600, 32)) + 1e-3 * signal / signal.std()

    exact = linear_probe_scores(
        constant_features()[:450], labels[:450], constant_features()[450:], labels[450:],
    )
    near = linear_probe_scores(
        contracted[:450], labels[:450], contracted[450:], labels[450:],
    )
    assert exact["accuracy"] <= exact["majority"] + 1e-9
    assert near["accuracy"] > 0.9


# --- the regularisation / stopping-rule confound ------------------------


def test_a_fixed_C_makes_probe_accuracy_depend_on_feature_scale():
    """Same information, different scale, fixed C: the score moves. This is why
    a raw-versus-standardized comparison at fixed C measures the protocol."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, 400)
    base = np.eye(4, 16)[labels] + 0.05 * rng.normal(size=(400, 16))
    fixed = ProbeConfig(c_grid=(1.0,), tol=1e-4, max_iter=1000)

    big = linear_probe_scores(base[:300], labels[:300], base[300:], labels[300:],
                              standardize=False, config=fixed)
    small = linear_probe_scores(base[:300] * 1e-3, labels[:300], base[300:] * 1e-3,
                                labels[300:], standardize=False, config=fixed)
    assert big["accuracy"] - small["accuracy"] > 0.3


def test_selecting_C_on_validation_removes_most_of_that_scale_dependence():
    """The correction. The same two representations, same information, now
    scored comparably because the regularisation is chosen per representation."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, 400)
    base = np.eye(4, 16)[labels] + 0.05 * rng.normal(size=(400, 16))

    big = linear_probe_scores(base[:300], labels[:300], base[300:], labels[300:],
                              standardize=False)
    small = linear_probe_scores(base[:300] * 1e-3, labels[:300], base[300:] * 1e-3,
                                labels[300:], standardize=False)
    assert abs(big["accuracy"] - small["accuracy"]) < 0.1


def test_C_is_never_selected_on_the_test_split():
    """Selection touching test would inflate every reported number. Verified by
    giving test labels that are pure noise: the choice must not react."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, 400)
    features = np.eye(4, 16)[labels] + 0.1 * rng.normal(size=(400, 16))

    honest = linear_probe_scores(features[:300], labels[:300], features[300:], labels[300:])
    shuffled = linear_probe_scores(
        features[:300], labels[:300], features[300:], rng.permutation(labels[300:])
    )
    assert honest["selected_C"] == shuffled["selected_C"]


# --- convergence reporting ----------------------------------------------


def test_every_score_carries_its_optimisation_metadata():
    """A score without these is uninterpretable, so they cannot be optional."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, 200)
    features = rng.normal(size=(200, 8)) + np.eye(3, 8)[labels]
    scores = linear_probe_scores(features[:150], labels[:150], features[150:], labels[150:])

    for key in ("n_iter", "converged", "underfit_train", "warnings",
                "selected_C", "feature_scale", "train_accuracy", "standardized"):
        assert key in scores, key


def test_a_convergence_failure_is_recorded_rather_than_swallowed():
    """Capped iterations on a hard problem must surface in the record."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, 300)
    features = rng.normal(size=(300, 40))
    capped = ProbeConfig(c_grid=(1e5,), tol=1e-14, max_iter=2)
    scores = linear_probe_scores(features[:200], labels[:200], features[200:], labels[200:],
                                 standardize=False, config=capped)
    assert scores["converged"] == 0.0
    assert scores["warnings"]


def test_the_silent_failure_is_flagged_even_though_sklearn_reports_success():
    """The dangerous case. lbfgs stops after two or three iterations having
    learned nothing, and raises NO ConvergenceWarning -- so `converged` alone
    would certify it. `underfit_train` is what exposes it."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, 400)
    tiny = np.ones((400, 32)) + 1e-9 * rng.normal(size=(400, 32))
    loose = ProbeConfig(c_grid=(1.0,), tol=1e-4, max_iter=1000)
    scores = linear_probe_scores(tiny[:300], labels[:300], tiny[300:], labels[300:],
                                 standardize=False, config=loose)
    assert scores["underfit_train"] == 1.0
    assert scores["converged"] == 1.0, "no warning is raised: that is the danger"
    assert scores["n_iter"] > 0, "and it is not detectable from the iteration count"
