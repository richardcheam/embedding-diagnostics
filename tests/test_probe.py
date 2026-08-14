import numpy as np
import torch

from jepa_lens.diagnostics.probe import linear_probe_accuracy


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


def test_standardizing_hides_scale_collapse_but_unscaled_does_not():
    """The measurement defect the pilot exposed, pinned as a test.

    A collapsed encoder emits a large constant plus a tiny input-dependent
    residue. StandardScaler divides that residue back up to unit variance, so
    the standardized probe reads it perfectly while the representation is, by
    every other measure, collapsed.
    """
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, 600)
    signal = np.eye(10)[labels] @ rng.normal(size=(10, 32))
    collapsed = np.ones((600, 32)) + 0.0019 * signal / signal.std()

    train = slice(0, 450)
    test = slice(450, 600)
    args = (collapsed[train], labels[train], collapsed[test], labels[test])

    standardized = linear_probe_accuracy(*args, standardize=True)
    unscaled = linear_probe_accuracy(*args, standardize=False)

    assert standardized > 0.9, "standardized probe should be fooled by scale collapse"
    assert unscaled < standardized, "unscaled probe must be the more honest of the two"


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
    from jepa_lens.diagnostics.probe import majority_rate

    labels = np.array([0] * 71 + [1] * 20 + [2] * 9)
    assert abs(majority_rate(labels) - 0.71) < 1e-12


def test_retrieval_chance_is_class_self_match_not_one_over_k():
    """With two dominant classes, random retrieval already scores ~0.6."""
    from jepa_lens.diagnostics.probe import retrieval_chance

    labels = np.array([0] * 71 + [1] * 28 + [2] * 1)
    expected = 0.71**2 + 0.28**2 + 0.01**2
    assert abs(retrieval_chance(labels) - expected) < 1e-9
    assert retrieval_chance(labels) > 1 / 3  # far above the naive 1/num_classes


def test_balanced_accuracy_exposes_a_majority_only_probe():
    """A probe that has learned nothing but the prior scores high raw accuracy
    and chance-level balanced accuracy. That gap is the point."""
    from jepa_lens.diagnostics.probe import linear_probe_scores

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
    from jepa_lens.diagnostics.probe import linear_probe_scores

    rng = np.random.default_rng(0)
    features = rng.normal(size=(120, 4))
    labels = np.array([0] * 60 + [1] * 57 + [2] * 3)  # class 2 has 3 samples
    scores = linear_probe_scores(features, labels, features, labels, min_support=10)
    assert scores["scored_classes"] == 2
    assert scores["dropped_classes"] == 1


def test_evaluate_logs_a_floor_for_every_headline_number():
    from jepa_lens.training.trainer import Trainer
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
