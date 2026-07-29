import numpy as np

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
