"""Frozen-encoder linear probe.

Standard self-supervised evaluation practice, not a contribution of this
project: freeze the encoder, fit a linear classifier on its embeddings, and
report held-out accuracy. This is the trustworthy but expensive signal that the
cheap diagnostics in `metrics.py` are being tested against.

IMPORTANT — standardization hides scale collapse. Dividing each feature by its
standard deviation is what keeps accuracy comparable across checkpoints as
embedding scale drifts, but it also rescales a collapsed representation's
numerical noise back to unit variance. That noise is still a deterministic
function of the input, so a linear model reads it happily.

Measured here on synthetic embeddings with mean pairwise cosine 1.0000 (total
collapse): the standardized probe scored 1.000 while the unstandardized probe
scored 0.592, against a chance floor of 0.100. In this project's own pilot, the
condition with no collapse prevention reached cosine 1.000 and per-feature std
0.0019 while its standardized probe accuracy *rose* from 0.375 to 0.409.

Both variants are therefore reported. The standardized one answers "is the
direction structure linearly decodable"; the unstandardized one answers "is it
decodable at the scale the encoder actually emits". A widening gap between them
is itself a collapse signal.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MIN_SUPPORT = 10  # per-class test samples needed before a class is scored


def linear_probe_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    seed: int = 0,
    standardize: bool = True,
) -> float:
    """Fit logistic regression on frozen features, return held-out accuracy.

    Args:
        standardize: divide features by their per-dimension standard deviation,
            fit on train only. True keeps the regularization strength meaning
            the same across checkpoints whose embedding scale drifts. False
            leaves scale intact, so a collapsed representation actually reads
            as collapsed. See the module docstring — the two answer different
            questions and this project logs both.
    """
    if standardize:
        scaler = StandardScaler().fit(train_features)
        train_features = scaler.transform(train_features)
        test_features = scaler.transform(test_features)

    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    return float((predictions == test_labels).mean())


def majority_rate(labels: np.ndarray) -> float:
    """Accuracy of always predicting the most common class.

    The only honest floor for raw accuracy on skewed data. BDD100K's `scene`
    attribute is 71% "city street", so a probe scoring 0.71 has learned
    nothing — without this number beside it, that reads as a result.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / counts.sum())


def retrieval_chance(labels: np.ndarray) -> float:
    """Expected P@k for a random neighbour: the class self-match probability.

    Sum of squared class frequencies, NOT 1/num_classes. On BDD's `scene` this
    is about 0.60 because two classes dominate, so a P@10 of 0.65 is nearly
    indistinguishable from retrieving at random.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    frequencies = counts / counts.sum()
    return float(np.square(frequencies).sum())


def linear_probe_scores(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    seed: int = 0,
    standardize: bool = True,
    min_support: int = MIN_SUPPORT,
) -> dict[str, float]:
    """Probe accuracy, balanced accuracy, and the floors both must beat.

    Returns:
        accuracy          - raw, comparable to `majority` below
        balanced_accuracy - macro-averaged recall over classes with at least
                            `min_support` test samples; unaffected by class
                            skew, which is what makes it the number to read on
                            BDD100K
        majority          - always-predict-the-largest-class accuracy
        scored_classes    - how many classes cleared min_support
        dropped_classes   - how many were too rare to score. On BDD's `scene`
                            this is typically 2-3 of 6, and those are the
                            safety-relevant ones (tunnel, gas station), so the
                            number is reported rather than hidden.
    """
    if standardize:
        scaler = StandardScaler().fit(train_features)
        train_features = scaler.transform(train_features)
        test_features = scaler.transform(test_features)

    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    test_labels = np.asarray(test_labels)
    classes, counts = np.unique(test_labels, return_counts=True)
    scorable = classes[counts >= min_support]
    recalls = [
        float((predictions[test_labels == cls] == cls).mean())
        for cls in scorable
    ]

    # Macro-F1 over the same scorable classes: unlike balanced accuracy it
    # also penalises over-predicting a class, which a probe that has latched
    # onto the majority prior will do.
    f1s = []
    for cls in scorable:
        true_positive = float(((predictions == cls) & (test_labels == cls)).sum())
        predicted = float((predictions == cls).sum())
        actual = float((test_labels == cls).sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1s.append(
            2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        )

    return {
        "accuracy": float((predictions == test_labels).mean()),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "majority": majority_rate(test_labels),
        "scored_classes": float(len(scorable)),
        "dropped_classes": float(len(classes) - len(scorable)),
    }


def chance_adjusted(score: float, chance: float) -> float:
    """Rescale a score so 0 is chance and 1 is perfect.

        (score - chance) / (1 - chance)

    On BDD100K's `scene`, random retrieval already scores about 0.46, so a raw
    P@10 of 0.50 is 0.07 adjusted rather than the "three times chance" a reader
    would infer from 1/K. Returns NaN when chance is at or above 1, where the
    rescaling is undefined (a single-class split).
    """
    if not np.isfinite(chance) or chance >= 1.0:
        return float("nan")
    return float((score - chance) / (1.0 - chance))
