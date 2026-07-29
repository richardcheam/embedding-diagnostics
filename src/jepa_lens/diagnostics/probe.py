"""Frozen-encoder linear probe.

Standard self-supervised evaluation practice, not a contribution of this
project: freeze the encoder, fit a linear classifier on its embeddings, and
report held-out accuracy. This is the trustworthy but expensive signal that the
cheap diagnostics in `metrics.py` are being tested against.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def linear_probe_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    seed: int = 0,
) -> float:
    """Fit logistic regression on frozen features, return held-out accuracy.

    Features are standardized first so the regularization strength means the
    same thing across checkpoints, whose embedding scales drift during training.
    """
    scaler = StandardScaler().fit(train_features)
    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(scaler.transform(train_features), train_labels)
    predictions = classifier.predict(scaler.transform(test_features))
    return float((predictions == test_labels).mean())
