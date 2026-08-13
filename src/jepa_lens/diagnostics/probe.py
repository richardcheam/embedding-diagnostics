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
