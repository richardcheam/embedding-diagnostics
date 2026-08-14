"""Nearest-neighbour retrieval quality on frozen embeddings.

Standard evaluation practice, not a contribution of this project. Retrieval is
the operation scenario mining actually performs — "find me situations like this
one" — so for driving data it is the primary semantic endpoint, ahead of the
linear probes: an embedding can be linearly decodable yet organised so that
nearest neighbours are useless, and vice versa.

Cosine similarity on L2-normalized embeddings, so this measures the DIRECTION
structure only. Like every direction-based metric in this project it is blind
to scale collapse on its own; it is always logged beside `total_variance`.
"""

from __future__ import annotations

import numpy as np


def retrieval_precision_at_k(
    features: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> float:
    """Mean fraction of each sample's k nearest neighbours sharing its label.

    Self-matches are excluded. With fewer than k+1 samples, k shrinks to what
    is available. Chance level is the label distribution's self-match
    probability (sum of squared class frequencies), not 1/num_classes.
    """
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected (samples, dimensions), got shape {matrix.shape}")
    labels = np.asarray(labels)
    if len(labels) != len(matrix):
        raise ValueError(f"{len(matrix)} samples but {len(labels)} labels")
    count = len(matrix)
    if count < 2:
        return 0.0
    k = min(k, count - 1)

    normalized = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)  # a sample must not retrieve itself

    neighbour_indices = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    matches = labels[neighbour_indices] == labels[:, None]
    return float(matches.mean())
