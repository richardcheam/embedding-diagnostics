"""2D PCA projection of embeddings, logged per checkpoint for the animation.

Storing a small fixed sample of projected points at each checkpoint is what
lets the interactive report animate the embedding cloud contracting as a run
collapses. Kept deliberately small so the JSONL stays git-trackable.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import PCA


def project_2d(
    embeddings: np.ndarray,
    max_samples: int = 500,
    seed: int = 0,
) -> list[list[float]]:
    """Project embeddings to 2D with PCA, returning JSON-serializable points.

    Args:
        embeddings: array of shape (samples, dimensions).
        max_samples: subsample cap, applied before fitting.
        seed: seed for the subsampling draw.

    Returns:
        List of `[x, y]` pairs, one per retained sample.
    """
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected (samples, dimensions), got shape {matrix.shape}")

    if len(matrix) > max_samples:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(matrix), size=max_samples, replace=False)
        matrix = matrix[indices]

    n_components = min(2, matrix.shape[0], matrix.shape[1])
    with warnings.catch_warnings():
        # Exactly-collapsed embeddings make sklearn divide by a zero variance
        # while computing explained_variance_ratio_, which this function never
        # reads. The projection itself is still correct. Suppressed because the
        # collapsed condition would otherwise emit this at every checkpoint of
        # a 32k-step run.
        warnings.simplefilter("ignore", RuntimeWarning)
        projected = PCA(n_components=n_components).fit_transform(matrix)
    if projected.shape[1] < 2:
        padding = np.zeros((len(projected), 2 - projected.shape[1]))
        projected = np.hstack([projected, padding])

    return [[float(x), float(y)] for x, y in projected]
