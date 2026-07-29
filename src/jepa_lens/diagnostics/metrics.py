"""Standard self-supervised-learning collapse diagnostics.

These metrics are established practice in the SSL literature and are not a
contribution of this project. They are reimplemented here so the four
conditions are measured by identical code.

Note on effective rank: this module uses the participation-ratio definition,
`(sum lambda)^2 / sum(lambda^2)`, not the entropy-based Roy & Vetterli
definition. The two disagree numerically; comparisons against other papers
must check which variant was used.
"""

from __future__ import annotations

import numpy as np


def effective_rank(eigenvalues: np.ndarray) -> float:
    """Participation-ratio effective rank of a non-negative eigenvalue spectrum.

    Returns a value in [1, len(eigenvalues)]. Near 1 means a single direction
    dominates; near the count means variance is spread evenly.
    """
    values = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)
    total = values.sum()
    squared_total = np.square(values).sum()
    if squared_total <= 0.0:
        # A spectrum with no variance at all is maximal collapse. The ratio is
        # 0/0 here, so take the limit of the dominant-direction case rather
        # than returning 0.0 — otherwise the metric jumps discontinuously
        # below its own documented floor exactly when a run collapses hardest,
        # which is the case this study most needs to plot correctly.
        return 1.0
    return float(total**2 / squared_total)


def collapse_metrics(embeddings: np.ndarray) -> dict[str, float]:
    """Compute cheap collapse diagnostics for a batch of embeddings.

    Args:
        embeddings: array of shape (samples, dimensions).

    Returns:
        Dict of scalar diagnostics. Training loss is deliberately not among
        them: regularized objectives plateau at non-zero values whether or not
        the encoder has collapsed, so loss is not a collapse signal.
    """
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected (samples, dimensions), got shape {matrix.shape}")

    # ddof=1 to match the sample covariance below. Mixing population std with
    # sample covariance would report the spread of the same embeddings under two
    # different conventions, which is confusing when the two are read side by
    # side on the same plot.
    feature_std = matrix.std(axis=0, ddof=1) if len(matrix) > 1 else np.zeros(matrix.shape[1])
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(matrix) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.clip(norms, 1e-12, None)
    similarity = normalized @ normalized.T
    upper = similarity[np.triu_indices(len(matrix), k=1)]

    return {
        "mean_feature_std": float(feature_std.mean()),
        "min_feature_std": float(feature_std.min()),
        "effective_rank": effective_rank(eigenvalues),
        "mean_pairwise_cosine": float(upper.mean()) if upper.size else 0.0,
        "std_pairwise_cosine": float(upper.std()) if upper.size else 0.0,
    }
