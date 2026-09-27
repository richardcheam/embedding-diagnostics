"""Controlled feature-space degradations for probing diagnostic behaviour.

WHY THIS EXISTS. The project previously described its training conditions as
"manufacturing known degeneration modes". That claim was too strong: a training
intervention sets up an optimisation, it does not guarantee which degeneracy
emerges, and the observed modes were read off the runs afterwards. To learn
what a diagnostic is blind to, the transformation has to be known exactly.

These transformations are applied directly to an embedding matrix, so the
mechanics are exact by construction:

    scale_contraction   multiply by s < 1              pure magnitude loss
    rank_truncation     keep the top-k SVD components  pure subspace loss
    mean_injection      add a fixed vector             angular concentration
    isotropic_noise     add N(0, sigma^2)              structure swamped by noise
    mean_interpolation  pull samples toward the mean   loss of distinction

WHAT THIS IS AND IS NOT. It is a controlled stress test that exposes each
metric's invariances and sensitivities. It is NOT ground truth for what
happens during natural training collapse: real degeneration mixes these modes,
arrives gradually, and interacts with the optimiser. Findings here constrain
what a metric CAN detect; they do not establish what it WILL detect in a run.

No expectation is imposed that every metric respond monotonically to every
transformation. Where a metric is invariant, that invariance is the finding.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def scale_contraction(embeddings: np.ndarray, severity: float, seed: int = 0) -> np.ndarray:
    """Multiply by (1 - severity): pure magnitude loss, geometry untouched.

    Severity 0 leaves the input alone; severity 1 sends everything to the
    origin. Scale-invariant diagnostics (participation ratio, RankMe, cosine)
    are blind to this by construction — which is the point.
    """
    return np.asarray(embeddings, dtype=np.float64) * (1.0 - severity)


def rank_truncation(embeddings: np.ndarray, severity: float, seed: int = 0) -> np.ndarray:
    """Keep only the leading SVD components: pure subspace loss.

    Severity maps to the fraction of components discarded, always retaining at
    least one. Total variance falls somewhat (discarded components carried
    some), but the dominant change is dimensional.
    """
    matrix = np.asarray(embeddings, dtype=np.float64)
    full = min(matrix.shape)
    keep = max(1, int(round(full * (1.0 - severity))))
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    s = s.copy()
    s[keep:] = 0.0
    return (u * s) @ vt


def mean_injection(embeddings: np.ndarray, severity: float, seed: int = 0) -> np.ndarray:
    """Add a fixed offset scaled to the data: angular concentration.

    Every sample shifts by the same vector, so the CENTERED covariance is
    untouched while raw directions crowd together. This is the transformation
    that separates centered from uncentered diagnostics, and the one that
    exposed the centering bug in SIGReg.
    """
    matrix = np.asarray(embeddings, dtype=np.float64)
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=matrix.shape[1])
    direction /= max(np.linalg.norm(direction), 1e-12)
    magnitude = float(np.linalg.norm(matrix, axis=1).mean())
    # Grows without bound as severity -> 1 so angular concentration can be driven
    # arbitrarily far, unlike a bounded offset that plateaus.
    strength = severity / max(1.0 - severity, 1e-6)
    return matrix + direction * magnitude * strength


def isotropic_noise(embeddings: np.ndarray, severity: float, seed: int = 0) -> np.ndarray:
    """Add isotropic Gaussian noise scaled to the data's own spread.

    At high severity the noise dominates, so the spectrum flattens: rank
    measures RISE while the representation carries less signal. This is the
    regime that makes rank misleading as a health check.
    """
    matrix = np.asarray(embeddings, dtype=np.float64)
    rng = np.random.default_rng(seed)
    strength = severity / max(1.0 - severity, 1e-6)
    return matrix + rng.normal(scale=matrix.std() * strength, size=matrix.shape)


def mean_interpolation(embeddings: np.ndarray, severity: float, seed: int = 0) -> np.ndarray:
    """Interpolate every sample toward the global mean: loss of distinction.

    At severity 1 all samples equal the mean — complete collapse to a single
    non-zero point, which is what a collapsed encoder actually emits. Distinct
    from `scale_contraction`, which collapses toward the ORIGIN.
    """
    matrix = np.asarray(embeddings, dtype=np.float64)
    centre = matrix.mean(axis=0, keepdims=True)
    return matrix * (1.0 - severity) + centre * severity


TRANSFORMS: dict[str, Callable[[np.ndarray, float, int], np.ndarray]] = {
    "scale_contraction": scale_contraction,
    "rank_truncation": rank_truncation,
    "mean_injection": mean_injection,
    "isotropic_noise": isotropic_noise,
    "mean_interpolation": mean_interpolation,
}

DEFAULT_SEVERITIES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99)


def diagnose_transform(
    embeddings: np.ndarray,
    transform: str,
    severity: float,
    labels: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Apply one transformation and measure every diagnostic on the result.

    Semantic endpoints are included only when labels are supplied, and the
    probe is fit on the transformed features themselves (train == test): the
    question here is what information survives the transformation, not how well
    a probe generalises.
    """
    from .diagnostics.metrics import collapse_metrics
    from .diagnostics.probe import linear_probe_scores, retrieval_chance
    from .diagnostics.retrieval import retrieval_precision_at_k

    if transform not in TRANSFORMS:
        raise ValueError(f"unknown transform {transform!r}; have {sorted(TRANSFORMS)}")
    transformed = TRANSFORMS[transform](np.asarray(embeddings, dtype=np.float64), severity, seed)

    record = {"transform_severity": float(severity)}
    record.update(collapse_metrics(transformed))

    if labels is not None:
        labels = np.asarray(labels)
        scores = linear_probe_scores(transformed, labels, transformed, labels, seed=seed)
        raw = linear_probe_scores(
            transformed, labels, transformed, labels, seed=seed, standardize=False
        )
        record["probe_accuracy"] = scores["accuracy"]
        record["probe_balanced"] = scores["balanced_accuracy"]
        record["probe_accuracy_unscaled"] = raw["accuracy"]
        record["retrieval_p10"] = retrieval_precision_at_k(transformed, labels, k=10)
        record["retrieval_chance"] = retrieval_chance(labels)
    return record


def sweep(
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    severities: tuple[float, ...] = DEFAULT_SEVERITIES,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Every transformation at every severity, as flat records."""
    rows = []
    for name in TRANSFORMS:
        for severity in severities:
            record = diagnose_transform(embeddings, name, severity, labels=labels, seed=seed)
            record["transform"] = name
            rows.append(record)
    return rows
