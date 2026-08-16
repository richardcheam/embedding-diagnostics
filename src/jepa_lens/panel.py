"""A minimal sufficient panel of label-free collapse diagnostics.

This module turns the project's central negative result into something usable.
Phase A and the BDD pilot established that no single diagnostic can be trusted;
the constructive question is then: what is the smallest set that *can*, and how
should it be read?

Three findings are encoded here, each of which constrains the design.

FINDING 1 — no single diagnostic covers every collapse mode. Verified
exhaustively against the five controlled degradations in `degradation.py`
(`minimal_panels` does the search). Every singleton fails on at least one mode,
because each candidate is invariant to something: the scale-invariant metrics
(cosine, RankMe, participation ratio) cannot see pure scale contraction, and
the centered metrics (total variance, participation ratio) cannot see a shift
of the whole cloud away from the origin. Exactly four pairs suffice.

FINDING 2 — drift-based alarms are useless here. The obvious monitoring rule,
"alarm when a diagnostic moves far from its value at initialisation", fires on
healthy training: on CIFAR-10 it flagged six of seven conditions, and on
BDD100K it flagged `ema_stopgrad`, the one condition that actually learns
(total variance 2.2x, mean pairwise cosine 0.40x from init). Healthy
self-supervised training legitimately reshapes the embedding geometry, so
movement carries almost no information about health. The panel therefore uses
ABSOLUTE thresholds near each metric's degenerate limit, not drift.

FINDING 3 — a geometric alarm does not imply semantic loss. Of the five
controlled degradations, three (scale contraction, mean injection, mean
interpolation) leave retrieval P@10 at 1.000 while wrecking the geometry: the
residual structure still orders neighbours correctly. The panel's claim is
therefore the narrower and defensible one --- *the embedding has entered a
regime where your scale-invariant evaluation metrics are no longer
trustworthy* --- not "the representation is broken". Deciding the latter needs
a labelled probe on unstandardized features, which is what `verdict` reports
separately when it is available.

THRESHOLD PROVENANCE. The numbers in `PanelThresholds` are calibrated on this
project's own runs (7 conditions x 5 seeds CIFAR-10, x 3 seeds BDD100K) and are
not claimed to transfer. They sit roughly midway in log space between the
collapsed control and the worst genuinely-training condition; the observed
margins are recorded in `OBSERVED_MARGINS` so a reader can judge how much slack
there was. Recalibrate before using them on a different architecture or
embedding dimension.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

# Candidates considered for membership. `participation_ratio` is deliberately
# included so the search can reject it on the evidence rather than by fiat --
# and it does: it appears in no minimal panel, and on both datasets it reads
# HIGHER on the collapsed control than on the healthy baseline (CIFAR 34.98 vs
# 26.90; BDD 26.90 vs 11.21), because it is computed from the CENTERED
# covariance and so cannot see collapse to a non-zero constant.
CANDIDATES = (
    "total_variance",
    "mean_pairwise_cosine",
    "participation_ratio",
    "rankme",
    "mean_feature_std",
)

# The pair this project recommends, from the four the search returns. Chosen
# over {total_variance, mean_pairwise_cosine} because RankMe degrades
# gracefully -- it reports *how many* dimensions survive rather than saturating
# -- and over the feature-std variants because total variance is the more
# widely logged of the two scale measures.
RECOMMENDED_PANEL = ("total_variance", "rankme")


@dataclass(frozen=True)
class PanelThresholds:
    """Absolute limits below/above which a representation is called degenerate.

    See THRESHOLD PROVENANCE in the module docstring: calibrated, not universal.
    """

    rankme_floor: float = 3.0
    cosine_ceiling: float = 0.99
    feature_std_floor: float = 0.01
    total_variance_floor: float = 0.05


# Worst-case separation actually observed, as (collapsed, worst genuinely
# training). A reader can divide these to see the margin each threshold had.
OBSERVED_MARGINS = {
    "rankme": {"cifar10": (1.12, 9.67), "bdd100k": (1.06, 9.72)},
    "mean_pairwise_cosine": {"cifar10": (1.0000, 0.8497), "bdd100k": (1.0000, 0.6379)},
    "mean_feature_std": {"cifar10": (0.00106, 0.05697), "bdd100k": (0.00058, 0.09743)},
    "total_variance": {"cifar10": (0.0002, 0.6933), "bdd100k": (0.0001, 4.1716)},
}


def mode_coverage(
    embeddings: np.ndarray,
    severity: float = 0.99,
    ratio_threshold: float = 1.5,
    seed: int = 0,
    candidates: tuple[str, ...] = CANDIDATES,
) -> dict[str, set[str]]:
    """Which candidate diagnostics respond to each controlled degradation.

    A diagnostic "responds" when the transformation moves it by more than
    `ratio_threshold` in either direction. Direction is deliberately ignored:
    under isotropic noise total variance rises by four orders of magnitude and
    the rank measures rise too, which is anomalous and worth flagging even
    though it is the opposite of the shrinkage people watch for.

    Returns {transform name: set of diagnostics that responded}.
    """
    from .degradation import TRANSFORMS, diagnose_transform

    baseline = diagnose_transform(embeddings, "scale_contraction", 0.0, seed=seed)
    coverage: dict[str, set[str]] = {}
    for name in TRANSFORMS:
        degraded = diagnose_transform(embeddings, name, severity, seed=seed)
        responded = set()
        for diagnostic in candidates:
            before, after = baseline.get(diagnostic), degraded.get(diagnostic)
            if before is None or after is None or abs(before) < 1e-12:
                continue
            ratio = after / before
            if ratio > ratio_threshold or ratio < 1.0 / ratio_threshold:
                responded.add(diagnostic)
        coverage[name] = responded
    return coverage


def minimal_panels(coverage: dict[str, set[str]]) -> list[tuple[str, ...]]:
    """Every smallest subset of diagnostics that responds to all modes.

    Exhaustive rather than greedy: the point is to *prove* no smaller panel
    exists, which a greedy cover cannot do. The candidate set is five, so the
    32-subset search is free.

    Returns all panels of the minimum sufficient size, or [] if even the full
    candidate set leaves a mode undetected.
    """
    modes = list(coverage)
    present = sorted({d for responded in coverage.values() for d in responded})
    for size in range(1, len(present) + 1):
        sufficient = [
            combination
            for combination in itertools.combinations(present, size)
            if all(coverage[mode] & set(combination) for mode in modes)
        ]
        if sufficient:
            return sufficient
    return []


def verdict(
    metrics: dict[str, float],
    thresholds: PanelThresholds | None = None,
) -> dict[str, object]:
    """Apply the absolute panel checks to one checkpoint's logged metrics.

    Missing metrics are skipped rather than treated as passing, and reported in
    `missing`, so a partial log cannot silently produce a clean bill of health.

    Returns a dict with:
        degenerate   True if any check tripped
        tripped      names of the checks that tripped, with their readings
        missing      panel metrics absent from `metrics`
        semantic     the unstandardized probe reading if present, else None
        note         how to read the result
    """
    thresholds = thresholds or PanelThresholds()
    checks = {
        "rankme": (thresholds.rankme_floor, "below"),
        "mean_pairwise_cosine": (thresholds.cosine_ceiling, "above"),
        "mean_feature_std": (thresholds.feature_std_floor, "below"),
        "total_variance": (thresholds.total_variance_floor, "below"),
    }

    tripped: dict[str, str] = {}
    missing: list[str] = []
    for name, (limit, direction) in checks.items():
        value = metrics.get(name)
        if value is None:
            missing.append(name)
            continue
        if (direction == "below" and value < limit) or (direction == "above" and value > limit):
            tripped[name] = f"{value:.6g} {direction} {limit:g}"

    # The unstandardized probe is the only semantic reading that survives scale
    # collapse; the standardized one rates a dead encoder second of seven.
    semantic = metrics.get("probe_accuracy_unscaled")

    return {
        "degenerate": bool(tripped),
        "tripped": tripped,
        "missing": missing,
        "semantic": semantic,
        "note": (
            "A tripped check means the embedding is in a regime where "
            "scale-invariant metrics (standardized probes, cosine retrieval) are "
            "untrustworthy. It does NOT by itself mean the representation carries "
            "no information -- three of five controlled degradations leave "
            "retrieval intact. Confirm with an unstandardized probe."
        ),
    }
