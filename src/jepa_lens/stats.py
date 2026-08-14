"""Paired across-seed analysis for the pre-declared condition contrasts.

Replaces an earlier heuristic ("a difference is real if it exceeds twice the
seed SD and the probe's binomial standard error"), which was not a valid paired
analysis: it compared a difference against the spread of a single condition and
then applied a fixed threshold as a decision rule.

What is computed instead, for a contrast A - B over seeds shared by both:

    d_i  = endpoint(A, seed i) - endpoint(B, seed i)      per-seed difference
    dbar = mean(d_i)
    s_d  = sample standard deviation of d_i, ddof = 1
    SE   = s_d / sqrt(n)
    CI   = dbar +/- t(0.975, n-1) * SE                    two-sided 95%

Seeds are paired because, within a seed, every condition shares an
initialisation, a mask sequence, a batch order and — since `eval_split_seed`
was separated from the run seed — the same evaluation images. Pairing removes
that shared variation from the contrast.

WHAT THE INTERVAL COVERS. Training-seed variability, conditional on one fixed
evaluation split. It does not cover uncertainty from the choice of split, from
the dataset, or from the architecture. A wider claim needs a wider experiment.

NO VERDICTS. Nothing here labels a result real, significant, learned or
collapsed. It reports an estimate and its uncertainty; interpretation is the
reader's, in the report, with the multiplicity of contrasts in view.

MULTIPLICITY. Five contrasts times several endpoints means dozens of intervals.
They are deliberately reported as intervals rather than tests, and no
correction is applied; the report states this so a reader can discount
accordingly rather than being handed a false guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Pre-declared contrasts. EMA is excluded as a contrast arm on purpose: it
# differs from the shared-encoder conditions in two factors at once (EMA target
# AND regularizer), so it cannot estimate a stop-gradient-only effect. It
# remains an external reference baseline reported on its own.
CONTRASTS: list[tuple[str, str, str]] = [
    ("none_stopgrad", "none_nostopgrad", "stop-gradient effect, no SIGReg"),
    ("sigreg_stopgrad", "sigreg_nostopgrad", "stop-gradient effect, no projector"),
    ("proj_sigreg_stopgrad", "proj_sigreg_nostopgrad", "stop-gradient effect, with projector"),
    ("proj_sigreg_stopgrad", "sigreg_stopgrad", "projector effect, with stop-gradient"),
    ("proj_sigreg_nostopgrad", "sigreg_nostopgrad", "projector effect, without stop-gradient"),
]

# Two-sided 95% Student-t quantiles by degrees of freedom (n - 1). Table rather
# than a SciPy dependency: the project has no SciPy and this covers the seed
# counts any realistic run will use.
_T_TABLE: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    24: 2.064, 29: 2.045, 39: 2.023, 59: 2.001,
}


def t_critical(degrees_of_freedom: int) -> float:
    """Two-sided 95% t quantile; falls back to the nearest tabulated df."""
    if degrees_of_freedom < 1:
        raise ValueError("need at least two paired seeds for an interval")
    if degrees_of_freedom in _T_TABLE:
        return _T_TABLE[degrees_of_freedom]
    available = [df for df in _T_TABLE if df <= degrees_of_freedom]
    return _T_TABLE[max(available)] if available else 1.96


@dataclass
class PairedResult:
    """One contrast on one endpoint."""

    contrast: str
    endpoint: str
    seeds: list[int]
    differences: list[float]
    mean_difference: float
    sd_difference: float
    standard_error: float
    ci_low: float
    ci_high: float
    n_pairs: int


@dataclass
class ConditionSummary:
    """Per-condition descriptive statistics for one endpoint."""

    condition: str
    endpoint: str
    seeds: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else float("nan")

    @property
    def sd(self) -> float:
        if len(self.values) < 2:
            return float("nan")
        return float(np.std(self.values, ddof=1))


def summarize_condition(
    condition: str, endpoint: str, by_seed: dict[int, dict]
) -> ConditionSummary:
    """Descriptives for one condition, failing loudly on a missing endpoint."""
    seeds = sorted(by_seed)
    missing = [seed for seed in seeds if endpoint not in by_seed[seed]]
    if missing:
        raise KeyError(
            f"{condition}: endpoint {endpoint!r} missing at seed(s) {missing}. "
            "Every run in a comparison must log the same endpoints."
        )
    return ConditionSummary(
        condition=condition,
        endpoint=endpoint,
        seeds=seeds,
        values=[float(by_seed[seed][endpoint]) for seed in seeds],
    )


def paired_contrast(
    name_a: str,
    name_b: str,
    endpoint: str,
    runs: dict[str, dict[int, dict]],
    require_matching_steps: bool = True,
) -> PairedResult:
    """Paired difference A - B over the seeds both conditions ran.

    Fails loudly rather than quietly dropping seeds: unequal seed sets, missing
    endpoints, mismatched final steps, or fewer than two pairs all raise. A
    silently half-populated contrast is worse than no contrast.
    """
    for name in (name_a, name_b):
        if name not in runs:
            raise KeyError(f"condition {name!r} absent; have {sorted(runs)}")

    seeds_a, seeds_b = set(runs[name_a]), set(runs[name_b])
    if seeds_a != seeds_b:
        raise ValueError(
            f"{name_a} ran seeds {sorted(seeds_a)} but {name_b} ran {sorted(seeds_b)}. "
            "Paired analysis needs identical seed sets; rerun the missing jobs."
        )

    seeds = sorted(seeds_a)
    if len(seeds) < 2:
        raise ValueError(
            f"{name_a} - {name_b}: {len(seeds)} paired seed(s); an interval needs at least 2."
        )

    if require_matching_steps:
        for seed in seeds:
            step_a = runs[name_a][seed].get("step")
            step_b = runs[name_b][seed].get("step")
            if step_a != step_b:
                raise ValueError(
                    f"{name_a} - {name_b} at seed {seed}: final steps differ "
                    f"({step_a} vs {step_b}). Endpoints must be read at the same step."
                )

    summary_a = summarize_condition(name_a, endpoint, runs[name_a])
    summary_b = summarize_condition(name_b, endpoint, runs[name_b])
    differences = [a - b for a, b in zip(summary_a.values, summary_b.values, strict=True)]

    n = len(differences)
    mean_difference = float(np.mean(differences))
    sd_difference = float(np.std(differences, ddof=1))
    standard_error = sd_difference / np.sqrt(n)
    half_width = t_critical(n - 1) * standard_error

    return PairedResult(
        contrast=f"{name_a} - {name_b}",
        endpoint=endpoint,
        seeds=seeds,
        differences=differences,
        mean_difference=mean_difference,
        sd_difference=sd_difference,
        standard_error=float(standard_error),
        ci_low=mean_difference - half_width,
        ci_high=mean_difference + half_width,
        n_pairs=n,
    )
