"""Frozen-encoder linear probe.

Standard self-supervised evaluation practice, not a contribution of this
project: freeze the encoder, fit a linear classifier on its embeddings, and
report held-out accuracy.

THE CONFOUND THIS MODULE EXISTS TO CONTROL. Comparing a probe on standardized
features against one on raw features is NOT a clean comparison of "is the
information there" unless the optimisation is matched, and by default it is
not. Two separate scale couplings bite:

1. Fixed L2 strength. `C` is defined relative to the feature scale, so shrinking
   features by 1e3 makes the same `C` enormously stronger.
2. The stopping rule -- and this is the one that actually dominates here.
   lbfgs stops when the gradient norm falls below `tol`, and the gradient scales
   with feature magnitude. On features with per-dimension std around 1e-3,
   sklearn's default `tol=1e-4` is met at the all-zero initial point: the solver
   returns after ZERO iterations, predicts the majority class, scores chance,
   and emits NO ConvergenceWarning. It reports success.

Measured, on synthetic data whose label signal is perfectly present by
construction, at the per-feature scale of this project's most contracted run
(192 dimensions, std about 8e-4): the unstandardized probe scored 0.096 against
a 0.100 chance floor while the standardized probe scored 1.000, with n_iter of
2. A reading of "chance" there is an artifact of the optimiser never moving,
not evidence that information is absent.

Consequences for how this project's numbers may be read: an unstandardized
probe accuracy near chance on a severely scale-contracted representation is
NOT by itself evidence of information loss, and any earlier claim of ours that
rested on that reading is withdrawn pending recomputation under this protocol.
Neither probe is "the truth": they answer different, protocol-dependent
questions, and the protocol has to be reported with the number.

What this module does about it:

* `C` is selected on a validation split carved out of the TRAIN set, over a
  fixed logarithmic grid, never on test;
* `tol` defaults far below sklearn's, and `max_iter` far above;
* convergence status, iteration count and any warnings are recorded in the
  result and cannot be dropped, alongside `underfit_train` -- failing to beat
  the trivial predictor on the TRAINING set, which is what actually detects the
  silent case, since lbfgs reports two or three iterations rather than zero;
* the feature scale and the selected `C` travel with every score.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MIN_SUPPORT = 10  # per-class test samples needed before a class is scored


@dataclass(frozen=True)
class ProbeConfig:
    """Optimisation and regularisation settings, reported with every score.

    Defaults are deliberately far from sklearn's. `tol` is 1e-8 rather than
    1e-4 because the default is met at the initial point on small-scale
    features, and `max_iter` is 5000 because a tighter tolerance needs more
    steps. The `C` grid spans eight decades so that a scale-contracted
    representation can reach a usable effective regularisation.
    """

    c_grid: tuple[float, ...] = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5)
    tol: float = 1e-8
    max_iter: int = 5000
    solver: str = "lbfgs"
    val_fraction: float = 0.2
    metadata: dict = field(default_factory=dict)


def _fit_recording_convergence(features, labels, C, config, seed):
    """Fit once, capturing what the optimiser actually did.

    Returns (classifier, diagnostics). A caught ConvergenceWarning and a zero
    iteration count are both recorded; neither is raised, because the caller
    needs the number *and* the caveat rather than an exception.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier = LogisticRegression(
            C=C,
            tol=config.tol,
            max_iter=config.max_iter,
            solver=config.solver,
            random_state=seed,
        )
        classifier.fit(features, labels)

    messages = [str(w.message).split(".")[0] for w in caught
                if issubclass(w.category, ConvergenceWarning)]
    iterations = int(np.max(classifier.n_iter_)) if classifier.n_iter_ is not None else -1
    return classifier, {
        "n_iter": float(iterations),
        "converged": float(not messages),
        "warnings": "; ".join(sorted(set(messages))),
    }


def _select_c(features, labels, config, seed):
    """Choose C on a validation split carved from TRAIN. Never touches test.

    Returns (best_C, validation_accuracy). Falls back to the middle of the grid
    when the split cannot be made (too few samples, or a class that would
    vanish), rather than silently selecting on the full set.
    """
    features = np.asarray(features)
    labels = np.asarray(labels)
    fallback = config.c_grid[len(config.c_grid) // 2]
    if len(config.c_grid) == 1:
        return config.c_grid[0], float("nan")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    cut = int(len(features) * (1.0 - config.val_fraction))
    if cut < 2 or len(features) - cut < 1:
        return fallback, float("nan")
    fit_idx, val_idx = order[:cut], order[cut:]
    if len(np.unique(labels[fit_idx])) < 2:
        return fallback, float("nan")

    best_c, best_score = fallback, -1.0
    for candidate in config.c_grid:
        classifier, _ = _fit_recording_convergence(
            features[fit_idx], labels[fit_idx], candidate, config, seed
        )
        score = float((classifier.predict(features[val_idx]) == labels[val_idx]).mean())
        if score > best_score:
            best_c, best_score = candidate, score
    return best_c, best_score


def linear_probe_accuracy(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    seed: int = 0,
    standardize: bool = True,
    config: ProbeConfig | None = None,
) -> float:
    """Held-out accuracy only. Prefer `linear_probe_scores`, which also returns
    the optimisation diagnostics needed to know whether the number means
    anything -- see the module docstring on silent non-convergence.
    """
    return linear_probe_scores(
        train_features,
        train_labels,
        test_features,
        test_labels,
        seed=seed,
        standardize=standardize,
        config=config,
    )["accuracy"]


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
    config: ProbeConfig | None = None,
) -> dict[str, object]:
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
        selected_C        - regularisation chosen on a validation split carved
                            from TRAIN, never on test
        n_iter            - lbfgs iterations actually taken. ZERO means the
                            stopping rule was met at the initial point and the
                            accuracy below is an optimiser artifact
        converged         - 0.0 if a ConvergenceWarning was raised
        underfit_train    - 1.0 when the probe failed to beat the trivial
                            predictor on its own TRAINING set, which together
                            with converged == 1.0 is the silent-failure
                            signature described in the module docstring
        warnings          - the convergence messages, joined
        feature_scale     - mean per-dimension std BEFORE standardization
        train_accuracy    - to expose the underfit case n_iter=0 produces
    """
    config = config or ProbeConfig()
    train_features = np.asarray(train_features, dtype=np.float64)
    test_features = np.asarray(test_features, dtype=np.float64)

    # Recorded before standardization, because it is the raw scale that drives
    # both the regularisation and the stopping-rule coupling.
    raw_scale = float(np.mean(np.std(train_features, axis=0))) if train_features.size else 0.0

    if standardize:
        scaler = StandardScaler().fit(train_features)
        train_features = scaler.transform(train_features)
        test_features = scaler.transform(test_features)

    selected_c, val_score = _select_c(train_features, train_labels, config, seed)
    classifier, optimisation = _fit_recording_convergence(
        train_features, train_labels, selected_c, config, seed
    )
    predictions = classifier.predict(test_features)
    train_accuracy = float((classifier.predict(train_features) == train_labels).mean())

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
        # Protocol metadata. These travel with the score because the score is
        # not interpretable without them: a probe that took no steps reports
        # chance regardless of what the representation contains.
        "train_accuracy": train_accuracy,
        # THE SILENT FAILURE FLAG. lbfgs almost never reports zero iterations --
        # it takes two or three and stops -- so an iteration count cannot detect
        # this on its own. What does detect it: the probe failed to beat the
        # trivial predictor ON ITS OWN TRAINING SET. A correctly optimised probe
        # with any usable information clears that easily, so `underfit_train`
        # together with `converged == 1` is the signature of a stopping rule met
        # before the optimiser moved anywhere useful. Any accuracy reported
        # alongside underfit_train == 1 says nothing about the representation.
        "underfit_train": float(train_accuracy <= majority_rate(train_labels) + 1e-9),
        "val_accuracy": float(val_score),
        "selected_C": float(selected_c),
        "feature_scale": raw_scale,
        "standardized": float(bool(standardize)),
        **optimisation,
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
