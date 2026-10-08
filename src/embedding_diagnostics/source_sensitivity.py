"""Post-audit exclusion sensitivity, delegating to accepted Phase-C instruments.

This is not an independent replication or a new metric. It preserves the
participation-ratio covariance definition and raw RankMe from existing metrics.
All inference and original-source access are outside this module's scope.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .bdd100k import ATTRIBUTE_VOCAB
from .diagnostics import probe
from .phase_c_pilot import support

ATTRIBUTES = tuple(ATTRIBUTE_VOCAB)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def array_digest(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if not Path(name).is_file() or digest(name) != expected:
            raise ValueError(f"protected file changed: {name}")


def exclude_rows(matrix, sample, exclusion_ids):
    """Exact ID exclusion; never interpret a prefix or mutate historical rows."""
    rows = sample["selected_rows"]
    ids = [r["image_id"] for r in rows]
    exclusions = set(exclusion_ids)
    if (
        len(matrix) != len(rows)
        or len(set(ids)) != len(ids)
        or len(exclusions) != len(exclusion_ids)
        or not exclusions <= set(ids)
    ):
        raise ValueError("invalid exclusion IDs or row alignment")
    if any(r["split"] != "val" for r in rows if r["image_id"] in exclusions):
        raise ValueError("exclusions must be physical validation rows")
    if {r["split"] for r in rows} != {"train", "val"}:
        raise ValueError("invalid physical split")
    train_mask = np.array([r["split"] == "train" for r in rows])
    val_rows = [r for r in rows if r["split"] == "val"]
    keep = np.array([r["image_id"] not in exclusions for r in val_rows])
    reduced = copy.deepcopy(sample)
    reduced["selected_rows"] = [
        r for r in reduced["selected_rows"] if r["image_id"] not in exclusions
    ]
    reduced["support"] = support(reduced["selected_rows"])
    return matrix[train_mask], matrix[~train_mask][keep], reduced, keep


def class_scores(labels, predictions, class_count, eligible):
    """Explicit zero-support rows are unscorable, not zero recall or new eligibility."""
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    result = {}
    for cls in range(class_count):
        chosen = labels == cls
        n = int(chosen.sum())
        tp = int(np.sum(chosen & (predictions == cls)))
        predicted = int(np.sum(predictions == cls))
        precision = tp / predicted if predicted else 0.0
        recall = tp / n if n else None
        f1 = (
            (2 * precision * recall / (precision + recall) if n and precision + recall else 0.0)
            if n
            else None
        )
        result[str(cls)] = {
            "support": n,
            "eligible": cls in eligible,
            "scorable": bool(n and cls in eligible),
            "recall": recall,
            "f1": f1,
        }
    return result


def refit_probe(
    train,
    y,
    val,
    labels,
    original,
    *,
    seed=0,
    standardize=True,
    min_support=10,
    config=None,
    eligible_classes=None,
    class_count=None,
):
    """Reuse recorded train-only C selection; refit the unchanged final estimator.

    No saved models exist. Candidate diagnostics/C are reusable training-fitted
    parameters: scaler/final LBFGS fitting and scoring delegate to the existing
    implementation under its full original config, rather than a fixed-C config.
    The runner binds original condition/training bytes, selection and source code.
    """
    config = config or probe.ProbeConfig()
    selected = float(original["selected_C"])
    if selected not in config.c_grid:
        raise ValueError("historical selected C outside original grid")
    y, labels = np.asarray(y), np.asarray(labels)
    captured = []
    fit = probe._fit_recording_convergence

    def selection(features, train_labels, requested, requested_seed):
        if requested != config or requested_seed != seed or not np.array_equal(train_labels, y):
            raise ValueError("historical selection configuration mismatch")
        return selected, original["inner_validation_accuracy"], original["selection_fits"]

    def final_fit(features, train_labels, c, requested, requested_seed):
        if c != selected or requested != config or requested_seed != seed:
            raise ValueError("final fitting configuration mismatch")
        estimator, diagnostics = fit(features, train_labels, c, requested, requested_seed)
        captured.append(estimator)
        return estimator, diagnostics

    with (
        patch.object(probe, "_select_c", selection),
        patch.object(probe, "_fit_recording_convergence", final_fit),
    ):
        scores = probe.linear_probe_scores(
            train,
            y,
            val,
            labels,
            seed=seed,
            standardize=standardize,
            min_support=min_support,
            config=config,
            eligible_classes=eligible_classes,
        )
    if len(captured) != 1:
        raise ValueError("expected exactly one final fit")
    x = np.asarray(val, dtype=np.float64)
    if standardize:
        x = probe.StandardScaler().fit(np.asarray(train, dtype=np.float64)).transform(x)
    predictions = captured[0].predict(x)
    eligible = tuple(eligible_classes) if eligible_classes is not None else tuple(np.unique(labels))
    count = class_count if class_count is not None else int(max(y.max(), labels.max())) + 1
    scores["validation_per_class"] = class_scores(labels, predictions, count, eligible)
    scores["fixed_eligible_class_codes"] = list(eligible)
    scores["eligible_zero_support_classes"] = [c for c in eligible if not np.any(labels == c)]
    scores["validation_support"] = {str(c): int(np.sum(labels == c)) for c in range(count)}
    scores["refit_provenance"] = {
        "saved_classifier_available": False,
        "training_selection_reused": True,
        "final_estimator_refitted": True,
        "selection_source": "accepted train-only candidate log",
        "train_feature_sha256": array_digest(np.asarray(train, dtype=np.float64)),
        "train_label_sha256": array_digest(y),
        "historical_train_accuracy": original.get("train_accuracy"),
        "train_accuracy_difference": scores["train_accuracy"]
        - original.get("train_accuracy", scores["train_accuracy"]),
        "historical_iterations": original.get("n_iter"),
        "historical_converged": original.get("converged"),
    }
    return scores


def gallery_scores(full_val, rows, query_indices, *, k=10):
    """Supplement: retained queries against old gallery using inherited arithmetic.

    This delegates the same normalization/dot/argpartition operations as the
    accepted C1/C2 retrieval. It is a rectangular query/gallery diagnostic, not
    a new retrieval condition. Self is excluded using original row positions.
    """
    x = np.asarray(full_val, dtype=np.float64)
    indices = np.asarray(query_indices)
    if (
        x.ndim != 2
        or len(x) != len(rows)
        or indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or len(set(indices)) != len(indices)
        or np.any(indices < 0)
        or np.any(indices >= len(rows))
        or len({r["image_id"] for r in rows}) != len(rows)
        or not 0 < k < len(rows)
    ):
        raise ValueError("query/gallery row alignment mismatch")
    normalized = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    # Original full shape preserves the historical arithmetic, then select queries.
    scores = (normalized @ normalized.T)[indices]
    scores[np.arange(len(indices)), indices] = -np.inf
    neighbours = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    boundary = np.sort(scores, axis=1)[:, -(k + 1) :]
    output = {
        "query_ids": [rows[i]["image_id"] for i in indices],
        "query_count": len(indices),
        "gallery_count": len(rows),
        "k": k,
        "tie_rule": "inherited NumPy argpartition; no new tie ordering",
        "boundary_tie_count": int(np.sum(boundary[:, 0] == boundary[:, 1])),
        "neighbour_ids": [[rows[j]["image_id"] for j in a] for a in neighbours],
        "attributes": {},
    }
    for attr in ATTRIBUTES:
        labels = np.array([r[attr] for r in rows])
        hits = (labels[neighbours] == labels[indices, None]).mean(axis=1)
        output["attributes"][attr] = {
            "p10": float(hits.mean()),
            "per_query_p10": hits.tolist(),
            "chance_self_excluded": float(
                np.mean([(np.sum(labels == labels[i]) - 1) / (len(rows) - 1) for i in indices])
            ),
            "gallery_chance_sum_p_squared": probe.retrieval_chance(labels),
        }
    return output


def add_support_context(record, sample):
    """All vocabulary supports, including zero classes; eligibility never recomputed."""
    for attr, vocab in ATTRIBUTE_VOCAB.items():
        counts = sample["support"]["val"][attr]
        scores = record["attributes"][attr]
        scores["class_support"] = {
            name: {
                "train": sample["support"]["train"][attr][name],
                "val": counts[name],
                "balanced_probe_eligible": i in sample["eligible_probe_class_codes"][attr],
                "scorable": counts[name] > 0,
            }
            for i, name in enumerate(vocab)
        }
        scores["zero_support_classes"] = [name for name, n in counts.items() if n == 0]
        scores["validation_majority_floor"] = max(counts.values()) / sum(counts.values())
        scores["balanced_constant_class_floor"] = 1 / len(
            sample["eligible_probe_class_codes"][attr]
        )
        if "per_class" in scores:  # C3 normally omits absent classes; disclose explicitly.
            for i, name in enumerate(vocab):
                scores["per_class"].setdefault(
                    str(i), {"support": 0, "p10": None, "delta": None, "scorable": False}
                )


def rebind_fit_history(scores, original, *, reused):
    """Reuse an identical fit score while comparing each condition to its own record."""
    scores = copy.deepcopy(scores)
    provenance = scores["refit_provenance"]
    provenance.update(
        historical_train_accuracy=original.get("train_accuracy"),
        train_accuracy_difference=scores["train_accuracy"]
        - original.get("train_accuracy", scores["train_accuracy"]),
        historical_iterations=original.get("n_iter"),
        historical_converged=original.get("converged"),
        reused_identical_fit_score=reused,
    )
    return scores
