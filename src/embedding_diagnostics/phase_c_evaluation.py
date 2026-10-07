"""Held-out C1 pilot endpoints, using existing diagnostics without thresholds.

Participation ratio is the centered covariance eigenvalue ratio; RankMe is
raw-matrix spectral entropy. This is exploratory instrument validation.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .bdd100k import ATTRIBUTE_VOCAB
from .diagnostics.metrics import collapse_metrics
from .diagnostics.probe import MIN_SUPPORT, ProbeConfig, linear_probe_scores, retrieval_chance
from .diagnostics.retrieval import retrieval_macro_precision_at_k, retrieval_precision_at_k
from .phase_c_pilot import support


def json_safe(value):
    """Unsupported numeric scores remain explicit nulls, never omitted."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def evaluate_vectors(train, val, rows):
    """Fit only train; score physical validation. Retrieval uses val alone."""
    train_rows = [r for r in rows if r['split'] == 'train']
    val_rows = [r for r in rows if r['split'] == 'val']
    if len(train_rows) != len(train) or len(val_rows) != len(val):
        raise ValueError('matrix/split alignment mismatch')
    if not np.isfinite(train).all() or not np.isfinite(val).all():
        raise ValueError('nonfinite diagnostic input')
    config = ProbeConfig()
    counts = support(rows)
    result = {'claim_label': '[measurement]', 'status': 'PILOT / EXPLORATORY',
              'geometry': {'train': collapse_metrics(train), 'val': collapse_metrics(val)},
              'threshold_verdicts': None, 'probe_config': asdict(config), 'attributes': {}}
    for attr, vocab in ATTRIBUTE_VOCAB.items():
        train_labels = np.array([r[attr] for r in train_rows])
        val_labels = np.array([r[attr] for r in val_rows])
        scores = {}
        for standardize in (True, False):
            scores['probe_standardized' if standardize else 'probe_unscaled'] = linear_probe_scores(
                train, train_labels, val, val_labels, seed=0, standardize=standardize,
                min_support=MIN_SUPPORT, config=config)
        macro, per_class = retrieval_macro_precision_at_k(val, val_labels, k=10,
                                                         min_support=MIN_SUPPORT)
        train_majority = int(np.argmax(np.bincount(train_labels, minlength=len(vocab))))
        scored = [name for name in vocab if counts['val'][attr][name] >= MIN_SUPPORT]
        scores.update({
            'support': {split: counts[split][attr] for split in ('train', 'val')},
            'macro_scored_classes': scored,
            'macro_excluded_classes': [name for name in vocab if name not in scored],
            'classes_absent_from_train': [name for name in vocab
                                          if counts['train'][attr][name] == 0],
            'retrieval_p10': retrieval_precision_at_k(val, val_labels, k=10),
            'retrieval_macro_p10': macro,
            'retrieval_per_class_p10': {vocab[k]: v for k, v in per_class.items()},
            'floors': {
                'validation_majority': max(counts['val'][attr].values()) / len(val),
                'train_majority_predictor_validation_accuracy':
                    float((val_labels == train_majority).mean()),
                'retrieval_chance': retrieval_chance(val_labels),
                'retrieval_chance_convention': 'Phase B sum of squared validation frequencies',
                'uniform_random_probe_accuracy': 1 / len(vocab),
                'uniform_random_probe_convention': 'uniform over full canonical vocabulary',
            },
        })
        result['attributes'][attr] = scores
    return json_safe(result)
