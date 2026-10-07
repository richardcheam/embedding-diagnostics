"""Exploratory external-reference pilot using existing A/B instruments.

No representation-family health thresholds are applied. Participation ratio
uses centered covariance eigenvalues; RankMe uses raw singular values. Probe
fitting and data-dependent interventions use training data exclusively.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .bdd100k import ATTRIBUTE_VOCAB, _encode
from .degradation import TRANSFORMS

PROTOCOL = {
    'seed': 0, 'train_count': 384, 'val_count': 128,
    'selection': 'uniform without replacement from fully labelled, image-ID-sorted rows; '
                 'independent NumPy PCG64 SeedSequence([seed, split_index]), train=0, val=1; '
                 'global image-ID ordering; no quotas or coverage intervention',
    'min_scoring_support': 10,
    'status': 'PILOT / EXPLORATORY',
}


def census_rows(rows: list[dict]) -> dict:
    """Raw metadata counts, including unknowns and the semantic eligible pool."""
    result = {'projected_columns': ['image_id', 'split', *ATTRIBUTE_VOCAB], 'splits': {}}
    for split in sorted({r['split'] for r in rows}):
        subset = [r for r in rows if r['split'] == split]
        eligible = [r for r in subset if all(v >= 0 for v in _encode(r).values())]
        entry = {'total_rows': len(subset), 'fully_labelled_rows': len(eligible),
                 'counts': {}, 'fully_labelled_counts': {}, 'unknown': {}, 'rare_classes': {}}
        for attr, vocab in ATTRIBUTE_VOCAB.items():
            entry['counts'][attr] = {v: sum(r.get(attr) == v for r in subset) for v in vocab}
            entry['fully_labelled_counts'][attr] = {
                v: sum(r.get(attr) == v for r in eligible) for v in vocab}
            entry['unknown'][attr] = dict(sorted(Counter(
                '<null>' if r.get(attr) is None else str(r.get(attr))
                for r in subset if r.get(attr) not in vocab).items()))
            entry['rare_classes'][attr] = {
                v: n for v, n in entry['fully_labelled_counts'][attr].items() if n < 100}
        entry['joint_counts'] = [
            dict(zip(ATTRIBUTE_VOCAB, labels, strict=True), count=n)
            for labels, n in sorted(Counter(tuple(
                '<null>' if r.get(a) is None else str(r.get(a)) for a in ATTRIBUTE_VOCAB)
                for r in subset).items())]
        entry['fully_labelled_joint_counts'] = [
            dict(zip(ATTRIBUTE_VOCAB, labels, strict=True), count=n)
            for labels, n in sorted(Counter(tuple(r[a] for a in ATTRIBUTE_VOCAB)
                                           for r in eligible).items())]
        result['splits'][split] = entry
    return result


def select_sample(rows: list[dict], *, seed=0, train_count=384, val_count=128) -> list[dict]:
    """Uniform independent physical-split samples; no labels used for balancing."""
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate image IDs')
    chosen = []
    for index, (split, count) in enumerate((('train', train_count), ('val', val_count))):
        eligible = sorted((dict(image_id=r['image_id'], split=split, **_encode(r))
                           for r in rows if r['split'] == split
                           and all(v >= 0 for v in _encode(r).values())),
                          key=lambda r: r['image_id'])
        if count < 1 or count > len(eligible):
            raise ValueError(f'insufficient eligible {split} rows')
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        chosen.extend(eligible[i] for i in rng.choice(len(eligible), count, replace=False))
    return sorted(chosen, key=lambda r: r['image_id'])


def support(rows: list[dict]) -> dict:
    return {split: {attr: {name: sum(r['split'] == split and r[attr] == index for r in rows)
                          for index, name in enumerate(vocab)}
                    for attr, vocab in ATTRIBUTE_VOCAB.items()} for split in ('train', 'val')}


def fitted_transform(train, val, name, severity, seed=0):
    """Existing interventions with scales/means/SVD fitted on train only.

    Zero severity is identity for both splits, including held-out directions
    outside the training row span. Positive rank severities retain the existing
    round(min(train.shape)*(1-severity)) components of the raw training SVD.
    Noise draws for validation continue the training draw, at training scale.
    """
    a, b = np.asarray(train, dtype=np.float64), np.asarray(val, dtype=np.float64)
    if name not in TRANSFORMS or not 0 <= severity < 1:
        raise ValueError('invalid intervention')
    if severity == 0:
        return a.copy(), b.copy()
    transformed = TRANSFORMS[name](a, severity, seed)
    if name == 'scale_contraction':
        return transformed, TRANSFORMS[name](b, severity, seed)
    if name == 'rank_truncation':
        _, _, vt = np.linalg.svd(a, full_matrices=False)
        basis = vt[:max(1, int(round(min(a.shape) * (1 - severity))))]
        return transformed, (b @ basis.T) @ basis
    if name == 'mean_injection':
        return transformed, b + (transformed[0] - a[0])
    if name == 'mean_interpolation':
        return transformed, b * (1 - severity) + a.mean(0) * severity
    rng = np.random.default_rng(seed)
    noise = rng.normal(scale=a.std() * severity / max(1 - severity, 1e-6),
                       size=(len(a) + len(b), a.shape[1]))
    return transformed, b + noise[len(a):]
