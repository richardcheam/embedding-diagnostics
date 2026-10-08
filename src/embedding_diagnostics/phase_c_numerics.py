"""Bounded cache-only numerical instrument study, not a new quantizer/metric.

Score-aware error has established precedent: Guo et al., ICML 2020,
https://proceedings.mlr.press/v119/guo20h.html. Our scalar quantizer is a simple
training-calibrated control, not a reimplementation of AVQ. Margin inequalities
are elementary sufficient order conditions, checked empirically without rigorous
floating-point certification. Tests validate wiring, not research claims.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embedding_cache import atomic_json
from .phase_c_pilot import fitted_transform

ATTRIBUTES = ('weather', 'scene', 'timeofday')
CONDITIONS = ('native_ref', 'native_fp16_gallery', 'native_fp16_both',
              'native_int8_gallery', 'native_int8_both', 'native_fp32_arithmetic',
              'mean99_ref', 'mean99_fp16_gallery')


def finite_matrix(x, dtype=np.float64):
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError('expected finite nonempty matrix')
    return x


def normalize(x, dtype):
    """Explicit multiplication, reduction, sqrt and division in the chosen dtype."""
    if np.dtype(dtype) not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('analysis normalization must be FP32 or FP64')
    x = finite_matrix(x, dtype)
    squared = np.multiply(x, x, dtype=dtype)
    lengths = np.sqrt(np.sum(squared, axis=1, keepdims=True, dtype=dtype), dtype=dtype)
    if np.any(lengths <= 0) or not np.isfinite(lengths).all():
        raise ValueError('zero/nonfinite normalization norm')
    return np.divide(x, lengths, dtype=dtype)


def cosine_scores(q, g):
    if q.dtype != g.dtype or q.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('scoring inputs must share explicit FP32/FP64 dtype')
    scores = q @ g.T
    if scores.dtype != q.dtype or not np.isfinite(scores).all():
        raise ValueError('invalid scoring dtype/values')
    return scores


def arithmetic_audit():
    """Empirical reduction witness, not proof of every BLAS internal operation."""
    a = np.ones((2, 768), dtype=np.float32)
    a[:, 0], a[:, -1] = 1e8, -1e8
    b = np.ones_like(a)
    s32 = cosine_scores(a, b)
    s64 = cosine_scores(a.astype(np.float64), b.astype(np.float64))
    if s32[0, 0] == s64[0, 0]:
        raise ValueError('FP32 accumulation witness did not distinguish arithmetic')
    return {'normalization': ['float32', 'float64'], 'score_output': ['float32', 'float64'],
            'fp32_accumulation_witness': float(s32[0, 0]),
            'fp64_accumulation_witness': float(s64[0, 0]),
            'verification': 'explicit FP32 reductions and BLAS FP32 dot; cancellation witness; '
                            'no rigorous floating-point certification'}


@dataclass(frozen=True)
class TrainInt8:
    scales: np.ndarray

    @classmethod
    def fit(cls, train):
        train = finite_matrix(train)
        maxima = np.max(np.abs(train), axis=0)
        return cls(np.where(maxima == 0, 1., maxima/127.))

    def encode(self, x):
        x = finite_matrix(x)
        if x.shape[1] != len(self.scales):
            raise ValueError('quantizer dimension mismatch')
        scaled = x/self.scales
        rounded = np.rint(scaled)
        clipped = np.abs(rounded) > 127
        return np.clip(rounded, -127, 127).astype(np.int8), {
            'outside_calibration_coordinates': int(np.sum(np.abs(scaled) > 127)),
            'per_row_clipped_coordinates': np.sum(clipped, axis=1).tolist(),
            'per_row_outside_calibration_coordinates': np.sum(np.abs(scaled) > 127,
                                                              axis=1).tolist(),
            'clipped_coordinates': int(clipped.sum()),
            'clipped_rows': int(np.any(clipped, axis=1).sum()),
            'coordinate_count': int(x.size)}

    def decode(self, packed):
        if packed.dtype != np.int8 or packed.ndim != 2 or packed.shape[1] != len(self.scales):
            raise ValueError('invalid packed INT8 matrix')
        return packed.astype(np.float64)*self.scales


def neighbours(scores, ids, k=10):
    ids = np.asarray(ids, dtype=str)
    n = len(ids)
    if len(set(ids)) != n or scores.shape != (n, n) or not 0 < k < n-1:
        raise ValueError('IDs/scores/k alignment mismatch or duplicate IDs')
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite similarity')
    masked = scores.copy()
    np.fill_diagonal(masked, -np.inf)
    id_order = np.argsort(ids, kind='stable')
    order = id_order[np.argsort(-masked[:, id_order], axis=1, kind='stable')]
    selected = order[:, :k]
    ranked = np.take_along_axis(masked, order[:, :k+1], axis=1)
    margin = (ranked[:, k-1].astype(np.float64)-ranked[:, k].astype(np.float64))
    return selected, margin, margin == 0


def margin_bins(margins):
    margins = np.asarray(margins, dtype=np.float64)
    if margins.ndim != 1 or not len(margins) or not np.isfinite(margins).all():
        raise ValueError('invalid reference margins')
    cuts = np.unique(np.quantile(margins, [.25, .5, .75], method='inverted_cdf'))
    cuts = cuts[cuts < margins.max()]
    assignments = np.searchsorted(cuts, margins, side='left')
    groups = [{'bin': int(b), 'count': int(np.sum(assignments == b)),
               'minimum': float(margins[assignments == b].min()),
               'maximum': float(margins[assignments == b].max())}
              for b in np.unique(assignments)]
    return assignments, groups


def margin_checks(margins, reconstruction_bound, observed_error):
    return (np.asarray(margins) > 2*np.asarray(reconstruction_bound),
            np.asarray(margins) > 2*np.asarray(observed_error))


def attribute_turnover(reference, changed, gallery_labels, query_labels):
    reference, changed = np.asarray(reference), np.asarray(changed)
    gallery_labels, query_labels = np.asarray(gallery_labels), np.asarray(query_labels)
    if (reference.shape != changed.shape or reference.ndim != 2 or
        len(query_labels) != len(reference) or min(reference.min(), changed.min()) < 0 or
        max(reference.max(), changed.max()) >= len(gallery_labels)):
        raise ValueError('neighbour/label alignment mismatch')
    entering, departing, same = [], [], []
    for before, after in zip(reference, changed, strict=True):
        ins = Counter(str(x) for x in gallery_labels[sorted(set(after)-set(before))])
        outs = Counter(str(x) for x in gallery_labels[sorted(set(before)-set(after))])
        entering.append(dict(sorted(ins.items())))
        departing.append(dict(sorted(outs.items())))
        same.append(ins == outs)
    before_hits = np.sum(gallery_labels[reference] == query_labels[:, None], axis=1)
    after_hits = np.sum(gallery_labels[changed] == query_labels[:, None], axis=1)
    deltas = np.array([ins.get(str(label), 0)-outs.get(str(label), 0)
                       for ins, outs, label in zip(entering, departing, query_labels, strict=True)])
    if not np.array_equal(deltas, after_hits-before_hits):
        raise ValueError('attribute entering/departing count invariant failed')
    return {'entering_counts': entering, 'departing_counts': departing,
            'label_multiset_unchanged': same,
            'p10_reference': (before_hits/reference.shape[1]).tolist(),
            'p10': (after_hits/reference.shape[1]).tolist(),
            'p10_delta': (deltas/reference.shape[1]).tolist(),
            'relevant_count_delta': deltas.tolist()}


def split_rows(matrix, manifest, sample):
    rows = sample['selected_rows']
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate source image IDs')
    if manifest['selected_rows'] != rows or len(matrix) != len(rows):
        raise ValueError('source row alignment mismatch')
    if {r['split'] for r in rows} != {'train', 'val'}:
        raise ValueError('invalid physical splits')
    train_mask = np.array([r['split'] == 'train' for r in rows])
    return matrix[train_mask], matrix[~train_mask], [r for r in rows if r['split'] == 'val']


def mean_stress(train, val):
    return fitted_transform(train, val, 'mean_injection', .99, seed=0)


def payload_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def resume_results(path, identity):
    path = Path(path)
    state = json.loads(path.read_text()) if path.exists() else {'identity': identity, 'records': []}
    if state['identity'] != identity:
        raise ValueError('C3 resume provenance mismatch')
    names = []
    for envelope in state['records']:
        if payload_hash(envelope['payload']) != envelope['sha256']:
            raise ValueError('C3 record checksum mismatch')
        names.append(envelope['payload']['condition'])
    if len(set(names)) != len(names):
        raise ValueError('duplicate resumed conditions')
    return state


def append_record(path, state, record):
    state['records'].append({'payload': record, 'sha256': payload_hash(record)})
    atomic_json(path, state)


def run_conditions(path, identity, conditions, evaluate):
    state = resume_results(path, identity)
    completed = [r['payload']['condition'] for r in state['records']]
    if completed != list(conditions[:len(completed)]):
        raise ValueError('C3 resumed condition order mismatch')
    calls = 0
    for name in conditions[len(completed):]:
        record = evaluate(name)
        if record['condition'] != name:
            raise ValueError('evaluated condition alignment mismatch')
        append_record(path, state, record)
        calls += 1
    return state, calls


def evaluate_condition(name, raw, reference, rows, quantizer, *, k=10):
    """Exact full-gallery scoring; raw stress coordinates are rounded before L2."""
    if name not in CONDITIONS:
        raise ValueError('unknown frozen condition')
    raw = finite_matrix(raw)
    ref = normalize(reference, np.float64)
    scores_ref = cosine_scores(ref, ref)
    ids = [r['image_id'] for r in rows]
    expected, margins, reference_ties = neighbours(scores_ref, ids, k)
    bins, bin_definitions = margin_bins(margins)
    q, g = normalize(raw, np.float64), normalize(raw, np.float64)
    clipping = {'clipped_coordinates': 0, 'clipped_rows': 0, 'coordinate_count': raw.size,
                'outside_calibration_coordinates': 0,
                'per_row_clipped_coordinates': [0]*len(raw),
                'per_row_outside_calibration_coordinates': [0]*len(raw)}
    packed_bytes = raw.size*4 if name.startswith('native') else raw.size*8
    raw_reconstruction = raw.copy()
    if 'fp16' in name:
        packed = raw.astype(np.float16)
        raw_reconstruction = finite_matrix(packed)
        g = normalize(raw_reconstruction, np.float64)
        packed_bytes = packed.nbytes
    elif 'int8' in name:
        packed, clipping = quantizer.encode(raw)
        raw_reconstruction = quantizer.decode(packed)
        g = normalize(raw_reconstruction, np.float64)
        packed_bytes = packed.nbytes + quantizer.scales.nbytes
    if name.endswith('_both'):
        q = g.copy()
    elif name.endswith('_arithmetic'):
        q = g = normalize(raw, np.float32)
    scores = cosine_scores(q, g)
    actual, _, boundary_ties = neighbours(scores, ids, k)
    diagonal = np.eye(len(rows), dtype=bool)
    error = np.max(np.where(diagonal, 0., np.abs(scores.astype(np.float64)-scores_ref)), axis=1)
    arithmetic_error = np.max(np.where(diagonal, 0., np.abs(
        scores.astype(np.float64)-cosine_scores(q.astype(np.float64), g.astype(np.float64)))),
        axis=1)
    q_error = np.linalg.norm(q.astype(np.float64)-ref, axis=1)
    g_error = np.linalg.norm(g.astype(np.float64)-ref, axis=1)
    bound = q_error+g_error.max()
    prospective, retrospective = margin_checks(margins, bound, error)
    applicable = not name.endswith('_arithmetic')
    overlap = np.array([len(set(a) & set(b))/k for a, b in zip(expected, actual, strict=True)])
    if np.any(retrospective & (overlap < 1)):
        raise ValueError('observed-error sufficient condition violated; investigate instrument')
    # Prospective checks are empirical floating-point checks, never certifications.
    record = {'condition': name, 'query_ids': ids, 'dimension': raw.shape[1],
              'normalization_dtype': str(q.dtype), 'gallery_normalization_dtype': str(g.dtype),
              'score_dtype': str(scores.dtype), 'gallery_storage_bytes': int(packed_bytes),
              'clipping': clipping, 'reference_boundary_tie_count': int(reference_ties.sum()),
              'boundary_tie_count': int(boundary_ties.sum()), 'margin_bins': bin_definitions,
              'summary': {'mean_overlap': float(overlap.mean()),
                          'identity_changed_queries': int(np.sum(overlap < 1)),
                          'mean_max_score_error': float(error.mean()),
                          'max_score_error': float(error.max()),
                          'max_arithmetic_residual': float(arithmetic_error.max()),
                          'prospective_sufficient_queries': int(prospective.sum()) if applicable
                              else None,
                          'prospective_empirical_violations': int(np.sum(
                              prospective & (overlap < 1)))
                              if applicable else None,
                          'retrospective_sufficient_queries': int(retrospective.sum())},
              'per_query': {'margin': margins.tolist(), 'margin_bin': bins.tolist(),
                            'gallery_clipped_coordinates':
                                clipping['per_row_clipped_coordinates'],
                            'gallery_outside_calibration_coordinates':
                                clipping['per_row_outside_calibration_coordinates'],
                            'overlap': overlap.tolist(), 'max_score_error': error.tolist(),
                            'query_reconstruction_error': q_error.tolist(),
                            'gallery_reconstruction_error': g_error.tolist(),
                            'raw_storage_error': np.linalg.norm(raw_reconstruction-raw,
                                                               axis=1).tolist(),
                            'raw_norm': np.linalg.norm(raw, axis=1).tolist(),
                            'prospective_bound': bound.tolist() if applicable else None,
                            'prospective_sufficient': prospective.tolist() if applicable else None,
                            'retrospective_sufficient': retrospective.tolist(),
                            'reference_neighbour_ids': [[ids[j] for j in a] for a in expected],
                            'neighbour_ids': [[ids[j] for j in a] for a in actual]},
              'attributes': {}, 'bin_analysis': []}
    for attr in ATTRIBUTES:
        labels = np.array([r[attr] for r in rows])
        turnover = attribute_turnover(expected, actual, labels, labels)
        p10, delta = np.array(turnover['p10']), np.array(turnover['p10_delta'])
        counts = Counter(labels)
        n = len(labels)
        per_class = {str(c): {'support': count, 'p10': float(p10[labels == c].mean()),
                             'delta': float(delta[labels == c].mean())}
                     for c, count in sorted(counts.items())}
        macro = [v for v in per_class.values() if v['support'] >= 10]
        record['attributes'][attr] = {'p10': float(p10.mean()),
            'reference_p10': float(np.mean(turnover['p10_reference'])),
            'delta': float(delta.mean()), 'gained_queries': int(np.sum(delta > 0)),
            'lost_queries': int(np.sum(delta < 0)), 'unchanged_queries': int(np.sum(delta == 0)),
            'identity_changed_p10_unchanged': int(np.sum((overlap < 1) & (delta == 0))),
            'identity_changed_label_counts_unchanged': int(np.sum((overlap < 1) &
                                                        turnover['label_multiset_unchanged'])),
            'chance_sum_p_squared': sum((c/n)**2 for c in counts.values()),
            'chance_self_excluded': sum(c*(c-1) for c in counts.values())/(n*(n-1)),
            'macro_p10_support10': float(np.mean([v['p10'] for v in macro])) if macro else None,
            'macro_delta_support10': float(np.mean([v['delta'] for v in macro])) if macro else None,
            'per_class': per_class, 'per_query': turnover}
    for definition in bin_definitions:
        chosen = bins == definition['bin']
        record['bin_analysis'].append(definition | {
            'mean_overlap': float(overlap[chosen].mean()),
            'changed_queries': int(np.sum(overlap[chosen] < 1)),
            'mean_max_score_error': float(error[chosen].mean()),
            'prospective_sufficient': int(prospective[chosen].sum()) if applicable else None,
            'retrospective_sufficient': int(retrospective[chosen].sum()),
            'p10_deltas': {a: float(np.mean(np.array(
                record['attributes'][a]['per_query']['p10_delta'])[chosen])) for a in ATTRIBUTES}})
    return record
