"""Investigate C1 evaluator limits on cached vectors; never load an encoder."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from embedding_diagnostics.bdd100k import ATTRIBUTE_VOCAB
from embedding_diagnostics.diagnostics.probe import ProbeConfig, linear_probe_scores
from embedding_diagnostics.embedding_cache import atomic_json, load_cache
from embedding_diagnostics.phase_c_evaluation import json_safe
from embedding_diagnostics.phase_c_pilot import fitted_transform

ROOT = Path('experiments/phaseC_c1_hardening')
PILOT = Path('experiments/phaseC_c1_pilot')


def fixed_fit(train, val, y, labels, c, standardize=False):
    return linear_probe_scores(train, y, val, labels, standardize=standardize,
                               config=ProbeConfig(c_grid=(c,)))


def main():
    original = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in PILOT.iterdir() if p.is_file()}
    matrix, manifest = load_cache(PILOT / 'embedding-cache')
    rows = manifest['selected_rows']
    train = matrix[[r['split'] == 'train' for r in rows]].astype(np.float64)
    val = matrix[[r['split'] == 'val' for r in rows]].astype(np.float64)
    root_identity = {'pilot_cache_manifest_sha256': hashlib.sha256(
        (PILOT / 'embedding-cache/manifest.json').read_bytes()).hexdigest(),
        'pilot_files_sha256': original,
        'source_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
            'scripts/harden_c1_protocol.py',
            'src/embedding_diagnostics/diagnostics/probe.py')},
        'claim_label': '[measurement]', 'neural_inference_calls': 0}
    results = root_identity | {'scale': {}, 'offset': {}, 'rank_four': {}}
    shifted, _ = fitted_transform(train, val, 'mean_injection', .99)
    def condition(x):
        singular = np.linalg.svd(np.column_stack([x, np.ones(len(x))]), compute_uv=False)
        return float((singular[0] / singular[-1])**2)
    results['augmented_design_condition_squared'] = {
        'pristine': condition(train), 'offset': condition(shifted),
        'centered_offset': condition(shifted-shifted.mean(0)),
    }
    with threadpool_limits(limits=4):
        for attr in ATTRIBUTE_VOCAB:
            y = np.array([r[attr] for r in rows if r['split'] == 'train'])
            labels = np.array([r[attr] for r in rows if r['split'] == 'val'])
            baseline = linear_probe_scores(train, y, val, labels, standardize=False)
            # C must scale by 1/a^2 to preserve the penalized linear objective
            # for X'=aX, w'=w/a and the same unpenalized intercept.
            candidates = (baseline['selected_C'], 1e5, 1e6, 1e7, 1e8, 1e9)
            scale = {'baseline': baseline, 'contraction': .01,
                     'objective_equivalent_C': baseline['selected_C'] / .01**2,
                     'fixed_C_fits': []}
            for c in sorted(set(candidates)):
                scale['fixed_C_fits'].append(fixed_fit(train*.01, val*.01, y, labels, c))
            scale['extended_selection'] = linear_probe_scores(
                train*.01, y, val*.01, labels, standardize=False,
                config=ProbeConfig(c_grid=tuple(10.**p for p in range(-2, 10))))
            results['scale'][attr] = scale
            a, b = fitted_transform(train, val, 'mean_injection', .99)
            c = baseline['selected_C']
            centred_train, centred_val = a-a.mean(0), b-a.mean(0)
            offset = {'fixed_C': c, 'raw_offset': fixed_fit(a,b,y,labels,c),
                      'centered_only_offset': fixed_fit(centred_train,centred_val,y,labels,c),
                      'centered_only_pristine': fixed_fit(train-train.mean(0),val-train.mean(0),
                                                        y,labels,c),
                      'standardized_pristine': linear_probe_scores(train,y,val,labels),
                      'standardized_offset': linear_probe_scores(a,y,b,labels),
                      'centered_feature_max_difference': float(np.abs(
                          centred_train-(train-train.mean(0))).max())}
            results['offset'][attr] = offset
            if attr == 'weather':
                a,b = fitted_transform(train,val,'rank_truncation',.99)
                # Work in the retained orthonormal coordinates: same L2
                # penalty/objective as the 768d projected raw vectors.
                _,_,vt = np.linalg.svd(train,full_matrices=False)
                coords_train,coords_val = train@vt[:4].T, val@vt[:4].T
                results['rank_four'][attr] = {
                    'projected_raw_selection': linear_probe_scores(a,y,b,labels,standardize=False),
                    'projected_standardized_selection': linear_probe_scores(a,y,b,labels),
                    'orthonormal_coordinate_fixed_C': [fixed_fit(
                        coords_train,coords_val,y,labels,c) for c in (.01,1.,100.,1e4,1e6)],
                    'full_dimension_fixed_C': fixed_fit(train,val,y,labels,100.),
                    'geometry_dimension': 4}
            atomic_json(ROOT / 'investigation.json', json_safe(results))
            print(f'investigated {attr}', flush=True)
    if {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in PILOT.iterdir()
        if p.is_file()} != original:
        raise ValueError('historical pilot artifacts changed')
    atomic_json(ROOT / 'pilot_preservation.json', original)


if __name__ == '__main__':
    main()
