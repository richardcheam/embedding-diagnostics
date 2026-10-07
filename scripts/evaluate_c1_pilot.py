"""Evaluate the frozen C1-pilot cache; checkpoint each completed stress row."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from embedding_diagnostics.degradation import DEFAULT_SEVERITIES, TRANSFORMS
from embedding_diagnostics.embedding_cache import atomic_json, load_cache
from embedding_diagnostics.phase_c_evaluation import evaluate_vectors
from embedding_diagnostics.phase_c_pilot import fitted_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiments/phaseC_c1_pilot'))
    args = parser.parse_args()
    matrix, manifest = load_cache(args.root / 'embedding-cache')
    sample = json.loads((args.root / 'sample_manifest.json').read_text())
    if manifest['selected_rows'] != sample['selected_rows']:
        raise ValueError('cache differs from frozen sample')
    rows = manifest['selected_rows']
    train = matrix[[r['split'] == 'train' for r in rows]]
    val = matrix[[r['split'] == 'val' for r in rows]]
    identity = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        'src/embedding_diagnostics/phase_c_evaluation.py',
        'src/embedding_diagnostics/phase_c_pilot.py', 'scripts/evaluate_c1_pilot.py',
        'src/embedding_diagnostics/diagnostics/probe.py',
        'src/embedding_diagnostics/diagnostics/retrieval.py',
        'src/embedding_diagnostics/diagnostics/metrics.py',
        'src/embedding_diagnostics/degradation.py')}
    identity['cache_manifest'] = hashlib.sha256(
        (args.root / 'embedding-cache' / 'manifest.json').read_bytes()).hexdigest()
    identity['runtime_versions'] = {name: importlib.metadata.version(name)
                                    for name in ('numpy', 'scipy', 'scikit-learn', 'threadpoolctl')}
    identity['blas_thread_limit'] = 4
    result_path = args.root / 'stress_results.json'
    if result_path.exists():
        results = json.loads(result_path.read_text())
        if results['identity'] != identity:
            raise ValueError('analysis provenance mismatch on resume')
    else:
        results = {'identity': identity, 'status': 'PILOT / EXPLORATORY', 'records': []}
    with threadpool_limits(limits=4):
        pristine_path = args.root / 'pristine_metrics.json'
        if pristine_path.exists():
            pristine = json.loads(pristine_path.read_text())
            if pristine['identity'] != identity:
                raise ValueError('pristine analysis provenance mismatch')
        else:
            started = time.perf_counter()
            pristine = evaluate_vectors(train, val, rows)
            pristine.update(identity=identity, wall_seconds=time.perf_counter() - started)
            atomic_json(pristine_path, pristine)
        for name in TRANSFORMS:
            for severity in DEFAULT_SEVERITIES:
                if any(r['transform'] == name and r['severity'] == severity
                       for r in results['records']):
                    continue
                started = time.perf_counter()
                if severity == 0:
                    record = {k: v for k, v in pristine.items()
                              if k not in ('identity', 'wall_seconds')}
                else:
                    a, b = fitted_transform(train, val, name, severity, seed=0)
                    record = evaluate_vectors(a, b, rows)
                record.update(transform=name, severity=severity,
                              wall_seconds=time.perf_counter() - started)
                record['intervention_parameters'] = {
                    'fit_split': 'train', 'seed': 0, 'renormalize': False,
                    'retained_svd_components':
                        max(1, round(min(train.shape) * (1 - severity)))
                        if name == 'rank_truncation' and severity > 0 else None,
                    'zero_severity_identity': severity == 0,
                }
                results['records'].append(record)
                atomic_json(result_path, results)
                print(f'{name} severity={severity}: {record["wall_seconds"]:.1f}s', flush=True)
    image_hashes = [h for chunk in manifest['chunks'] for h in chunk['image_sha256']]
    train_hashes = {h for h, row in zip(image_hashes, rows, strict=True) if row['split'] == 'train'}
    val_hashes = {h for h, row in zip(image_hashes, rows, strict=True) if row['split'] == 'val'}
    atomic_json(args.root / 'verification.json', {
        'claim_label': '[measurement]', 'status': 'PILOT / EXPLORATORY',
        'shape': list(matrix.shape), 'all_finite': bool(np.isfinite(matrix).all()),
        'norm_range': [float(np.linalg.norm(matrix, axis=1).min()),
                       float(np.linalg.norm(matrix, axis=1).max())],
        'sample_cache_alignment': True, 'stress_record_count': len(results['records']),
        'train_count': len(train), 'validation_count': len(val),
        'train_validation_source_hash_overlap_count': len(train_hashes & val_hashes),
        'duplicate_source_hash_count': len(image_hashes) - len(set(image_hashes)),
        'threshold_verdicts_applied': False,
        'canonical_dimension': 768, 'neural_inference_during_analysis': False,
        'semantics_check': json.loads((args.root / 'extraction_metrics.json').read_text())
                           ['semantics_check'],
        'identity': identity,
    })


if __name__ == '__main__':
    main()
