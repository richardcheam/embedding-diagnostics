"""Freeze/run C2 entirely from cached vectors, without any encoder or image access."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from embedding_diagnostics.embedding_cache import atomic_json, load_cache
from embedding_diagnostics.phase_c_compression import (
    DIMENSIONS,
    TrainPCA,
    evaluate_representation,
    matryoshka,
    validate_source,
)

ROOT = Path('experiments/phaseC_c2')
SOURCE = Path('experiments/phaseC_c1_main')
C1_COMMIT = '0c19a85'
VERSIONS = ('numpy', 'scipy', 'scikit-learn', 'torch', 'torchvision', 'pillow', 'threadpoolctl')
FILES = ('scripts/run_c2.py', 'src/embedding_diagnostics/phase_c_compression.py',
         'src/embedding_diagnostics/embedding_cache.py',
         'src/embedding_diagnostics/diagnostics/metrics.py',
         'src/embedding_diagnostics/diagnostics/probe.py',
         'src/embedding_diagnostics/diagnostics/retrieval.py',
         'src/embedding_diagnostics/bdd100k.py',
         'src/embedding_diagnostics/phase_c_evaluation.py',
         'pyproject.toml', 'uv.lock', 'docs/c2-protocol.md',
         'experiments/phaseC_c2/rank_transform_audit.json',
         'experiments/phaseC_c2/model_card_evidence.json',
         'experiments/phaseC_c1_main/sample_manifest.json',
         'experiments/phaseC_c1_main/pristine_metrics.json',
         'experiments/phaseC_c1_main/stress_results.json')
FILES = tuple(sorted(set(FILES) | {str(p) for p in
                      Path('src/embedding_diagnostics').rglob('*.py')}))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def committed_bytes(path, commit='HEAD'):
    try:
        return subprocess.check_output(['git', 'show', f'{commit}:{path}'],
                                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as error:
        raise ValueError(f'file is not committed: {path}') from error


def source_cache():
    sample_path = SOURCE / 'sample_manifest.json'
    if sample_path.read_bytes() != committed_bytes(sample_path, C1_COMMIT):
        raise ValueError('accepted C1 sample changed')
    matrix, manifest = load_cache(SOURCE / 'embedding-cache')
    if matrix.shape != (3000, 768) or matrix.dtype != np.float32:
        raise ValueError('canonical matrix contract changed')
    sample = json.loads(sample_path.read_text())
    return matrix, manifest, sample


def prepare_freeze():
    path = ROOT / 'freeze.json'
    if path.exists():
        raise ValueError('C2 freeze already exists; do not silently replace it')
    code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    for name in FILES:
        if Path(name).read_bytes() != committed_bytes(name):
            raise ValueError(f'commit implementation/protocol first: {name}')
    matrix, manifest, sample = source_cache()
    manifest_digest = digest(SOURCE / 'embedding-cache/manifest.json')
    validate_source(manifest, sample, manifest_digest,
                    {'canonical_cache_manifest_sha256': manifest_digest})
    recorded = SOURCE / 'extraction_manifest.json'
    if manifest != json.loads(committed_bytes(recorded, C1_COMMIT)):
        raise ValueError('accepted C1 extraction manifest changed')
    freeze = {
        'created_at': datetime.now(UTC).isoformat(), 'C1_source_commit': C1_COMMIT,
        'C2_code_commit': code_commit,
        'sample_manifest_sha256': digest(SOURCE / 'sample_manifest.json'),
        'canonical_cache_manifest_sha256': manifest_digest,
        'canonical_matrix_sha256': hashlib.sha256(matrix.tobytes()).hexdigest(),
        'files_sha256': {name: digest(name) for name in FILES},
        'runtime_versions': {name: importlib.metadata.version(name) for name in VERSIONS},
        'dimensions': [768, *DIMENSIONS],
        'representations': ['native_768', 'mrl_512', 'pca_512', 'mrl_256',
                            'pca_256', 'mrl_128', 'pca_128'],
        'normalization': 'native exact canonical copy; compressed per-row L2; zero fails',
        'arithmetic': 'canonical FP32; both compression paths FP64',
        'PCA': 'centered train-only full SVD; nested shared basis; no whitening',
        'eligible_classes': sample['eligible_probe_class_names'],
        'eligibility_rule': sample['eligibility_rule'],
        'semantic_endpoints': ['standardized accuracy', 'eligible balanced accuracy', 'P@10'],
        'geometry_endpoints': 'all canonical metrics plus RankMe/D and participation_ratio/D',
        'hypotheses': ['C2-H1', 'C2-H2', 'C2-H3', 'C2-H4'],
        'sample_status': 'previously examined C1 sample; not independent replication',
        'uncertainty': 'descriptive paired effects; training-seed intervals not estimable',
    }
    atomic_json(path, freeze)
    print('Prepared C2 freeze. Commit it before endpoint computation.', flush=True)


def checked_freeze():
    path = ROOT / 'freeze.json'
    if path.read_bytes() != committed_bytes(path):
        raise ValueError('C2 freeze is not committed unchanged')
    freeze = json.loads(path.read_text())
    for name, expected in freeze['files_sha256'].items():
        if digest(name) != expected:
            raise ValueError(f'frozen file changed: {name}')
    for name, version in freeze['runtime_versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'frozen runtime changed: {name}')
    return freeze


def paired_semantic_effects(records):
    def effects(a, b):
        return {attr: {
            'accuracy_delta': a['attributes'][attr]['probe_standardized']['accuracy']-
                              b['attributes'][attr]['probe_standardized']['accuracy'],
            'balanced_accuracy_delta':
                a['attributes'][attr]['probe_standardized']['balanced_accuracy']-
                b['attributes'][attr]['probe_standardized']['balanced_accuracy'],
            'retrieval_p10_delta': a['attributes'][attr]['retrieval_p10']-
                                  b['attributes'][attr]['retrieval_p10'],
        } for attr in a['attributes']}
    by_name = {r['representation']: r for r in records}
    native = by_name['native_768']
    return {
        'native_relative': {name: effects(record, native) for name, record in by_name.items()},
        'MRL_minus_PCA': {str(d): effects(by_name[f'mrl_{d}'], by_name[f'pca_{d}'])
                          for d in DIMENSIONS},
        '128_minus_larger': {method: {str(d): effects(by_name[f'{method}_128'],
                                                    by_name[f'{method}_{d}'])
                                    for d in (256, 512)} for method in ('mrl', 'pca')},
    }



def save_pca_parameters(path, parameters):
    """Preserve compatible parameters; reject mismatches without overwriting."""
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(parameters) or any(
                saved[name].shape != value.shape or saved[name].dtype != value.dtype or
                saved[name].tobytes() != value.tobytes() for name, value in parameters.items()
            ):
                raise ValueError('PCA parameter resume mismatch')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(handle, **parameters)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def run():
    freeze = checked_freeze()
    matrix, manifest, sample = source_cache()
    validate_source(manifest, sample, digest(SOURCE / 'embedding-cache/manifest.json'), freeze)
    if hashlib.sha256(matrix.tobytes()).hexdigest() != freeze['canonical_matrix_sha256']:
        raise ValueError('canonical matrix bytes changed')
    rows = sample['selected_rows']
    train = matrix[[r['split'] == 'train' for r in rows]]
    val = matrix[[r['split'] == 'val' for r in rows]]
    if (len(train), len(val)) != (2000, 1000):
        raise ValueError('physical sample splits changed')
    started = time.perf_counter()
    with threadpool_limits(limits=4):
        tick = time.perf_counter()
        pca = TrainPCA.fit(train, 512)
        parameters = {'mean': pca.mean, 'components': pca.components,
                      'explained_variance': pca.explained_variance}
        hashes = {name: hashlib.sha256(x.tobytes()).hexdigest()
                  for name, x in parameters.items()}
        fit_seconds = time.perf_counter()-tick
        identity = {'freeze_sha256': digest(ROOT / 'freeze.json'),
                    'C2_code_commit': freeze['C2_code_commit'],
                    'canonical_matrix_sha256': freeze['canonical_matrix_sha256'],
                    'PCA_parameter_sha256': hashes}
        path = ROOT / 'results.json'
        results = json.loads(path.read_text()) if path.exists() else {
            'claim_label': '[measurement]', 'identity': identity, 'records': [],
            'support': sample['support'], 'PCA_training_rows': pca.training_rows,
            'PCA_fit_seconds': fit_seconds}
        if results['identity'] != identity:
            raise ValueError('C2 result resume provenance mismatch')
        save_pca_parameters(ROOT / 'transform-cache/train_pca.npz', parameters)
        for name in freeze['representations']:
            if any(r['representation'] == name for r in results['records']):
                continue
            method, dimension = name.split('_')
            dimension = int(dimension)
            tick = time.perf_counter()
            if method == 'pca':
                a, b = pca.transform(train, dimension), pca.transform(val, dimension)
            else:
                a, b = matryoshka(train, dimension), matryoshka(val, dimension)
            record = evaluate_representation(a, b, sample)
            norms = np.concatenate([np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)])
            record.update(representation=name, dimension=dimension,
                          wall_seconds=time.perf_counter()-tick,
                          all_finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),
                          norm_range=[float(norms.min()), float(norms.max())])
            results['records'].append(record)
            atomic_json(path, results)
            print(f'{name} complete', flush=True)
        effects = paired_semantic_effects(results['records'])
        atomic_json(ROOT / 'paired_effects.json', effects | {'identity': identity,
                    'uncertainty': freeze['uncertainty']})
        pristine = json.loads((SOURCE / 'pristine_metrics.json').read_text())
        native = next(r for r in results['records'] if r['representation'] == 'native_768')
        continuity = {attr: {key: native['attributes'][attr]['probe_standardized'][key]-
                            pristine['attributes'][attr]['probe_standardized'][key]
                            for key in ('accuracy', 'balanced_accuracy')}
                      for attr in native['attributes']}
        atomic_json(ROOT / 'verification.json', {
            'claim_label': '[measurement]', 'identity': identity, 'representations': 7,
            'train_count': 2000, 'validation_count': 1000,
            'neural_inference_calls': 0, 'image_decodes': 0,
            'PCA_training_rows': pca.training_rows, 'PCA_shared_projection': True,
            'all_finite': all(r['all_finite'] for r in results['records']),
            'C1_native_probe_deltas': continuity,
            'this_run_wall_seconds': time.perf_counter()-started,
            'protocol_deviations': [], 'C3_C4_started': False,
        })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true')
    group.add_argument('--run', action='store_true')
    args = parser.parse_args()
    prepare_freeze() if args.freeze else run()
