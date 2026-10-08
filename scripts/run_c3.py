"""Freeze and execute the bounded C3 exact-search study from canonical vectors."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import resource
import socket
import subprocess
import time
from contextlib import ExitStack, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from embedding_diagnostics.embedding_cache import atomic_json, load_cache
from embedding_diagnostics.phase_c_numerics import (
    CONDITIONS,
    TrainInt8,
    arithmetic_audit,
    evaluate_condition,
    mean_stress,
    run_conditions,
    split_rows,
)

ROOT = Path('experiments/phaseC_c3')
SOURCE = Path('experiments/phaseC_c1_main')
ACCEPTED = '50d03a8d5726465a49ef60d426e652905646cefb'
VERSIONS = ('numpy', 'scipy', 'scikit-learn', 'torch', 'pillow', 'threadpoolctl')
FILES = ('scripts/run_c3.py', 'scripts/report_c3.py', 'tests/test_phase_c_numerics.py',
         'docs/c3-protocol.md', 'pyproject.toml', 'uv.lock')
FILES = tuple(sorted(set(FILES) | {str(p) for p in
                      Path('src/embedding_diagnostics').rglob('*.py')}))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def committed_bytes(path, commit='HEAD'):
    try:
        return subprocess.check_output(['git', 'show', f'{commit}:{path}'],
                                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise ValueError(f'file is not committed: {path}') from e


def historical_paths():
    return subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', ACCEPTED,
                                    'experiments/phaseC_c0', 'experiments/phaseC_c1_pilot',
                                    'experiments/phaseC_c1_hardening', 'experiments/phaseC_c1_main',
                                    'experiments/phaseC_c2'], text=True).splitlines()


def historical_hashes():
    hashes = {}
    for name in historical_paths():
        if Path(name).read_bytes() != committed_bytes(name, ACCEPTED):
            raise ValueError(f'accepted historical file changed: {name}')
        hashes[name] = digest(name)
    return hashes


def source_cache():
    matrix, manifest = load_cache(SOURCE/'embedding-cache')
    sample = json.loads((SOURCE/'sample_manifest.json').read_text())
    c2 = json.loads(Path('experiments/phaseC_c2/freeze.json').read_text())
    if matrix.shape != (3000, 768) or matrix.dtype != np.float32:
        raise ValueError('canonical cache shape/dtype mismatch')
    for path, key in ((SOURCE/'sample_manifest.json', 'sample_manifest_sha256'),
                      (SOURCE/'embedding-cache/manifest.json', 'canonical_cache_manifest_sha256')):
        if digest(path) != c2[key]:
            raise ValueError(f'accepted canonical hash changed: {path}')
    if hashlib.sha256(matrix.tobytes()).hexdigest() != c2['canonical_matrix_sha256']:
        raise ValueError('accepted matrix bytes changed')
    if manifest != json.loads((SOURCE/'extraction_manifest.json').read_text()):
        raise ValueError('extraction manifest mismatch')
    train, val, rows = split_rows(matrix, manifest, sample)
    if (len(train), len(val)) != (2000, 1000):
        raise ValueError('frozen split counts changed')
    return matrix, sample, train, val, rows


def backend_identity():
    backend = io.StringIO()
    with redirect_stdout(backend):
        np.show_config()
    return {'numpy_configuration': backend.getvalue(),
            'threadpools': threadpool_info()}


def validate_execution_runtime(freeze):
    if backend_identity() != freeze['runtime_backend']:
        raise ValueError('C3 frozen backend/thread configuration changed')
    if arithmetic_audit() != freeze['arithmetic_audit']:
        raise ValueError('C3 frozen arithmetic witness changed')


def prepare_freeze():
    if (ROOT/'freeze.json').exists():
        raise ValueError('C3 freeze exists; never overwrite')
    for name in FILES:
        if Path(name).read_bytes() != committed_bytes(name):
            raise ValueError(f'commit tested code/protocol first: {name}')
    history = historical_hashes()
    matrix, sample, train, _, _ = source_cache()
    quantizer = TrainInt8.fit(train)  # calibration only; no validation scores
    with threadpool_limits(limits=4):
        backend = backend_identity()
        audit = arithmetic_audit()
    atomic_json(ROOT/'freeze.json', {
        'created_at': datetime.now(UTC).isoformat(), 'accepted_source_commit': ACCEPTED,
        'C3_code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'canonical_matrix_sha256': hashlib.sha256(matrix.tobytes()).hexdigest(),
        'canonical_cache_manifest_sha256': digest(SOURCE/'embedding-cache/manifest.json'),
        'sample_manifest_sha256': digest(SOURCE/'sample_manifest.json'),
        'files_sha256': {name: digest(name) for name in FILES}, 'historical_sha256': history,
        'runtime_versions': {name: importlib.metadata.version(name) for name in VERSIONS},
        'conditions': list(CONDITIONS), 'dimension': 768, 'k': 10, 'threads': 4,
        'quantizer_scales_sha256': hashlib.sha256(quantizer.scales.tobytes()).hexdigest(),
        'quantizer_scales': quantizer.scales.tolist(),
        'mean_stress': {'transform': 'existing training-fitted mean_injection',
                        'severity': .99, 'seed': 0, 'dtype': 'float64',
                        'storage_rounding': 'before row L2 normalization'},
        'ordering': 'descending cosine then ascending image_id; self excluded by ID',
        'margin_bins': 'inverted_cdf quartiles; unique cuts below maximum; '
                       'searchsorted side=left; ties together; one to four bins',
        'arithmetic_audit': audit, 'runtime_backend': backend,
        'train_count': 2000, 'validation_count': 1000,
        'eligibility': sample['eligibility_rule'],
        'uncertainty': 'paired descriptive effects on previously examined fixed sample',
        'scope': 'no neural inference, image decoding, ANN, new sample or C4',
    })
    print('C3 freeze prepared; commit before endpoints.', flush=True)


def checked_freeze():
    path = ROOT/'freeze.json'
    if path.read_bytes() != committed_bytes(path):
        raise ValueError('C3 freeze must be committed unchanged')
    freeze = json.loads(path.read_text())
    for section in ('files_sha256', 'historical_sha256'):
        for name, expected in freeze[section].items():
            if digest(name) != expected:
                raise ValueError(f'C3 frozen file changed: {name}')
    for name, version in freeze['runtime_versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'C3 frozen runtime changed: {name}')
    return freeze


def forbidden(*args, **kwargs):
    raise RuntimeError('C3 forbids network, model loading and image decoding')


def run():
    started = time.perf_counter()
    freeze = checked_freeze()
    history = historical_hashes()
    with ExitStack() as guards:
        guards.enter_context(patch.object(socket.socket, 'connect', forbidden))
        guards.enter_context(patch.object(socket, 'create_connection', forbidden))
        guards.enter_context(patch('PIL.Image.open', forbidden))
        guards.enter_context(patch('embedding_diagnostics.models.embeddinggemma2.'
                                   'EmbeddingGemma2.__init__', forbidden))
        guards.enter_context(threadpool_limits(limits=freeze['threads']))
        validate_execution_runtime(freeze)
        matrix, sample, train, val, rows = source_cache()
        if hashlib.sha256(matrix.tobytes()).hexdigest() != freeze['canonical_matrix_sha256']:
            raise ValueError('C3 matrix freeze mismatch')
        quantizer = TrainInt8.fit(train)
        if hashlib.sha256(quantizer.scales.tobytes()).hexdigest() != \
                freeze['quantizer_scales_sha256']:
            raise ValueError('C3 calibration resume mismatch')
        _, stressed = mean_stress(train, val)
        identity = {'freeze_sha256': digest(ROOT/'freeze.json'),
                    'C3_code_commit': freeze['C3_code_commit'],
                    'canonical_matrix_sha256': freeze['canonical_matrix_sha256']}
        def evaluate(name):
            tick = time.perf_counter()
            raw = stressed if name.startswith('mean99') else val
            record = evaluate_condition(name, raw, raw, rows, quantizer, k=freeze['k'])
            record['wall_seconds'] = time.perf_counter()-tick
            print(f'{name} complete ({record["wall_seconds"]:.3f}s)', flush=True)
            return record
        state, calls = run_conditions(ROOT/'results.json', identity, freeze['conditions'], evaluate)
        pristine = json.loads((SOURCE/'pristine_metrics.json').read_text())
        stresses = json.loads((SOURCE/'stress_results.json').read_text())['records']
        mean = next(r for r in stresses if r['transform'] == 'mean_injection' and
                    r['severity'] == .99)
        refs = {e['payload']['condition']: e['payload'] for e in state['records']}
        continuity = {name: {attr: refs[name]['attributes'][attr]['p10']-
                            historical['attributes'][attr]['retrieval_p10']
                            for attr in ('weather', 'scene', 'timeofday')}
                      for name, historical in (('native_ref', pristine), ('mean99_ref', mean))}
        if historical_hashes() != history:
            raise ValueError('C3 modified historical artifacts')
        if (digest(SOURCE/'embedding-cache/manifest.json') !=
                freeze['canonical_cache_manifest_sha256']):
            raise ValueError('C3 cache manifest changed')
        # Verify every canonical chunk again without accessing image bytes.
        final, _ = load_cache(SOURCE/'embedding-cache')
        if hashlib.sha256(final.tobytes()).hexdigest() != freeze['canonical_matrix_sha256']:
            raise ValueError('C3 canonical cache changed')
        atomic_json(ROOT/'verification.json' if calls else ROOT/'resume_verification.json', {
            'claim_label': '[ours]', 'identity': identity,
            'condition_records': len(state['records']), 'this_run_endpoint_calls': calls,
            'wall_seconds': time.perf_counter()-started,
            'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'historical_files_preserved': len(history), 'canonical_cache_preserved': True,
            'network_model_image_guards': True, 'protocol_deviations': [],
            'arithmetic_audit': arithmetic_audit(), 'threadpools': threadpool_info(),
            'historical_reference_p10_deltas': continuity,
            'training_rows': len(train), 'validation_rows': len(val),
            'support': sample['support'], 'C4_started': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--freeze', action='store_true')
    choice.add_argument('--run', action='store_true')
    args = parser.parse_args()
    prepare_freeze() if args.freeze else run()
