"""Execute a committed C1-main freeze; no extraction is the default."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
from threadpoolctl import threadpool_limits

from embedding_diagnostics.bdd_lance import LanceBdd
from embedding_diagnostics.degradation import DEFAULT_SEVERITIES, TRANSFORMS
from embedding_diagnostics.embedding_cache import atomic_json, load_cache
from embedding_diagnostics.models.embeddinggemma2 import MODEL_REVISION, EmbeddingGemma2
from embedding_diagnostics.phase_c import extract
from embedding_diagnostics.phase_c_main import evaluate_main, paired_effects
from embedding_diagnostics.phase_c_pilot import fitted_transform

ROOT = Path('experiments/phaseC_c1_main')


def checked_freeze():
    path = ROOT / 'freeze.json'
    payload = path.read_bytes()
    try:
        committed = subprocess.check_output(['git','show',f'HEAD:{path}'],
                                            stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as error:
        raise ValueError('main protocol freeze is not committed') from error
    if payload != committed:
        raise ValueError('main protocol freeze is not committed or changed')
    freeze = json.loads(payload)
    for name, digest in freeze['files_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'frozen file changed: {name}')
    for name, version in freeze['runtime_versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'frozen runtime changed: {name}')
    return freeze, json.loads((ROOT / 'sample_manifest.json').read_text())


def main_identity(freeze, payload):
    """Stable committed-freeze identity; generated result files cannot break resume."""
    commit = subprocess.check_output(
        ['git', 'log', '-1', '--format=%H', '--', str(ROOT / 'freeze.json')],
        text=True).strip()
    if not commit:
        raise ValueError('main protocol freeze has no commit')
    return {'commit': commit,
            'main_freeze_sha256': hashlib.sha256(payload).hexdigest(),
            'frozen_files_sha256': freeze['files_sha256']}


def validate_main_cache(manifest, sample, identity):
    if manifest['selected_rows'] != sample['selected_rows']:
        raise ValueError('cache does not match frozen IDs')
    if manifest['provenance']['code'] != identity:
        raise ValueError('cache extraction freeze provenance mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--extract',action='store_true')
    group.add_argument('--evaluate',action='store_true')
    parser.add_argument('--model',type=Path,default=Path(
        '/mnt/hdd/data/hf/hub/models--google--embeddinggemma-2/snapshots/'+MODEL_REVISION))
    args = parser.parse_args()
    freeze,sample = checked_freeze()
    torch.set_num_threads(4)
    identity = main_identity(freeze, (ROOT / 'freeze.json').read_bytes())
    if args.extract:
        started = time.perf_counter()
        dataset = LanceBdd(sample['dataset']['path'])
        if dataset.identity != sample['dataset']:
            raise ValueError('source identity changed')
        encoder = EmbeddingGemma2(str(args.model),revision=MODEL_REVISION,device='cpu')
        report = extract(dataset,sample['selected_rows'],encoder,ROOT / 'embedding-cache',
                         code_identity=identity,selection_procedure=sample['protocol']['selection'])
        report.update(phase='C1-main',role='frozen H1-H4 study',claim_label='[measurement]',
                      total_wall_seconds=time.perf_counter()-started)
        atomic_json(ROOT / 'extraction_metrics.json',report)
        atomic_json(ROOT / 'extraction_manifest.json',json.loads(
            (ROOT / 'embedding-cache/manifest.json').read_text()))
        return
    matrix,manifest = load_cache(ROOT / 'embedding-cache')
    validate_main_cache(manifest, sample, identity)
    train = matrix[[r['split']=='train' for r in sample['selected_rows']]]
    val = matrix[[r['split']=='val' for r in sample['selected_rows']]]
    identity['cache_manifest_sha256'] = hashlib.sha256(
        (ROOT / 'embedding-cache/manifest.json').read_bytes()).hexdigest()
    results_path = ROOT / 'stress_results.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else {
        'identity':identity,'records':[]}
    if results['identity'] != identity:
        raise ValueError('analysis resume provenance mismatch')
    with threadpool_limits(limits=4):
        path = ROOT / 'pristine_metrics.json'
        pristine = json.loads(path.read_text()) if path.exists() else None
        if pristine is not None and pristine['identity'] != identity:
            raise ValueError('pristine resume provenance mismatch')
        if pristine is None:
            pristine = evaluate_main(train,val,sample)
            pristine['identity'] = identity
            atomic_json(path,pristine)
        for name in TRANSFORMS:
            for severity in DEFAULT_SEVERITIES:
                if any(r['transform']==name and r['severity']==severity
                       for r in results['records']):
                    continue
                tick = time.perf_counter()
                if severity == 0:
                    record = {k:v for k,v in pristine.items() if k != 'identity'}
                else:
                    a,b = fitted_transform(train,val,name,severity)
                    record = evaluate_main(a,b,sample)
                record.update(transform=name,severity=severity,wall_seconds=time.perf_counter()-tick,
                              status='exploratory' if name=='mean_interpolation' else
                                     'predeclared H1-H4 endpoint')
                results['records'].append(record)
                atomic_json(results_path,results)
                print(f'{name} {severity} complete',flush=True)
        atomic_json(ROOT / 'hypothesis_effects.json',paired_effects(pristine,results['records']))
        atomic_json(ROOT / 'verification.json', {
            'claim_label':'[measurement]', 'main_freeze_sha256':identity['main_freeze_sha256'],
            'shape':list(matrix.shape), 'train_count':len(train), 'validation_count':len(val),
            'stress_records':len(results['records']), 'primary_probe':'standardized',
            'phase_b_thresholds_applied':False, 'pilot_ID_overlap':sample['pilot_overlap'],
            'eligible_probe_classes':sample['eligible_probe_class_names'],
        })


if __name__ == '__main__':
    main()
