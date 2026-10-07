"""Extract the already frozen C1-pilot sample through the accepted C0 path."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch

from embedding_diagnostics.bdd_lance import LanceBdd
from embedding_diagnostics.embedding_cache import atomic_json
from embedding_diagnostics.models.embeddinggemma2 import MODEL_REVISION, EmbeddingGemma2
from embedding_diagnostics.phase_c import code_provenance, extract


def verify_semantics(snapshot):
    modules = json.loads((snapshot / 'modules.json').read_text())
    if [m['type'].rsplit('.', 1)[-1] for m in modules] != ['Transformer', 'Pooling', 'Normalize']:
        raise ValueError('pinned embedding module sequence changed')
    if [m['idx'] for m in modules] != [0, 1, 2]:
        raise ValueError('pinned module ordering changed')
    pooling = json.loads((snapshot / modules[1]['path'] / 'config.json').read_text())
    if pooling != {'embedding_dimension': 768, 'pooling_mode': 'mean', 'include_prompt': True}:
        raise ValueError('pinned pooling semantics changed')
    return {'module_sequence': ['Transformer', 'Pooling', 'Normalize'], 'pooling': pooling,
            'modules_sha256': hashlib.sha256((snapshot / 'modules.json').read_bytes()).hexdigest(),
            'adapter_policy': 'attention-mask mean pooling followed by L2 normalization'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiments/phaseC_c1_pilot'))
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    sample_path = args.root / 'sample_manifest.json'
    sample = json.loads(sample_path.read_text())
    if sample['protocol']['train_count'] != 384 or sample['protocol']['val_count'] != 128:
        raise ValueError('expected frozen 512-image pilot')
    torch.set_num_threads(4)
    started = time.perf_counter()
    dataset = LanceBdd(sample['dataset']['path'])
    if dataset.identity != sample['dataset']:
        raise ValueError('dataset identity changed since sample freeze')
    semantics = verify_semantics(args.model)
    identity = code_provenance()
    for name in ('scripts/extract_c1_pilot.py', 'src/embedding_diagnostics/phase_c_pilot.py'):
        identity['source_sha256'][name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    identity['sample_manifest_sha256'] = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    encoder = EmbeddingGemma2(str(args.model), revision=MODEL_REVISION, device='cpu')
    load_seconds = time.perf_counter() - started
    report = extract(dataset, sample['selected_rows'], encoder, args.root / 'embedding-cache',
                     batch_size=1, chunk_size=4, code_identity=identity,
                     selection_procedure=sample['protocol']['selection'])
    report.update(phase='C1-pilot', role='PILOT / EXPLORATORY', claim_label='[measurement]',
                  load_seconds=load_seconds, total_wall_seconds=time.perf_counter() - started,
                  images_per_second=report['new_images'] / report['extraction_seconds'],
                  semantics_check=semantics,
                  optional_batch_benchmark='omitted; validated batch 1 retained')
    atomic_json(args.root / 'extraction_metrics.json', report)
    manifest = json.loads((args.root / 'embedding-cache' / 'manifest.json').read_text())
    atomic_json(args.root / 'extraction_manifest.json', manifest)


if __name__ == '__main__':
    main()
