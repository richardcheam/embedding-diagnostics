"""Source/sample/input freeze only; no model inference or scientific endpoints."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import runpy
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from transformers import AutoProcessor

from embedding_diagnostics.embedding_cache import atomic_json
from embedding_diagnostics.models.embeddinggemma2 import MODEL_REVISION
from embedding_diagnostics.paired_data import group_sources, join_source, select_groups
from embedding_diagnostics.source_sensitivity import digest

ROOT=Path('experiments/phaseC_paired_replication')
SOURCE=Path('/mnt/hdd/data/datasets/COCO-original-2017')
MODEL=Path('/mnt/hdd/data/hf/hub/models--google--embeddinggemma-2/snapshots')/MODEL_REVISION
PREFIXES={'query':'task: search result | query: ','document':'title: none | text: '}


def validate_inspection(rows, inspection):
    """Machine gate checks coverage, not truthfulness or human-level adjudication."""
    if len(rows) != 1000 or inspection['reviewed_ids'] != [r['image_id'] for r in rows]:
        raise ValueError('complete selected-source content inspection required before freeze')
    if inspection['confirmed_source_defect_ids']:
        raise ValueError('source defect requires explicit pre-inference protocol amendment')


def prepare():
    if (ROOT/'freeze.json').exists():
        raise ValueError('existing freeze must never be overwritten')
    sample=json.loads((ROOT/'sample_draft.json').read_text())
    inspection=json.loads((ROOT/'content_inspection.json').read_text())
    rows=sample['selected_rows']
    joined=join_source(json.loads((SOURCE/'captions_val2017.json').read_text()),
                       json.loads((SOURCE/'instances_val2017.json').read_text()))
    inventory=json.loads((ROOT/'source_inventory.json').read_text())
    grouped=group_sources(joined,{int(k):v for k,v in inventory['inventory'].items()})
    if select_groups(grouped) != rows:
        raise ValueError('source joins or deterministic sample draft changed')
    validate_inspection(rows,inspection)
    acquisition=json.loads((ROOT/'acquisition.json').read_text())
    for p,record in acquisition['files'].items():
        if digest(Path(p)) != record['sha256']:
            raise ValueError('source acquisition hash changed')
    configs=json.loads((MODEL/'config_sentence_transformers.json').read_text())
    if (configs['prompts']['SearchQuery'] != PREFIXES['query'] or
            configs['prompts']['Document'] != PREFIXES['document']):
        raise ValueError('pinned role instructions differ')
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    inputs={'image':[{'id':r['image_id'],'parent_image_id':r['image_id'],'role':'image',
                     'input_sha256':r['raw_sha256']} for r in rows]}
    for role in ('query','document'):
        inputs[role]=[]
        for row in rows:
            for ann in row['captions']:
                rendered=PREFIXES[role]+ann['caption']
                processed=processor(text=[rendered],return_tensors='pt',truncation=False)
                tokens=processed['input_ids'][0].tolist()
                if len(tokens)>8192:
                    raise ValueError('caption exceeds context; no silent truncation')
                inputs[role].append({'id':ann['id'],'parent_image_id':row['image_id'],
                                     'role':role,'input_sha256':hashlib.sha256(
                                         rendered.encode()).hexdigest(),
                                     'token_ids':tokens,'rendered':rendered})
    counts=Counter(c for r in rows for c in r['categories'])
    sample.update({'status':'frozen pre-inference independent evaluation sample',
                   'category_support':dict(sorted(counts.items())),
                   'category_ids':[c['id'] for c in acquisition['categories']],
                   'eligible_categories':sorted(c for c,n in counts.items() if n>=10),
                   'caption_count':5000,'image_group_count':1000})
    atomic_json(ROOT/'sample_manifest.json',sample)
    atomic_json(ROOT/'input_manifest.json',inputs)
    protected={}
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only','96a28b9',
                                    'experiments'],text=True).splitlines()
    for path in paths:
        expected=subprocess.check_output(['git','show',f'96a28b9:{path}'])
        if Path(path).read_bytes()!=expected:
            raise ValueError(f'accepted historical artifact changed: {path}')
        protected[path]=digest(Path(path))
    cache=Path('experiments/phaseC_c1_main/embedding-cache')
    manifest=json.loads((cache/'manifest.json').read_text())
    if digest(cache/'manifest.json') != json.loads(Path(
            'experiments/phaseC_c2/freeze.json').read_text())['canonical_cache_manifest_sha256']:
        raise ValueError('accepted canonical manifest changed')
    protected[str(cache/'manifest.json')]=digest(cache/'manifest.json')
    for chunk in manifest['chunks']:
        path=cache/chunk['file']
        if digest(path)!=chunk['sha256']:
            raise ValueError('historical canonical chunk changed')
        protected[str(path)]=chunk['sha256']
    pca=Path('experiments/phaseC_c2/transform-cache/train_pca.npz')
    protected[str(pca)]=digest(pca)
    files=list(Path('src/embedding_diagnostics').rglob('*.py'))+[
        Path('scripts/prepare_image_text_replication.py'),
        Path('scripts/run_image_text_replication.py'),
        Path('scripts/report_image_text_replication.py'),
        Path('docs/image-text-replication-proposal.md'),
        Path('docs/image-text-replication-plan.md'),Path('pyproject.toml'),Path('uv.lock')]
    files+=list(Path('tests').glob('test_paired*.py'))+[Path('tests/test_embeddinggemma2.py')]
    for p in files:
        if subprocess.check_output(['git','show',f'HEAD:{p}']) != p.read_bytes():
            raise ValueError('commit tested implementation and final protocol before freeze')
    files+=[ROOT/name for name in ('acquisition.json','source_inventory.json',
                                 'content_inspection.json')]
    source_hashes={p:r['sha256'] for p,r in acquisition['files'].items()}
    source_hashes.update({str(p):digest(p) for p in (SOURCE/'acquisition').iterdir()
                         if p.is_file()})
    model_files={str(p):digest(p) for p in MODEL.iterdir() if p.is_file()}
    for p in [MODEL/'1_Pooling/config.json',MODEL/'2_Normalize/config.json']:
        if p.exists():
            model_files[str(p)]=digest(p)
    freeze={'accepted_source_commit':'96a28b9','implementation_commit':subprocess.check_output(
        ['git','rev-parse','HEAD'],text=True).strip(),'created_at':datetime.now(UTC).isoformat(),
        'files_sha256':{str(p):digest(p) for p in files},'source_sha256':source_hashes,
        'protected_sha256':protected,'model_files_sha256':model_files,
        'sample_sha256':digest(ROOT/'sample_manifest.json'),
        'input_sha256':digest(ROOT/'input_manifest.json'),
        'conditions':['native_768','mrl_256','mrl_128'],'prefixes':PREFIXES,
        'model_revision':MODEL_REVISION,'device':'cpu','dtype':'float32','batch':1,'threads':4,
        'normalization':'masked mean with prompt; L2; MRL prefix plus L2',
        'scoring':'FP64 normalization and dot; block64; descending score then numeric ID',
        'bootstrap':{'resamples':5000,'seed':20261010,'unit':'parent image',
                     'scope':'conditional on fixed gallery; percentile 95% marginal intervals'},
        'resource_gate':{'max_rss_mib':4608,'window_seconds':60,
                         'consecutive_active_swap_windows':2,'swap_io_bytes':256*2**20},
        'runtime_backend':runpy.run_path('scripts/run_image_text_replication.py')[
            'runtime_backend'](),
        'runtime_versions':{n:importlib.metadata.version(n) for n in
                            ('numpy','torch','transformers','pillow','scikit-learn')},
        'source_scope':'publisher archive/annotation chain; '
                       'unknown pretraining and near duplicates',
        'endpoint_status':'not computed; no inference before freeze'}
    atomic_json(ROOT/'freeze.json',freeze)
    print('Prepared source/sample/input freeze; commit unchanged before inference.')


if __name__=='__main__':
    prepare()
