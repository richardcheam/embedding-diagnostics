"""Frozen paired-data extraction/evaluation; no changes to historical BDD caches."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import resource
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from threadpoolctl import threadpool_info, threadpool_limits

from embedding_diagnostics.embedding_cache import atomic_json
from embedding_diagnostics.models.embeddinggemma2 import (
    MODEL_REVISION,
    EmbeddingGemma2,
)
from embedding_diagnostics.paired_cache import PairedCache, load_paired_cache
from embedding_diagnostics.paired_evaluation import (
    category_scores,
    paired_interval,
    paired_scores,
    rankings,
)
from embedding_diagnostics.phase_c_compression import geometry_with_capacity, matryoshka
from embedding_diagnostics.phase_c_numerics import arithmetic_audit, resume_results, run_conditions
from embedding_diagnostics.source_sensitivity import digest, verify_hashes

ROOT = Path("experiments/phaseC_paired_replication")
SOURCE = Path("/mnt/hdd/data/datasets/COCO-original-2017")
MODEL = Path("/mnt/hdd/data/hf/hub/models--google--embeddinggemma-2/snapshots")/MODEL_REVISION
PREFIXES = {"query":"task: search result | query: ","document":"title: none | text: "}
CONDITIONS = ("native_768","mrl_256","mrl_128")


def read(path):
    return json.loads(Path(path).read_text())


def caption_input(ann, role, token_ids):
    rendered = PREFIXES[role] + ann["caption"]
    return {"id":ann["id"],"parent_image_id":ann["image_id"],"role":role,
            "input_sha256":hashlib.sha256(rendered.encode()).hexdigest(),
            "token_ids":token_ids,"rendered":rendered}


def resource_gate(*, rss_mib, windows):
    """Predeclared operational gate, not a semantic-performance threshold."""
    consecutive = 0
    for window in windows:
        if window["seconds"] < 60:
            continue
        consecutive = consecutive+1 if window["swap_io_bytes"] >= 256*2**20 else 0
        if consecutive >= 2:
            return False
    return rss_mib <= 4608


def validate_smoke(report):
    if not report.get('resource_gate_passed'):
        raise ValueError('passed contract resource gate required')
    for role, count in [('image',16),('query',80),('document',80)]:
        if (report.get('roles',{}).get(role,{}).get('completed',0) < count or
                not report.get('repeat_checks',{}).get(role,{}).get('equivalent')):
            raise ValueError('complete passed image and both text-role contract required')


def scoring_witness():
    audit = arithmetic_audit()
    return {'score_dtype':'float64', 'fp64_dot':audit['fp64_accumulation_witness'],
            'fp32_dot':audit['fp32_accumulation_witness'],
            'scope':'empirical arithmetic witness; not floating-point certification'}


def runtime_backend():
    with threadpool_limits(limits=4):
        return {'torch_config':torch.__config__.show(),
                'pools':[{k:v for k,v in p.items() if k != 'filepath'}
                         for p in threadpool_info()], 'scoring_witness':scoring_witness()}


def swap_counters():
    values = {k:int(v) for k,v in (line.split() for line in
                                 Path('/proc/vmstat').read_text().splitlines())}
    return (values['pswpin']+values['pswpout'])*os.sysconf('SC_PAGE_SIZE')


def checked_freeze():
    freeze = read(ROOT/'freeze.json')
    for name in ('freeze.json','sample_manifest.json','input_manifest.json'):
        path = ROOT/name
        committed = subprocess.check_output(['git','show',f'HEAD:{path}'])
        if path.read_bytes() != committed:
            raise ValueError('source/sample/input freeze must be committed unchanged')
    verify_hashes(freeze['files_sha256'])
    verify_hashes(freeze['protected_sha256'])
    verify_hashes(freeze['source_sha256'])
    verify_hashes(freeze['model_files_sha256'])
    for name,version in freeze['runtime_versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('frozen runtime version changed')
    if digest(ROOT/'sample_manifest.json') != freeze['sample_sha256']:
        raise ValueError('sample freeze mismatch')
    if digest(ROOT/'input_manifest.json') != freeze['input_sha256']:
        raise ValueError('input freeze mismatch')
    if runtime_backend() != freeze['runtime_backend']:
        raise ValueError('frozen numeric backend changed')
    return freeze


def cache_rows(role, inputs):
    return [{k:r[k] for k in ('id','parent_image_id','role','input_sha256')}
            for r in inputs[role]]


def cache_provenance(role,freeze):
    return {'freeze_sha256':digest(ROOT/'freeze.json'),'role':role,
            'model_revision':MODEL_REVISION,'dtype':'float32','batch':1,
            'threads':4,'pooling':'masked_mean_prompt_included_L2',
            'prefix':PREFIXES.get(role),'input_sha256':freeze['input_sha256']}


def validate_cache_identity(manifest,rows,provenance):
    if manifest['rows'] != rows or manifest['provenance'] != provenance:
        raise ValueError('canonical row alignment/inference provenance mismatch')


def saved_repeats(root,freeze_hash):
    path=root/'smoke_checks.json'
    if not path.exists():
        return {}
    state=read(path)
    if state['freeze_sha256'] != freeze_hash:
        raise ValueError('contract repeat freeze mismatch')
    return state['checks']


def finalize_intervals(root,records,image_ids):
    identity={'records_sha256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()
                                            ).hexdigest()}
    path=root/'interval_completion.json'
    if path.exists():
        completion=read(path)
        if completion['identity'] != identity:
            raise ValueError('interval completion source mismatch')
        verify_hashes(completion['files_sha256'])
        return 0
    intervals={}
    for record in records[1:]:
        intervals[record['condition']]={}
        for direction in ('t2i','i2t'):
            a=records[0]['retrieval'][direction]['per_parent']
            b=record['retrieval'][direction]['per_parent']
            intervals[record['condition']][direction]=paired_interval(
                [a[str(i)]['hit@10'] for i in image_ids],
                [b[str(i)]['hit@10'] for i in image_ids])
    atomic_json(root/'paired_intervals.json',intervals)
    atomic_json(path,{'identity':identity,'files_sha256':{
                     str(root/'paired_intervals.json'):digest(root/'paired_intervals.json')}})
    return 4


def completed_evaluation(path, identity):
    """Validate completed checksums/identity before any reference endpoint work."""
    state = resume_results(path,identity)
    names = [r['payload']['condition'] for r in state['records']]
    if names != list(CONDITIONS[:len(names)]):
        raise ValueError('resumed condition order mismatch')
    return names == list(CONDITIONS)


def extract(*, smoke=False):
    freeze = checked_freeze()
    if not smoke:
        validate_smoke(read(ROOT/'smoke_metrics.json'))
    sample, inputs = read(ROOT/'sample_manifest.json'), read(ROOT/'input_manifest.json')
    roles = ('image','query','document')
    targets = {'image':16,'query':80,'document':80} if smoke else {
        role:len(inputs[role]) for role in roles}
    started = time.perf_counter()
    counts, measures, windows = {}, {}, []
    repeats = saved_repeats(ROOT,digest(ROOT/'freeze.json')) if smoke else {}
    torch.set_num_threads(4)
    model = None
    last_clock, last_swap = time.perf_counter(), swap_counters()
    with threadpool_limits(limits=4), zipfile.ZipFile(SOURCE/'val2017.zip') as archive:
        for role in roles:
            rows = cache_rows(role, inputs)
            provenance = cache_provenance(role,freeze)
            phase_started = time.perf_counter()
            with PairedCache(ROOT/f'{role}/embedding-cache',provenance,rows) as cache:
                initial = cache.completed
                for index in range(initial,targets[role]):
                    if model is None:
                        model = EmbeddingGemma2(MODEL,revision=MODEL_REVISION,device='cpu')
                        atomic_json(ROOT/'model_provenance.json',model.provenance)
                    item = inputs[role][index]
                    if role == 'image':
                        source_row = sample['selected_rows'][index]
                        raw = archive.read('val2017/'+source_row['file_name'])
                        if hashlib.sha256(raw).hexdigest() != item['input_sha256']:
                            raise ValueError('source image continuity changed')
                        with Image.open(io.BytesIO(raw)) as image:
                            image = image.convert('RGB')
                        x = model.encode([image])
                    else:
                        caption = item['rendered'][len(PREFIXES[role]):]
                        encoded = model.processor(text=[item['rendered']],
                                                  return_tensors='pt',truncation=False)
                        if encoded['input_ids'][0].tolist() != item['token_ids']:
                            raise ValueError('frozen caption preprocessing changed')
                        x = model.encode_text([caption],role=role)
                    if smoke and index == 0 and role not in repeats:
                        repeated = model.encode([image]) if role == 'image' else (
                            model.encode_text([caption],role=role))
                        difference = float(np.max(np.abs(x-repeated)))
                        equivalent = bool(np.allclose(x,repeated,rtol=1e-5,atol=1e-6))
                        repeats[role] = {'independent_repeat_count':1,
                                         'max_abs_difference':difference,
                                         'equivalent':equivalent}
                        atomic_json(ROOT/'smoke_checks.json',{'freeze_sha256':
                                    digest(ROOT/'freeze.json'),'checks':repeats})
                        if not equivalent:
                            raise ValueError('FP32 contract repeat exceeds frozen tolerance')
                    cache.append(x)
                    now = time.perf_counter()
                    if now-last_clock >= 60:
                        current_swap = swap_counters()
                        windows.append({'seconds':now-last_clock,
                                        'swap_io_bytes':current_swap-last_swap})
                        last_clock, last_swap = now, current_swap
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
                    if not resource_gate(rss_mib=rss,windows=windows):
                        atomic_json(ROOT/'resource_blocker.json',{'role':role,'completed':
                                    cache.completed,'peak_rss_mib':rss,'swap_windows':windows})
                        raise RuntimeError('resource gate failed; prefix checkpoint retained')
                    if cache.completed % 16 == 0:
                        print(f'{role}: {cache.completed}/{targets[role]} RSS {rss:.1f} MiB',
                              flush=True)
                counts[role] = cache.completed-initial
                duration = time.perf_counter()-phase_started
                measures[role] = {'new_rows':counts[role],'completed':cache.completed,
                                  'seconds':duration,'seconds_per_new_row':
                                  duration/counts[role] if counts[role] else None}
    report = {'smoke':smoke,'roles':measures,'wall_seconds':time.perf_counter()-started,
              'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'swap_windows':windows,'resource_gate_passed':resource_gate(
                  rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,windows=windows),
              'new_inferences':sum(counts.values()),'model_loaded':model is not None}
    report['repeat_checks'] = repeats
    report['forward_calls'] = report['new_inferences'] + len(repeats)
    if smoke:
        report['projected_full_seconds'] = sum(
            len(inputs[r])*measures[r]['seconds_per_new_row'] for r in roles
            if measures[r]['seconds_per_new_row'] is not None)
    path = ROOT/('smoke_metrics.json' if smoke else 'extraction_metrics.json')
    if path.exists():
        path = ROOT/('smoke_resume.json' if smoke else 'extraction_resume.json')
    atomic_json(path,report)
    verify_hashes(freeze['protected_sha256'])
    print(json.dumps(report),flush=True)


def bind_cache():
    freeze = checked_freeze()
    files, shapes = {}, {}
    for role in ('image','query','document'):
        root = ROOT/f'{role}/embedding-cache'
        x, manifest = load_paired_cache(root)
        validate_cache_identity(manifest,cache_rows(role,read(ROOT/'input_manifest.json')),
                                cache_provenance(role,freeze))
        files[str(root/'manifest.json')] = digest(root/'manifest.json')
        for c in manifest['chunks']:
            files[str(root/c['file'])] = c['sha256']
        shapes[role] = list(x.shape)
    binding = {'freeze_sha256':digest(ROOT/'freeze.json'),'files_sha256':files,
               'shapes':shapes,'status':'canonical cache binding before scientific endpoints'}
    path = ROOT/'cache_binding.json'
    if path.exists() and read(path) != binding:
        raise ValueError('existing canonical binding mismatch')
    atomic_json(path,binding)
    verify_hashes(freeze['protected_sha256'])


def evaluate():
    freeze = checked_freeze()
    binding = read(ROOT/'cache_binding.json')
    if binding['freeze_sha256'] != digest(ROOT/'freeze.json'):
        raise ValueError('canonical binding/freeze mismatch')
    if subprocess.check_output(['git','show',f'HEAD:{ROOT}/cache_binding.json']) != (
        ROOT/'cache_binding.json').read_bytes():
        raise ValueError('canonical binding must be committed before endpoints')
    verify_hashes(binding['files_sha256'])
    identity = {'freeze_sha256':digest(ROOT/'freeze.json'),
                'cache_binding_sha256':digest(ROOT/'cache_binding.json')}
    if completed_evaluation(ROOT/'results.json',identity):
        state=resume_results(ROOT/'results.json',identity)
        interval_calls=finalize_intervals(ROOT,[r['payload'] for r in state['records']],
                         [r['id'] for r in read(ROOT/'input_manifest.json')['image']])
        atomic_json(ROOT/'endpoint_resume.json',{'endpoint_calls':0,'interval_calls':interval_calls,'reference_rankings':0,
                    'protected_files':len(freeze['protected_sha256']),'cache_preserved':True})
        return
    matrices = {r:load_paired_cache(ROOT/f'{r}/embedding-cache')[0]
                for r in ('image','query','document')}
    sample, inputs = read(ROOT/'sample_manifest.json'),read(ROOT/'input_manifest.json')
    ids = {r:[v['id'] for v in inputs[r]] for r in matrices}
    parents = {r:[v['parent_image_id'] for v in inputs[r]] for r in matrices}
    native_top = {}
    with threadpool_limits(limits=4):
        for direction,q,g in [('t2i','query','image'),('i2t','image','document')]:
            native_top[direction] = rankings(matrices[q],matrices[g],ids[q],ids[g])[0]
    started = time.perf_counter()
    def condition(name):
        dimension = int(name.split('_')[-1])
        derived = {r:matryoshka(x,dimension) for r,x in matrices.items()}
        result = {'condition':name,'dimension':dimension,'geometry':{},'retrieval':{}}
        for role,x in derived.items():
            result['geometry'][role] = geometry_with_capacity(x)
        for direction,q,g in [('t2i','query','image'),('i2t','image','document')]:
            top,ties = rankings(derived[q],derived[g],ids[q],ids[g],tie_depths=(1,5,10))
            result['retrieval'][direction] = paired_scores(top,parents[q],parents[g])
            overlap = [len(set(a)&set(b))/10 for a,b in zip(top,native_top[direction],
                                                         strict=True)]
            result['retrieval'][direction].update({'overlap@10':float(np.mean(overlap)),
                'changed_queries':sum(v < 1 for v in overlap),'boundary_ties':ties['10'],
                'ties_at_k':ties,
                'top10_ids':np.asarray(ids[g])[top].tolist()})
            if name == 'native_768':
                rotate = dict(zip(ids['image'],ids['image'][1:]+ids['image'][:1],strict=True))
                scrambled = [rotate[v] for v in parents['query']] if q == 'query' else parents[q]
                scrambled_gallery = [rotate[v] for v in parents[g]] if g == 'document' else (
                    parents[g])
                result.setdefault('pairing_control',{})[direction] = paired_scores(
                    top,scrambled,scrambled_gallery)
        top,ties = rankings(derived['image'],derived['image'],ids['image'],ids['image'],
                            self_exclude=True)
        result['category'] = category_scores(top,[r['categories'] for r in
                                            sample['selected_rows']],sample['category_ids'])
        result['category']['boundary_ties'] = ties
        return result
    with threadpool_limits(limits=4):
        state,calls = run_conditions(ROOT/'results.json',identity,CONDITIONS,condition)
    records = [r['payload'] for r in state['records']]
    interval_calls=finalize_intervals(ROOT,records,ids['image'])
    verify_hashes(freeze['protected_sha256'])
    verify_hashes(binding['files_sha256'])
    path=ROOT/('verification.json' if calls else 'endpoint_resume.json')
    atomic_json(path,{'endpoint_calls':calls,'interval_calls':interval_calls,'wall_seconds':time.perf_counter()-started,
                     'protected_files':len(freeze['protected_sha256']),'cache_preserved':True})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('smoke','extract','bind','evaluate'))
    args=parser.parse_args()
    if args.stage in ('smoke','extract'):
        extract(smoke=args.stage == 'smoke')
    elif args.stage == 'bind':
        bind_cache()
    else:
        evaluate()
