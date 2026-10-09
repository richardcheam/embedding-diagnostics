"""Frozen role identities, resource gate and zero-call endpoint resume."""

import runpy
from pathlib import Path

import pytest

from embedding_diagnostics.phase_c_numerics import run_conditions


def workflow():
    return runpy.run_path(str(Path("scripts/run_image_text_replication.py")))


def test_source_text_role_identity_binds_rendered_bytes_and_tokens():
    fn=workflow()["caption_input"]
    query=fn({"id":1,"image_id":2,"caption":"A caption."},"query",[1,2])
    document=fn({"id":1,"image_id":2,"caption":"A caption."},"document",[3,4])
    assert query["id"] == document["id"] and query["role"] != document["role"]
    assert query["input_sha256"] != document["input_sha256"]
    assert query["token_ids"] == [1,2]
    assert query["parent_image_id"] == 2


def test_active_swap_gate_distinguishes_historical_swap_use():
    gate=workflow()["resource_gate"]
    assert gate(rss_mib=3600, windows=[{"seconds":60,"swap_io_bytes":0}])
    assert not gate(rss_mib=6000, windows=[])
    assert not gate(rss_mib=3600,windows=[{"seconds":60,"swap_io_bytes":300*2**20}]*2)
    assert gate(rss_mib=3600,windows=[{"seconds":60,"swap_io_bytes":300*2**20}])


def test_completed_endpoint_resume_is_zero_calls_and_rejects_identity_change(tmp_path):
    path=tmp_path/'results.json'
    def evaluate(name):
        return {"condition":name,"fixed":True}
    names=("native_768","mrl_256","mrl_128")
    run_conditions(path,{"cache_hash":"a"},names,evaluate)
    before=path.read_bytes()
    def forbidden(name):
        raise AssertionError("completed resume computed endpoint")
    _,calls=run_conditions(path,{"cache_hash":"a"},names,forbidden)
    assert calls == 0 and before == path.read_bytes()
    with pytest.raises(ValueError,match="provenance"):
        run_conditions(path,{"cache_hash":"b"},names,forbidden)


def test_runner_short_circuits_complete_analysis_before_reference_ranking(tmp_path):
    fn=workflow()["completed_evaluation"]
    path=tmp_path/'results.json'
    names=("native_768","mrl_256","mrl_128")
    identity={"cache_hash":"a"}
    assert not fn(path,identity)
    run_conditions(path,identity,names,lambda name:{"condition":name})
    assert fn(path,identity)
    with pytest.raises(ValueError):
        fn(path,{"cache_hash":"b"})


def test_inspection_gate_requires_every_exact_id_and_no_source_defect():
    fn=runpy.run_path('scripts/prepare_image_text_replication.py')['validate_inspection']
    rows=[{'image_id':i} for i in range(1000)]
    inspection={'reviewed_ids':list(range(1000)),'confirmed_source_defect_ids':[]}
    fn(rows,inspection)
    for bad in [{'reviewed_ids':list(range(999)),'confirmed_source_defect_ids':[]},
                {'reviewed_ids':list(reversed(range(1000))),'confirmed_source_defect_ids':[]},
                {'reviewed_ids':list(range(1000)),'confirmed_source_defect_ids':[1]}]:
        with pytest.raises(ValueError):
            fn(rows,bad)


def test_full_extraction_requires_passed_complete_contract():
    fn=workflow()['validate_smoke']
    healthy={'resource_gate_passed':True,'roles':{r:{'completed':n} for r,n in
             [('image',16),('query',80),('document',80)]},
             'repeat_checks':{r:{'equivalent':True} for r in ['image','query','document']}}
    fn(healthy)
    for bad in [healthy | {'resource_gate_passed':False},
                healthy | {'repeat_checks':{}},healthy | {'roles':{}}]:
        with pytest.raises(ValueError):
            fn(bad)


def test_score_accumulation_witness_is_fp64_not_float32():
    fn=workflow()['scoring_witness']
    value=fn()
    assert value['score_dtype'] == 'float64'
    assert value['fp64_dot'] == 766 and value['fp32_dot'] != 766


def test_binding_rejects_same_rows_wrong_role_provenance():
    fn=workflow()['validate_cache_identity']
    rows=[{'id':1,'parent_image_id':1,'role':'image','input_sha256':'a'*64}]
    expected={'freeze':'a','role':'image','dtype':'float32'}
    fn({'rows':rows,'provenance':expected},rows,expected)
    with pytest.raises(ValueError,match='provenance'):
        fn({'rows':rows,'provenance':expected | {'freeze':'b'}},rows,expected)


def test_final_intervals_repair_after_last_record_without_rankings(tmp_path):
    fn=workflow()['finalize_intervals']
    records=[]
    for c in ('native_768','mrl_256','mrl_128'):
        records.append({'condition':c,'retrieval':{d:{'per_parent':{'1':{'hit@10':.5},
            '2':{'hit@10':.2}}} for d in ('t2i','i2t')}})
    assert fn(tmp_path,records,[1,2]) == 4
    before=(tmp_path/'paired_intervals.json').read_bytes()
    assert fn(tmp_path,records,[1,2]) == 0
    assert before == (tmp_path/'paired_intervals.json').read_bytes()


def test_saved_contract_repeat_survives_interruption_and_binds_freeze(tmp_path):
    fn=workflow()['saved_repeats']
    assert fn(tmp_path,'a') == {}
    from embedding_diagnostics.embedding_cache import atomic_json
    atomic_json(tmp_path/'smoke_checks.json',{'freeze_sha256':'a','checks':
                {'image':{'equivalent':True}}})
    assert fn(tmp_path,'a')['image']['equivalent']
    with pytest.raises(ValueError):
        fn(tmp_path,'b')


def test_actual_completed_extraction_loads_no_model_and_preserves_chunks(tmp_path,monkeypatch):
    import zipfile
    from types import SimpleNamespace

    import numpy as np

    from embedding_diagnostics.embedding_cache import atomic_json
    from embedding_diagnostics.paired_cache import PairedCache
    from embedding_diagnostics.source_sensitivity import digest
    functions=workflow()
    extract=functions['extract']
    context=extract.__globals__
    root=tmp_path/'experiment'
    root.mkdir()
    atomic_json(root/'freeze.json',{'input_sha256':'input'})
    freeze={'input_sha256':'input','protected_sha256':{}}
    monkeypatch.setitem(context,'ROOT',root)
    monkeypatch.setitem(context,'SOURCE',tmp_path)
    monkeypatch.setitem(context,'checked_freeze',lambda:freeze)
    # This synthetic resume test checks model/cache behavior, not Linux telemetry.
    monkeypatch.setitem(context,'swap_counters',lambda:0)
    monkeypatch.setitem(context,'resource',SimpleNamespace(
        RUSAGE_SELF=0,getrusage=lambda _:SimpleNamespace(ru_maxrss=128*1024)))
    def forbidden(*args,**kwargs):
        raise AssertionError('completed extraction loaded model')
    monkeypatch.setitem(context,'EmbeddingGemma2',forbidden)
    roles=('image','query','document')
    inputs={r:[{'id':1,'parent_image_id':1,'role':r,'input_sha256':'a'*64}] for r in roles}
    atomic_json(root/'input_manifest.json',inputs)
    atomic_json(root/'sample_manifest.json',{'selected_rows':[]})
    atomic_json(root/'smoke_metrics.json',{'resource_gate_passed':True,
       'roles':{r:{'completed':n} for r,n in [('image',16),('query',80),('document',80)]},
       'repeat_checks':{r:{'equivalent':True} for r in roles}})
    with zipfile.ZipFile(tmp_path/'val2017.zip','w'):
        pass
    before={}
    for r in roles:
        path=root/f'{r}/embedding-cache'
        with PairedCache(path,functions['cache_provenance'](r,freeze),inputs[r]) as cache:
            cache.append(np.ones((1,768),np.float32)/np.sqrt(np.float32(768)))
        before.update({str(p):digest(p) for p in path.glob('*') if p.name != '.lock'})
    extract(smoke=False)
    result=functions['read'](root/'extraction_metrics.json')
    assert result['new_inferences'] == 0 and not result['model_loaded']
    assert all(digest(Path(p)) == h for p,h in before.items())


def test_protected_historical_hash_failure_is_loud(tmp_path):
    from embedding_diagnostics.source_sensitivity import digest, verify_hashes
    path=tmp_path/'historical.json'
    path.write_text('historical')
    expected={str(path):digest(path)}
    verify_hashes(expected)
    path.write_text('edited')
    with pytest.raises(ValueError):
        verify_hashes(expected)
