import numpy as np

from embedding_diagnostics.phase_c_main import evaluate_main
from embedding_diagnostics.phase_c_pilot import select_sample, support


def test_main_eligibility_preserves_all_fit_and_retrieval_rows(monkeypatch):
    from embedding_diagnostics import phase_c_main as module
    calls = []
    def fake(train, y, val, labels, **options):
        calls.append((len(train),len(val),options))
        return {'accuracy': .5}
    monkeypatch.setattr(module,'linear_probe_scores',fake)
    labels = [0]*21 + [1]*19 + [0]*12 + [1]*12
    rows = [dict(image_id=str(i),split='train' if i<40 else 'val',
                 weather=y,scene=y,timeofday=y) for i,y in enumerate(labels)]
    sample = {'selected_rows':rows,'support':support(rows),
              'eligibility_rule':{'training_support':20,'validation_support':10},
              'eligible_probe_class_codes':{a:[0] for a in ('weather','scene','timeofday')}}
    x = np.random.default_rng(0).normal(size=(64,4))
    result = evaluate_main(x[:40],x[40:],sample)
    assert result['primary_probe'] == 'standardized'
    assert len(calls) == 6
    assert all(ntrain==40 and nval==24 and opt['eligible_classes']==(0,)
               for ntrain,nval,opt in calls)
    assert max(result['probe_configs']['unscaled']['c_grid']) == 1e6


def test_main_sampling_disjoint_from_pilot():
    rows = [dict(image_id=f'{split}-{i}',split=split,weather='clear',scene='highway',
                 timeofday='daytime') for split in ('train','val') for i in range(60)]
    pilot = select_sample(rows,seed=0,train_count=12,val_count=8)
    excluded = {r['image_id'] for r in pilot}
    main = select_sample([r for r in rows if r['image_id'] not in excluded],seed=1,
                         train_count=24,val_count=12)
    assert excluded.isdisjoint(r['image_id'] for r in main)
    assert len({r['image_id'] for r in main}) == 36


def test_uncommitted_freeze_rejected_before_model_loading(tmp_path,monkeypatch):
    import json
    import runpy
    import subprocess
    from pathlib import Path

    import pytest
    namespace = runpy.run_path(str(Path(__file__).parents[1]/'scripts/run_c1_main.py'))
    checked = namespace['checked_freeze']
    checked.__globals__['ROOT'] = tmp_path
    (tmp_path/'freeze.json').write_text(json.dumps({'files_sha256':{},'runtime_versions':{}}))
    def uncommitted(*args,**kwargs):
        raise subprocess.CalledProcessError(128,'git show')
    monkeypatch.setattr(subprocess,'check_output',uncommitted)
    with pytest.raises(ValueError,match='not committed'):
        checked()


def test_effects_keep_primary_probe_and_exploratory_scope_distinct():
    from embedding_diagnostics.phase_c_main import paired_effects
    pristine = {'geometry':{s:{'total_variance':1.} for s in ('train','val')},
                'attributes':{'weather':{
                    'retrieval_p10':.6,
                    'probe_standardized':{'accuracy':.8,'balanced_accuracy':.7},
                    'probe_unscaled':{'accuracy':.9,'balanced_accuracy':.6}}}}
    record = {'transform':'mean_interpolation','severity':.99,
              'geometry':{s:{'total_variance':.0001} for s in ('train','val')},
              'attributes':{'weather':{
                  'retrieval_p10':.6,
                  'probe_standardized':{'accuracy':.8,'balanced_accuracy':.7},
                  'probe_unscaled':{'accuracy':.7,'balanced_accuracy':.4}}}}
    result = paired_effects(pristine,[record])
    effect = result['paired_effects'][0]
    assert effect['status']=='exploratory' and effect['hypothesis'] is None
    assert effect['attributes']['weather']['standardized_accuracy_delta']==0
    assert effect['attributes']['weather']['unscaled_accuracy_delta'] < 0
    assert result['uncertainty']['confidence_intervals'] is None


def test_main_identity_ignores_result_worktree_dirtiness(monkeypatch):
    import runpy
    import subprocess
    from pathlib import Path
    namespace = runpy.run_path(str(Path(__file__).parents[1]/'scripts/run_c1_main.py'))
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **k: 'frozen-commit\n')
    identity = namespace['main_identity']({'files_sha256': {'a': 'digest'}}, b'freeze')
    assert identity['commit'] == 'frozen-commit'
    assert 'worktree_dirty' not in identity


def test_main_cache_requires_current_extraction_freeze():
    import runpy
    from pathlib import Path

    import pytest
    namespace = runpy.run_path(str(Path(__file__).parents[1]/'scripts/run_c1_main.py'))
    sample = {'selected_rows': [{'image_id': 'one'}]}
    identity = {'commit': 'one', 'main_freeze_sha256': 'current'}
    manifest = {'selected_rows': sample['selected_rows'],
                'provenance': {'code': {'commit': 'one', 'main_freeze_sha256': 'old'}}}
    with pytest.raises(ValueError, match='cache extraction freeze'):
        namespace['validate_main_cache'](manifest, sample, identity)
    manifest['provenance']['code'] = identity
    namespace['validate_main_cache'](manifest, sample, identity)
