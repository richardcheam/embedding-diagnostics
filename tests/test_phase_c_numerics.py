"""Model-free wiring checks for bounded C3; no scientific endpoint replication."""
import runpy
from pathlib import Path

import numpy as np
import pytest


def module():
    from embedding_diagnostics import phase_c_numerics
    return phase_c_numerics


def test_int8_uses_train_scales_and_clips_validation_without_refitting():
    m = module()
    quantizer = m.TrainInt8.fit(np.array([[127., 0.], [-63.5, 0.]]))
    np.testing.assert_array_equal(quantizer.scales, [1., 1.])
    packed, info = quantizer.encode(np.array([[200., -200.], [1.5, 2.5]]))
    np.testing.assert_array_equal(packed, [[127, -127], [2, 2]])
    assert packed.dtype == np.int8
    assert info['clipped_coordinates'] == 2
    assert info['clipped_rows'] == 1
    np.testing.assert_array_equal(quantizer.scales, [1., 1.])
    np.testing.assert_array_equal(quantizer.decode(packed), [[127., -127.], [2., 2.]])


def test_invalid_calibration_and_zero_reconstruction_fail():
    m = module()
    with pytest.raises(ValueError):
        m.TrainInt8.fit(np.array([[np.nan, 1.]]))
    with pytest.raises(ValueError, match='zero'):
        m.normalize(np.zeros((1, 2)), np.float64)


def test_tied_topk_uses_ids_and_excludes_self_by_identity():
    m = module()
    scores = np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
    indices, margins, ties = m.neighbours(scores, ['c', 'a', 'b'], 1)
    np.testing.assert_array_equal(indices[:, 0], [1, 2, 1])
    np.testing.assert_array_equal(margins, [0., 0., 0.])
    np.testing.assert_array_equal(ties, [True, True, True])
    assert np.all(indices[:, 0] != np.arange(3))


def test_bins_keep_ties_together_and_allow_fewer_bins():
    m = module()
    ids, groups = m.margin_bins(np.array([0., 0., 1., 1., 1., 1., 2., 2.]))
    assert len(groups) <= 3
    assert ids[0] == ids[1] and len(set(ids[2:6])) == 1 and ids[6] == ids[7]
    again = m.margin_bins(np.array([0., 0., 1., 1., 1., 1., 2., 2.]))
    np.testing.assert_array_equal(ids, again[0])
    one, groups = m.margin_bins(np.zeros(10))
    assert len(groups) == 1 and np.all(one == 0)


def test_float32_normalization_and_accumulation_are_distinct():
    m = module()
    raw = np.array([[1e8, 1., -1e8], [1., 1., 1.]], dtype=np.float32)
    z32 = m.normalize(raw, np.float32)
    z64 = m.normalize(raw, np.float64)
    assert z32.dtype == np.float32 and z64.dtype == np.float64
    assert m.cosine_scores(z32, z32).dtype == np.float32
    assert m.cosine_scores(z64, z64).dtype == np.float64
    audit = m.arithmetic_audit()
    assert audit['fp32_accumulation_witness'] != audit['fp64_accumulation_witness']


def test_margin_sufficiency_is_strict_not_necessary():
    m = module()
    prospective, retrospective = m.margin_checks(
        np.array([.3, .2, 0.]), np.array([.1, .1, 0.]), np.array([.01, .1, 0.]))
    np.testing.assert_array_equal(prospective, [True, False, False])
    np.testing.assert_array_equal(retrospective, [True, False, False])


def test_turnover_is_entering_departing_counts_not_pairings():
    m = module()
    reference = np.array([[0, 1], [0, 2]])
    changed = np.array([[2, 3], [0, 3]])
    result = m.attribute_turnover(reference, changed, np.array([0, 1, 1, 0]),
                                  np.array([0, 1]))
    assert result['entering_counts'] == [{'0': 1, '1': 1}, {'0': 1}]
    assert result['departing_counts'] == [{'0': 1, '1': 1}, {'1': 1}]
    np.testing.assert_array_equal(result['p10_delta'], [0., -.5])
    assert result['label_multiset_unchanged'] == [True, False]


def test_source_rows_must_match_exact_cache_order_and_physical_splits():
    m = module()
    rows = [dict(image_id='a', split='train'), dict(image_id='b', split='val')]
    with pytest.raises(ValueError, match='alignment'):
        m.split_rows(np.ones((2, 3)), {'selected_rows': rows},
                     {'selected_rows': rows[::-1]})
    with pytest.raises(ValueError, match='duplicate'):
        m.split_rows(np.ones((2, 3)), {'selected_rows': rows*2},
                     {'selected_rows': rows*2})


def test_stress_reuses_training_offset_without_validation_fit():
    m = module()
    train = np.array([[1., 0.], [0., 1.]])
    val = np.array([[10., 2.], [5., -3.]])
    a, b = m.mean_stress(train, val)
    np.testing.assert_allclose(b-val, np.broadcast_to(a[0]-train[0], val.shape))
    np.testing.assert_allclose(np.linalg.norm(a[0]-train[0]), 99.)
    _, changed = m.mean_stress(train, val*10)
    np.testing.assert_allclose(changed-val*10, b-val)


def test_score_and_label_alignment_fail_loudly():
    m = module()
    with pytest.raises(ValueError):
        m.neighbours(np.eye(3), ['a', 'a', 'b'], 1)
    with pytest.raises(ValueError):
        m.attribute_turnover(np.array([[0]]), np.array([[1]]), np.array([0]), np.array([0]))


def test_resume_rejects_identity_or_record_tampering_without_overwrite(tmp_path):
    m = module()
    p = tmp_path/'results.json'
    identity = {'freeze': 'one'}
    state = m.resume_results(p, identity)
    record = {'condition': 'native_ref', 'value': 1}
    m.append_record(p, state, record)
    before = p.read_bytes()
    assert len(m.resume_results(p, identity)['records']) == 1
    with pytest.raises(ValueError, match='provenance'):
        m.resume_results(p, {'freeze': 'two'})
    p.write_text(before.decode().replace('"value": 1', '"value": 2'))
    with pytest.raises(ValueError, match='checksum'):
        m.resume_results(p, identity)


def test_completed_resume_calls_no_evaluator_and_preserves_historical_files(tmp_path):
    m = module()
    old = tmp_path/'historical.json'
    old.write_bytes(b'accepted')
    output = tmp_path/'results.json'
    identity = {'one': 'two'}
    state = m.resume_results(output, identity)
    m.append_record(output, state, {'condition': 'native_ref', 'value': 1})
    before = output.read_bytes()
    def forbidden(_):
        pytest.fail('completed condition evaluated again')
    m.run_conditions(output, identity, ['native_ref'], forbidden)
    assert output.read_bytes() == before and old.read_bytes() == b'accepted'


def test_c3_uncommitted_freeze_rejected(tmp_path, monkeypatch):
    ns = runpy.run_path(str(Path(__file__).parents[1]/'scripts/run_c3.py'))
    checked = ns['checked_freeze']
    checked.__globals__['ROOT'] = tmp_path
    (tmp_path/'freeze.json').write_text('{}')
    monkeypatch.setitem(checked.__globals__, 'committed_bytes', lambda path: b'different')
    with pytest.raises(ValueError, match='committed'):
        checked()


def test_fixed_condition_evaluation_preserves_inputs_and_bounds_semantic_change():
    m = module()
    rng = np.random.default_rng(12)
    train = rng.normal(size=(20, 8)).astype(np.float32)
    val = rng.normal(size=(16, 8)).astype(np.float32)
    before = val.copy()
    rows = [dict(image_id=f'id{i:02}', weather=i % 2, scene=i % 3, timeofday=i % 2)
            for i in range(len(val))]
    quantizer = m.TrainInt8.fit(train)
    for name in m.CONDITIONS[:6]:
        r = m.evaluate_condition(name, val, val, rows, quantizer)
        assert r['query_ids'] == [row['image_id'] for row in rows]
        assert r['summary']['prospective_empirical_violations'] in (0, None)
        assert r['score_dtype'] == ('float32' if name.endswith('arithmetic') else 'float64')
        overlap = np.array(r['per_query']['overlap'])
        for attr in m.ATTRIBUTES:
            delta = np.abs(r['attributes'][attr]['per_query']['p10_delta'])
            assert np.all(delta <= 1-overlap+1e-15)
    np.testing.assert_array_equal(val, before)


def report_fixture(tmp_path):
    import hashlib

    from embedding_diagnostics.embedding_cache import atomic_json
    m = module()
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(12, 8))
    rows = [dict(image_id=str(i), weather=i % 2, scene=i % 2, timeofday=i % 2)
            for i in range(12)]
    atomic_json(tmp_path/'freeze.json', {'conditions': list(m.CONDITIONS),
                'dimension': 8, 'validation_count': 12})
    identity = {'freeze_sha256': hashlib.sha256((tmp_path/'freeze.json').read_bytes()).hexdigest()}
    state = m.resume_results(tmp_path/'results.json', identity)
    for name in m.CONDITIONS:
        r = m.evaluate_condition(name, raw, raw, rows, m.TrainInt8.fit(raw))
        r['wall_seconds'] = 0.
        m.append_record(tmp_path/'results.json', state, r)
    atomic_json(tmp_path/'verification.json', {'wall_seconds': 0., 'peak_rss_mib': 1.,
                'historical_files_preserved': 2, 'this_run_endpoint_calls': 8,
                'condition_records': 8, 'identity': identity, 'validation_rows': 12,
                'historical_reference_p10_deltas': {}})
    return state


def test_report_is_generated_from_records(tmp_path):
    report_fixture(tmp_path)
    ns = runpy.run_path(str(Path(__file__).parents[1]/'scripts/report_c3.py'))
    ns['render'](tmp_path)
    report = (tmp_path/'results.md').read_text()
    assert 'native_ref' in report and 'Margin' in report
    assert '1.000000' in report and '12 queries' in report


@pytest.mark.parametrize('change', ['checksum', 'incomplete', 'verification', 'freeze'])
def test_report_rejects_tampered_or_incomplete_evidence(tmp_path, change):
    import json

    from embedding_diagnostics.embedding_cache import atomic_json
    state = report_fixture(tmp_path)
    if change == 'checksum':
        state['records'][0]['payload']['wall_seconds'] = 99.
        atomic_json(tmp_path/'results.json', state)
    elif change == 'incomplete':
        state['records'].pop()
        atomic_json(tmp_path/'results.json', state)
    elif change == 'verification':
        v = json.loads((tmp_path/'verification.json').read_text())
        v['identity'] = {'freeze_sha256': 'wrong'}
        atomic_json(tmp_path/'verification.json', v)
    else:
        (tmp_path/'freeze.json').write_text('{}')
    ns = runpy.run_path(str(Path(__file__).parents[1]/'scripts/report_c3.py'))
    with pytest.raises(ValueError):
        ns['render'](tmp_path)
    assert not (tmp_path/'results.md').exists()


def test_c3_source_and_runtime_mismatch_are_rejected(tmp_path, monkeypatch):
    import json
    ns = runpy.run_path(str(Path(__file__).parents[1]/'scripts/run_c3.py'))
    checked = ns['checked_freeze']
    checked.__globals__['ROOT'] = tmp_path
    file = tmp_path/'source.py'
    file.write_text('changed')
    payload = json.dumps({'files_sha256': {str(file): 'wrong'},
                          'historical_sha256': {}, 'runtime_versions': {}}).encode()
    (tmp_path/'freeze.json').write_bytes(payload)
    monkeypatch.setitem(checked.__globals__, 'committed_bytes', lambda path: payload)
    with pytest.raises(ValueError, match='frozen file changed'):
        checked()


def test_range_exceedance_is_distinct_from_actual_clipping():
    m = module()
    q = m.TrainInt8.fit(np.array([[127.]]))
    packed, info = q.encode(np.array([[127.2], [127.6]]))
    np.testing.assert_array_equal(packed, [[127], [127]])
    assert info['outside_calibration_coordinates'] == 2
    assert info['clipped_coordinates'] == 1


def test_reference_and_stress_are_fixed_and_bad_conditions_rejected():
    m = module()
    raw = np.ones((12, 8))
    rows = [dict(image_id=str(i), weather=0, scene=0, timeofday=0) for i in range(12)]
    with pytest.raises(ValueError, match='condition'):
        m.evaluate_condition('unplanned', raw, raw, rows, m.TrainInt8.fit(raw))


def test_actual_runner_historical_guard_rejects_changes_without_mutation(tmp_path, monkeypatch):
    ns = runpy.run_path(str(Path(__file__).parents[1]/'scripts/run_c3.py'))
    check = ns['historical_hashes']
    accepted = tmp_path/'C1.json'
    accepted.write_bytes(b'accepted')
    monkeypatch.setitem(check.__globals__, 'historical_paths', lambda: [str(accepted)])
    monkeypatch.setitem(check.__globals__, 'committed_bytes', lambda path, commit: b'accepted')
    assert check()[str(accepted)]
    accepted.write_bytes(b'newer-uncommitted')
    with pytest.raises(ValueError, match='historical file changed'):
        check()
    assert accepted.read_bytes() == b'newer-uncommitted'


def test_runner_enforces_committed_freeze_before_loading_any_cache(tmp_path, monkeypatch):
    ns = runpy.run_path(str(Path(__file__).parents[1]/'scripts/run_c3.py'))
    run = ns['run']
    run.__globals__['ROOT'] = tmp_path
    (tmp_path/'freeze.json').write_text('{}')
    monkeypatch.setitem(run.__globals__, 'committed_bytes', lambda path: b'not-committed')
    def forbidden_cache():
        pytest.fail('cache loaded before commit gate')
    monkeypatch.setitem(run.__globals__, 'source_cache', forbidden_cache)
    with pytest.raises(ValueError, match='committed'):
        run()


def test_execution_backend_drift_is_rejected_before_endpoints(monkeypatch):
    ns = runpy.run_path(str(Path(__file__).parents[1]/'scripts/run_c3.py'))
    validate = ns['validate_execution_runtime']
    monkeypatch.setitem(validate.__globals__, 'backend_identity', lambda: {'threads': 4})
    monkeypatch.setitem(validate.__globals__, 'arithmetic_audit', lambda: {'witness': 1})
    validate({'runtime_backend': {'threads': 4}, 'arithmetic_audit': {'witness': 1}})
    with pytest.raises(ValueError, match='backend'):
        validate({'runtime_backend': {'threads': 8}, 'arithmetic_audit': {'witness': 1}})


def test_int8_per_row_clipping_can_be_related_to_turnover():
    m = module()
    q = m.TrainInt8.fit(np.array([[127., 127.]]))
    _, info = q.encode(np.array([[128., 127.2], [1., 1.]]))
    assert info['per_row_clipped_coordinates'] == [1, 0]
    assert info['per_row_outside_calibration_coordinates'] == [2, 0]
