import numpy as np
import pytest

from embedding_diagnostics.phase_c_compression import (
    TrainPCA,
    geometry_with_capacity,
    matryoshka,
    validate_source,
)


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.mark.parametrize('dimension', [512, 256, 128])
def test_mrl_prefix_is_renormalized_deterministic(dimension):
    x = unit(np.random.default_rng(2).normal(size=(5, 768))).astype(np.float32)
    result = matryoshka(x, dimension)
    assert result.shape == (5, dimension)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-7)
    np.testing.assert_allclose(result, unit(x[:, :dimension].astype(np.float64)), atol=1e-12)
    np.testing.assert_array_equal(result, matryoshka(x, dimension))
    assert not np.allclose(result, x[:, :dimension])


def test_mrl_768_preserves_canonical_exactly_and_rejects_invalid():
    x = unit(np.random.default_rng(3).normal(size=(4, 768))).astype(np.float32)
    np.testing.assert_array_equal(matryoshka(x, 768), x)
    for bad in [127, 0, 1024]:
        with pytest.raises(ValueError):
            matryoshka(x, bad)
    with pytest.raises(ValueError, match='zero'):
        matryoshka(np.zeros((3, 768)), 128)
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        matryoshka(x, 128)


def test_pca_train_mean_shared_projection_and_no_validation_leakage():
    rng = np.random.default_rng(9)
    train = rng.normal(size=(30, 8)) + np.arange(8)
    val = rng.normal(size=(12, 8)) + 20
    pca = TrainPCA.fit(train, 6)
    np.testing.assert_allclose(pca.mean, train.mean(0))
    a, b = pca.transform(train, 4), pca.transform(val, 4)
    np.testing.assert_allclose(b, unit((val-pca.mean)@pca.components[:4].T))
    before = pca.components.copy()
    pca.transform(val*100, 4)
    np.testing.assert_array_equal(pca.components, before)
    np.testing.assert_array_equal(a, pca.transform(train, 4))
    other = TrainPCA.fit(train, 6)
    np.testing.assert_array_equal(other.components, pca.components)
    assert a.shape == (30, 4) and b.shape == (12, 4)
    np.testing.assert_allclose(np.linalg.norm(b, axis=1), 1)
    with pytest.raises(ValueError):
        pca.transform(val, 7)
    with pytest.raises(ValueError):
        TrainPCA.fit(train, 9)


def test_capacity_metrics_keep_historical_raw_values():
    x = np.random.default_rng(0).normal(size=(20, 8))
    result = geometry_with_capacity(x)
    assert result['rankme_fraction'] == result['rankme']/8
    assert result['participation_ratio_fraction'] == result['participation_ratio']/8
    assert result['nominal_dimension'] == 8
    assert 0 <= result['rankme_fraction'] <= 1
    assert 0 <= result['participation_ratio_fraction'] <= 1


def test_source_provenance_mismatch_rejected():
    manifest = {'selected_rows': [{'image_id': 'one'}]}
    sample = {'selected_rows': [{'image_id': 'two'}]}
    with pytest.raises(ValueError, match='sample'):
        validate_source(manifest, sample, 'actual', {'canonical_cache_manifest_sha256': 'actual'})
    sample['selected_rows'] = manifest['selected_rows']
    with pytest.raises(ValueError, match='hash'):
        validate_source(manifest, sample, 'different',
                        {'canonical_cache_manifest_sha256': 'actual'})


def test_c1_rank_uses_train_only_common_projector(monkeypatch):
    from embedding_diagnostics.phase_c_pilot import fitted_transform
    train = np.array([[4., 0, 0], [0, 2, 0], [-4, 0, 0], [0, -2, 0]])
    val = np.array([[1., 0, 100], [0, 1, -100]])
    original = np.linalg.svd
    fitted_inputs = []
    def recording(x, **kwargs):
        fitted_inputs.append(x.copy())
        return original(x, **kwargs)
    monkeypatch.setattr(np.linalg, 'svd', recording)
    a, b = fitted_transform(train, val, 'rank_truncation', .5)
    assert all(np.array_equal(x, train) for x in fitted_inputs)
    assert np.all(b[:, 2] == 0)
    np.testing.assert_allclose(a[:, :2], train[:, :2])
    np.testing.assert_allclose(b[:, :2], val[:, :2])


def test_zero_pca_projection_fails_without_dropping_rows():
    pca = TrainPCA.fit(np.array([[1., 0], [-1, 0], [0, 1], [0, -1]]), 1)
    with pytest.raises(ValueError, match='zero'):
        pca.transform(pca.mean[None, :], 1)


def test_c2_probe_is_standardized_train_only_and_keeps_support(monkeypatch):
    from embedding_diagnostics import phase_c_compression as module
    from embedding_diagnostics.phase_c_pilot import support
    calls = []
    def fake(train, y, val, labels, **options):
        calls.append((train.copy(), val.copy(), options))
        return {'accuracy': .5, 'balanced_accuracy': .5}
    monkeypatch.setattr(module, 'linear_probe_scores', fake)
    rows = [dict(image_id=str(i), split='train' if i < 24 else 'val',
                 weather=i % 2, scene=i % 2, timeofday=i % 2) for i in range(36)]
    sample = {'selected_rows': rows, 'support': support(rows), 'eligibility_rule': 'frozen',
              'eligible_probe_class_codes': {a: [0] for a in ('weather', 'scene', 'timeofday')}}
    train = np.random.default_rng(1).normal(size=(24, 8))
    val = np.random.default_rng(2).normal(size=(12, 8))
    output = module.evaluate_representation(train, val, sample)
    assert len(calls) == 3
    for a, b, opts in calls:
        np.testing.assert_array_equal(a, train)
        np.testing.assert_array_equal(b, val)
        assert opts == {'eligible_classes': (0,)}  # default standardized C1 probe
    assert output['support'] == sample['support']


def test_c2_uncommitted_freeze_is_rejected(tmp_path, monkeypatch):
    import runpy
    import subprocess
    from pathlib import Path
    ns = runpy.run_path(str(Path(__file__).parents[1]/'scripts/run_c2.py'))
    checked = ns['checked_freeze']
    checked.__globals__['ROOT'] = tmp_path
    (tmp_path/'freeze.json').write_text('{}')
    def missing(*args, **kwargs):
        raise subprocess.CalledProcessError(128, 'git show')
    monkeypatch.setattr(subprocess, 'check_output', missing)
    with pytest.raises(ValueError, match='not committed'):
        checked()


def test_c2_frozen_source_mismatch_rejected(tmp_path, monkeypatch):
    import json
    import runpy
    from pathlib import Path
    ns = runpy.run_path(str(Path(__file__).parents[1]/'scripts/run_c2.py'))
    checked = ns['checked_freeze']
    checked.__globals__['ROOT'] = tmp_path
    source = tmp_path/'code.py'
    source.write_text('changed')
    payload = json.dumps({'files_sha256': {str(source): 'old'}, 'runtime_versions': {}}).encode()
    (tmp_path/'freeze.json').write_bytes(payload)
    monkeypatch.setitem(checked.__globals__, 'committed_bytes', lambda path: payload)
    with pytest.raises(ValueError, match='frozen file changed'):
        checked()


def test_capacity_fraction_preserves_inherited_rankme_epsilon():
    result = geometry_with_capacity(np.eye(8))
    assert result['rankme_fraction'] > 1  # existing entropy epsilon, no silent clipping
    assert result['rankme_fraction'] == result['rankme']/8


def test_parameter_resume_preserves_file_and_rejects_mismatch(tmp_path):
    import runpy
    from pathlib import Path
    ns = runpy.run_path(str(Path(__file__).parents[1]/'scripts/run_c2.py'))
    save = ns['save_pca_parameters']
    path = tmp_path/'parameters.npz'
    params = {'mean': np.array([1., 2.]), 'basis': np.eye(2)}
    save(path, params)
    original = path.read_bytes()
    save(path, params)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match='resume mismatch'):
        save(path, params | {'mean': np.array([1., 3.])})
    assert path.read_bytes() == original
