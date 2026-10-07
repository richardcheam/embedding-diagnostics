import numpy as np

from embedding_diagnostics.phase_c_pilot import census_rows, select_sample


def test_census_and_uniform_selection():
    rows = [dict(image_id=f'{split}-{i:03}', split=split, weather='clear',
                 scene='tunnel' if i == 0 else 'highway', timeofday='daytime')
            for split in ('train', 'val') for i in range(30)]
    rows.append(dict(image_id='missing', split='train', weather=None,
                     scene='undefined', timeofday='night'))
    census = census_rows(rows)
    assert census['splits']['train']['total_rows'] == 31
    assert census['splits']['train']['fully_labelled_rows'] == 30
    assert census['splits']['train']['unknown']['weather'] == {'<null>': 1}
    assert census['splits']['train']['fully_labelled_counts']['scene']['tunnel'] == 1
    first = select_sample(rows, seed=0, train_count=12, val_count=8)
    assert first == select_sample(list(reversed(rows)), seed=0, train_count=12, val_count=8)
    assert len(first) == len({r['image_id'] for r in first}) == 20
    assert sum(r['split'] == 'train' for r in first) == 12
    assert all(r['weather'] >= 0 for r in first)
    assert [r['image_id'] for r in first] == sorted(r['image_id'] for r in first)


def test_train_fitted_transform_and_scale_invariance():
    from embedding_diagnostics.degradation import TRANSFORMS
    from embedding_diagnostics.phase_c_pilot import fitted_transform
    rng = np.random.default_rng(3)
    train = rng.normal(size=(20, 12))
    val = rng.normal(size=(10, 12))
    for name in TRANSFORMS:
        a, b = fitted_transform(train, val, name, 0)
        np.testing.assert_array_equal(a, train)
        np.testing.assert_array_equal(b, val)
        a, b = fitted_transform(train, val, name, .5)
        np.testing.assert_allclose(a, TRANSFORMS[name](train, .5), atol=1e-12)
        other_a, _ = fitted_transform(train, val * 100, name, .5)
        np.testing.assert_array_equal(a, other_a)
    _, b = fitted_transform(train, val, 'mean_interpolation', .5)
    np.testing.assert_allclose(b, .5 * val + .5 * train.mean(0))


def test_pinned_pool_normalize_sequence_fixture(tmp_path):
    import json
    import runpy
    from pathlib import Path

    import pytest
    verify = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/extract_c1_pilot.py'))[
        'verify_semantics']
    modules = [dict(idx=i, path='1_Pooling' if i == 1 else '', type=f'fixture.{name}')
               for i, name in enumerate(('Transformer', 'Pooling', 'Normalize'))]
    (tmp_path / 'modules.json').write_text(json.dumps(modules))
    (tmp_path / '1_Pooling').mkdir()
    (tmp_path / '1_Pooling/config.json').write_text(json.dumps(
        {'embedding_dimension': 768, 'pooling_mode': 'mean', 'include_prompt': True}))
    assert verify(tmp_path)['module_sequence'] == ['Transformer', 'Pooling', 'Normalize']
    modules.reverse()
    (tmp_path / 'modules.json').write_text(json.dumps(modules))
    with pytest.raises(ValueError, match='sequence'):
        verify(tmp_path)


def test_invalid_sampling_and_interventions():
    import pytest

    from embedding_diagnostics.phase_c_pilot import fitted_transform
    with pytest.raises(ValueError, match='insufficient'):
        select_sample([])
    with pytest.raises(ValueError, match='duplicate'):
        select_sample([{'image_id': 'x'}, {'image_id': 'x'}])
    with pytest.raises(ValueError, match='invalid'):
        fitted_transform(np.ones((2, 3)), np.ones((2, 3)), 'unknown', .5)


def test_heldout_rank_basis_and_noise_scale_are_training_fitted():
    from embedding_diagnostics.phase_c_pilot import fitted_transform
    train = np.diag([4., 3., 2., 1.])
    val = np.diag([1., 2., 30., 40.])
    _, projected = fitted_transform(train, val, 'rank_truncation', .5)
    np.testing.assert_allclose(projected, np.diag([1., 2., 0., 0.]), atol=1e-12)
    rng = np.random.default_rng(0)
    expected_noise = rng.normal(scale=train.std(), size=(8, 4))
    noisy_train, noisy_val = fitted_transform(train, val, 'isotropic_noise', .5)
    np.testing.assert_allclose(noisy_train - train, expected_noise[:4])
    np.testing.assert_allclose(noisy_val - val, expected_noise[4:])
    shifted_train, shifted_val = fitted_transform(train, val, 'mean_injection', .9)
    np.testing.assert_allclose(shifted_val - val, shifted_train - train)


def test_zero_severity_preserves_heldout_directions_outside_train_span():
    from embedding_diagnostics.phase_c_pilot import fitted_transform
    train = np.eye(6)[:2]
    val = np.eye(6)[2:]
    a, b = fitted_transform(train, val, 'rank_truncation', 0)
    np.testing.assert_array_equal(a, train)
    np.testing.assert_array_equal(b, val)
