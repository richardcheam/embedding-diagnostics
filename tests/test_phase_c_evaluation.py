import numpy as np

from embedding_diagnostics.phase_c_evaluation import evaluate_vectors


def test_probe_receives_distinct_physical_splits_and_all_support(monkeypatch):
    from embedding_diagnostics import phase_c_evaluation as module
    calls = []
    def probe(train, train_labels, val, val_labels, **kwargs):
        calls.append((train.copy(), val.copy(), train_labels.copy(), val_labels.copy(), kwargs))
        return {'accuracy': .5, 'balanced_accuracy': float('nan')}
    monkeypatch.setattr(module, 'linear_probe_scores', probe)
    rows = [dict(image_id=str(i), split='train' if i < 12 else 'val',
                 weather=i % 2, scene=i % 2, timeofday=i % 2) for i in range(24)]
    vectors = np.random.default_rng(0).normal(size=(24, 4))
    result = evaluate_vectors(vectors[:12], vectors[12:], rows)
    assert len(calls) == 6
    for train, val, labels, test_labels, options in calls:
        np.testing.assert_array_equal(train, vectors[:12])
        np.testing.assert_array_equal(val, vectors[12:])
        assert len(labels) == len(test_labels) == 12
        assert options['seed'] == 0
    assert result['attributes']['scene']['support']['train']['tunnel'] == 0
    assert result['attributes']['scene']['probe_standardized']['balanced_accuracy'] is None
    assert result['attributes']['scene']['macro_scored_classes'] == []


def test_real_corrected_probes_and_retrieval_on_heldout_synthetic_vectors():
    from threadpoolctl import threadpool_limits
    rng = np.random.default_rng(7)
    labels = np.arange(40) % 2
    vectors = rng.normal(scale=.01, size=(40, 4))
    vectors[:, 0] += 2 * labels - 1
    rows = [dict(image_id=str(i), split='train' if i < 20 else 'val',
                 weather=int(labels[i]), scene=int(labels[i]), timeofday=int(labels[i]))
            for i in range(40)]
    with threadpool_limits(limits=1):
        result = evaluate_vectors(vectors[:20], vectors[20:], rows)
    for scores in result['attributes'].values():
        for key in ('probe_standardized', 'probe_unscaled'):
            assert scores[key]['accuracy'] == scores[key]['balanced_accuracy'] == 1
            assert scores[key]['converged'] == 1
            assert scores[key]['selected_C'] in result['probe_config']['c_grid']
        assert scores['retrieval_p10'] == .9  # nine other examples of the same class
