"""Hand-checkable paired relevance, ordering and conditional uncertainty."""

import numpy as np
import pytest

from embedding_diagnostics import paired_evaluation as evaluation


def test_numeric_ties_and_image_self_exclusion():
    q = np.array([[1., 0.]])
    g = np.array([[1.,0.],[1.,0.],[0.,1.]])
    top, ties = evaluation.rankings(q,g,[10],[10,2,3],k=2)
    assert top.tolist() == [[1,0]] and ties == 0
    top, _ = evaluation.rankings(q,g,[10],[10,2,3],k=2,self_exclude=True)
    assert top.tolist() == [[1,2]]
    # Cross-modal ID collision must not remove its recorded positive.
    assert 0 in evaluation.rankings(q,g,[10],[10,2,3],k=2)[0][0]


def test_multiple_positives_hit_is_distinct_from_set_recall():
    top = np.array([[0,2],[2,0]])
    scores = evaluation.paired_scores(top,[1,2],[1,1,2],ks=(1,2))
    assert scores["hit@1"] == 1
    assert scores["set_recall@2"] == .75
    assert scores["per_parent"]["1"]["set_recall@2"] == .5


def test_caption_bundles_have_equal_parent_weight():
    top = np.array([[0],[0],[0],[0],[0],[0]])
    scores = evaluation.paired_scores(top,[1,1,1,1,1,2],[1,2],ks=(1,))
    assert scores["hit@1"] == .5
    assert scores["per_query"]["hit@1"] == [1,1,1,1,1,0]


def test_blocked_rankings_match_dense_order_and_reject_misalignment():
    rng = np.random.default_rng(1)
    q, g = rng.normal(size=(4,3)), rng.normal(size=(8,3))
    gids = [8,3,5,2,4,1,6,7]
    score = (q/np.linalg.norm(q,axis=1,keepdims=True)) @ (
        g/np.linalg.norm(g,axis=1,keepdims=True)).T
    expected = np.array([np.lexsort((gids,-r))[:3] for r in score])
    top,_=evaluation.rankings(q,g,list(range(4)),gids,k=3,block_size=2)
    np.testing.assert_array_equal(top,expected)
    with pytest.raises(ValueError):
        evaluation.rankings(q,g,[1],gids,k=3)
    with pytest.raises(ValueError):
        evaluation.rankings(q,g,list(range(4)),[1]*8,k=3)


def test_bootstrap_is_paired_and_conditional_on_parent_scores():
    reference=np.array([0.,1.,0.,1.])
    same=evaluation.paired_interval(reference,reference,resamples=50)
    assert same == {"delta":0.,"lower":0.,"upper":0.}
    changed=evaluation.paired_interval(reference,reference+.2,resamples=50)
    assert changed["delta"] == pytest.approx(.2)
    assert changed["lower"] == pytest.approx(.2)
    assert evaluation.paired_interval(reference,reference+.2,resamples=50) == changed


def test_category_floor_uses_full_self_excluded_gallery():
    top=np.array([[1],[0],[0]])
    labels=[[1],[1],[2]]
    result=evaluation.category_scores(top,labels,[1,2],min_support=2)
    assert result["eligible"] == [1]
    assert result["macro_precision@1"] == 1
    assert result["categories"]["1"]["chance"] == .5
    assert result["categories"]["2"]["support"] == 1


def test_tie_counts_cover_declared_endpoint_boundaries():
    q=np.array([[1.,0.]])
    g=np.array([[1.,0.]]*12)
    top,ties=evaluation.rankings(q,g,[5],list(range(12)),tie_depths=(1,5,10))
    assert top.tolist() == [list(range(10))]
    assert ties == {'1':1,'5':1,'10':1}
