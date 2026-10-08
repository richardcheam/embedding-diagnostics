"""Exact paired retrieval and image-group conditional uncertainty.

Recorded COCO caption correspondences (Chen et al., arXiv:1504.00325v2) are
incomplete relevance judgments. Recall/Hit@K is a standard retrieval endpoint;
the category comparator is a declared coarse-label workload, not the same task.
Bootstrap intervals condition on fixed galleries and approximate image-group
independence; they do not quantify encoder or gallery-sampling uncertainty.
"""

from __future__ import annotations

import numpy as np


def _normalize(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("finite embedding matrix required")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero vector")
    return x/norms


def rankings(query, gallery, query_ids, gallery_ids, *, k=10, self_exclude=False,
             block_size=64, tie_depths=None):
    """FP64 normalize/dot, descending score then ascending original numeric ID."""
    q, g = _normalize(query), _normalize(gallery)
    qids, gids = np.asarray(query_ids), np.asarray(gallery_ids)
    if len(q) != len(qids) or len(g) != len(gids) or q.shape[1] != g.shape[1]:
        raise ValueError("row/dimension alignment mismatch")
    if len(set(qids)) != len(qids) or len(set(gids)) != len(gids):
        raise ValueError("unique IDs required within each modality")
    if qids.dtype.kind not in "iu" or gids.dtype.kind not in "iu":
        raise ValueError("numeric original IDs required")
    if k <= 0 or k > len(g)-int(self_exclude) or block_size < 1:
        raise ValueError("invalid neighbour budget")
    result, ties = [], 0
    boundaries={str(d):0 for d in tie_depths} if tie_depths else {}
    if any(d<1 or d>k for d in (tie_depths or [])):
        raise ValueError("invalid tie boundary")
    for start in range(0,len(q),block_size):
        score = q[start:start+block_size] @ g.T
        if score.dtype != np.float64:
            raise ValueError("score accumulation must be FP64")
        for local, row in enumerate(score):
            if self_exclude:
                row[gids == qids[start+local]] = -np.inf
            order = np.lexsort((gids,-row))
            if len(g) > k and row[order[k-1]] == row[order[k]]:
                ties += 1
            for d in tie_depths or []:
                if len(g)>d and row[order[d-1]] == row[order[d]]:
                    boundaries[str(d)]+=1
            result.append(order[:k])
    return np.asarray(result), boundaries if tie_depths else ties


def paired_scores(top, query_parents, gallery_parents, *, ks=(1,5,10)):
    """Per-query hits/set recall and equal-parent aggregation, no inferred positives."""
    top = np.asarray(top)
    q, g = np.asarray(query_parents), np.asarray(gallery_parents)
    if top.ndim != 2 or len(top) != len(q) or top.max() >= len(g) or top.min() < 0:
        raise ValueError("relevance row alignment mismatch")
    if any(np.sum(g == parent) == 0 for parent in q):
        raise ValueError("query without a recorded positive")
    matches = g[top] == q[:,None]
    count = np.array([np.sum(g == parent) for parent in q])
    per_query = {}
    for k in ks:
        if k > top.shape[1]:
            raise ValueError("ranking depth insufficient")
        per_query[f"hit@{k}"] = np.any(matches[:,:k],axis=1).astype(float)
        per_query[f"set_recall@{k}"] = np.sum(matches[:,:k],axis=1)/count
    parents = sorted(set(q.tolist()))
    per_parent = {str(p): {name:float(v[q == p].mean()) for name,v in per_query.items()}
                  for p in parents}
    aggregate = {name:float(np.mean([r[name] for r in per_parent.values()]))
                 for name in per_query}
    return aggregate | {"per_parent":per_parent,
                        "per_query":{k:v.tolist() for k,v in per_query.items()},
                        "query_count":len(q),"gallery_count":len(g)}


def category_scores(top, labels, category_ids, *, min_support=10):
    """Category-macro precision, fixed support rule, natural self-excluded gallery."""
    top = np.asarray(top)
    if len(top) != len(labels) or top.max() >= len(labels):
        raise ValueError("category row alignment mismatch")
    records, eligible = {}, []
    for c in category_ids:
        membership = np.array([c in row for row in labels])
        support = int(membership.sum())
        score = membership[top].mean(axis=1)
        records[str(c)] = {"support":support,"chance":(support-1)/(len(labels)-1)
                          if support else None,
                          "precision":float(score[membership].mean()) if support else None}
        if support >= min_support:
            eligible.append(c)
    k = top.shape[1]
    return {"eligible":eligible,"categories":records,
            f"macro_precision@{k}":float(np.mean([records[str(c)]["precision"]
                                                  for c in eligible])) if eligible else None,
            "macro_chance":float(np.mean([records[str(c)]["chance"]
                                           for c in eligible])) if eligible else None}


def paired_interval(reference, condition, *, resamples=5000, seed=20261010):
    """Paired parent-unit percentile interval, fixed-gallery score resampling."""
    a, b = np.asarray(reference,dtype=float), np.asarray(condition,dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) == 0 or not np.isfinite(a+b).all():
        raise ValueError("aligned finite parent scores required")
    delta = b-a
    rng = np.random.Generator(np.random.PCG64(seed))
    values = []
    for start in range(0,resamples,256):
        draw = rng.integers(0,len(a),size=(min(256,resamples-start),len(a)))
        values.extend(delta[draw].mean(axis=1).tolist())
    lo, hi = np.quantile(values,[.025,.975])
    return {"delta":float(delta.mean()),"lower":float(lo),"upper":float(hi)}
