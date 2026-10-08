"""Cache-only C2 compression and model-independent endpoints.

Matryoshka prefix+renormalization follows Kusupati et al., Matryoshka
Representation Learning (NeurIPS 2022, arXiv:2205.13147), and the pinned
EmbeddingGemma 2 model card. Tests validate wiring, not the published research
claims. Centered PCA delegates to sklearn's full deterministic SVD solver.
Participation ratio is (sum covariance eigenvalues)^2/sum squared eigenvalues;
RankMe uses the existing raw singular-value entropy definition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.decomposition import PCA

from .bdd100k import ATTRIBUTE_VOCAB
from .diagnostics.metrics import collapse_metrics
from .diagnostics.probe import ProbeConfig, linear_probe_scores, retrieval_chance
from .diagnostics.retrieval import retrieval_macro_precision_at_k, retrieval_precision_at_k
from .phase_c_evaluation import json_safe

DIMENSIONS = (512, 256, 128)


def _matrix(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError('expected nonempty finite embedding matrix')
    return x


def _normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError('zero/nonfinite projected norm; no silent removal')
    return x / norms


def matryoshka(canonical, dimension):
    """Official learned leading dimensions, followed by mandatory L2 normalization.

    Native 768 returns an exact copy of canonical vectors, retaining their
    validated FP32 rounding. Compressed arithmetic/output use float64; this is
    analysis arithmetic on cached FP32 vectors, not a change in model inference.
    """
    x = _matrix(canonical)
    if x.shape[1] != 768 or dimension not in (768, *DIMENSIONS):
        raise ValueError('unsupported native/Matryoshka dimension')
    if np.any(np.linalg.norm(x, axis=1) == 0):
        raise ValueError('zero canonical norm')
    if not np.allclose(np.linalg.norm(x, axis=1), 1, rtol=1e-5, atol=1e-6):
        raise ValueError('canonical vectors must be unit normalized')
    if dimension == 768:
        return np.asarray(canonical).copy()
    return _normalize(x[:, :dimension])


@dataclass(frozen=True)
class TrainPCA:
    """One centered training-only basis, shared across splits and dimensions."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    training_rows: int

    @classmethod
    def fit(cls, train, dimension):
        x = _matrix(train)
        if dimension < 1 or dimension > min(len(x)-1, x.shape[1]):
            raise ValueError('invalid PCA dimension')
        estimator = PCA(n_components=dimension, svd_solver='full', whiten=False).fit(x)
        return cls(estimator.mean_.copy(), estimator.components_.copy(),
                   estimator.explained_variance_.copy(), len(x))

    def transform(self, x, dimension):
        x = _matrix(x)
        if x.shape[1] != len(self.mean) or not 1 <= dimension <= len(self.components):
            raise ValueError('projection dimension mismatch')
        return _normalize((x-self.mean)@self.components[:dimension].T)


def geometry_with_capacity(x):
    """Historical raw metrics plus descriptive fractions of nominal D capacity.

    D is not a claim of fitted rank. Fractions do not replace historical RankMe
    or covariance participation ratio, and do not define health thresholds.
    The inherited RankMe epsilon can marginally exceed nominal capacity;
    fractions are left unclipped to preserve its existing definition.
    """
    x = _matrix(x)
    metrics = collapse_metrics(x)
    return metrics | {'nominal_dimension': x.shape[1],
                      'rankme_fraction': metrics['rankme']/x.shape[1],
                      'participation_ratio_fraction': metrics['participation_ratio']/x.shape[1]}


def validate_source(manifest, sample, manifest_digest, freeze):
    """Bind loaded, integrity-checked cache to the exact frozen C1 source."""
    if manifest['selected_rows'] != sample['selected_rows']:
        raise ValueError('canonical sample alignment mismatch')
    if manifest_digest != freeze['canonical_cache_manifest_sha256']:
        raise ValueError('canonical cache manifest hash mismatch')


def evaluate_representation(train, val, sample):
    """C1 standardized protocol and eligibility, with all retrieval rows retained."""
    train, val = _matrix(train), _matrix(val)
    a = [r for r in sample['selected_rows'] if r['split'] == 'train']
    b = [r for r in sample['selected_rows'] if r['split'] == 'val']
    if len(a) != len(train) or len(b) != len(val) or train.shape[1] != val.shape[1]:
        raise ValueError('physical split/dimension alignment mismatch')
    output = {'geometry': {'train': geometry_with_capacity(train),
                           'val': geometry_with_capacity(val)},
              'probe_config': asdict(ProbeConfig()), 'primary_probe': 'standardized',
              'support': sample['support'], 'eligibility_rule': sample['eligibility_rule'],
              'attributes': {}}
    for attr, vocab in ATTRIBUTE_VOCAB.items():
        y, labels = np.array([r[attr] for r in a]), np.array([r[attr] for r in b])
        eligible = tuple(sample['eligible_probe_class_codes'][attr])
        macro, per_class = retrieval_macro_precision_at_k(val, labels, min_support=10)
        output['attributes'][attr] = {
            'probe_standardized': linear_probe_scores(train, y, val, labels,
                                                     eligible_classes=eligible),
            'eligible_probe_classes': [vocab[i] for i in eligible],
            'balanced_constant_class_floor': 1/len(eligible) if eligible else None,
            'retrieval_p10': retrieval_precision_at_k(val, labels),
            'retrieval_macro_p10': macro,
            'retrieval_per_class_p10': {vocab[i]: value for i, value in per_class.items()},
            'retrieval_macro_support_rule': 'validation support >=10; all rows remain in gallery',
            'retrieval_chance': retrieval_chance(labels),
            'retrieval_chance_convention': 'C1/Phase B sum of squared validation frequencies',
        }
    return json_safe(output)
