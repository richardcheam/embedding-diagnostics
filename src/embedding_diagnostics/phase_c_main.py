"""Predeclared C1-main endpoints; standardized probe is primary.

Reuses existing instruments. Participation ratio uses centered covariance
 eigenvalues; RankMe uses raw singular values. No Phase-B health thresholds.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .bdd100k import ATTRIBUTE_VOCAB
from .diagnostics.metrics import collapse_metrics
from .diagnostics.probe import ProbeConfig, linear_probe_scores, retrieval_chance
from .diagnostics.retrieval import retrieval_macro_precision_at_k, retrieval_precision_at_k
from .phase_c_evaluation import json_safe

RAW_CONFIG = ProbeConfig(c_grid=tuple(10.**p for p in range(-2, 7)))


def evaluate_main(train, val, sample):
    rows = sample['selected_rows']
    train_rows = [r for r in rows if r['split'] == 'train']
    val_rows = [r for r in rows if r['split'] == 'val']
    if len(train_rows) != len(train) or len(val_rows) != len(val):
        raise ValueError('physical split alignment mismatch')
    output = {'claim_label': '[measurement]', 'primary_probe': 'standardized',
              'secondary_probe': 'unscaled numerical sensitivity',
              'geometry': {'train': collapse_metrics(train), 'val': collapse_metrics(val)},
              'probe_configs': {'standardized': asdict(ProbeConfig()),
                                'unscaled': asdict(RAW_CONFIG)},
              'support': sample['support'], 'eligibility_rule': sample['eligibility_rule'],
              'attributes': {}}
    for attr, vocab in ATTRIBUTE_VOCAB.items():
        y = np.array([r[attr] for r in train_rows])
        labels = np.array([r[attr] for r in val_rows])
        eligible = tuple(sample['eligible_probe_class_codes'][attr])
        macro, per_class = retrieval_macro_precision_at_k(val, labels, min_support=10)
        output['attributes'][attr] = {
            'probe_standardized': linear_probe_scores(train,y,val,labels,eligible_classes=eligible),
            'probe_unscaled': linear_probe_scores(train,y,val,labels,standardize=False,
                                                  config=RAW_CONFIG,eligible_classes=eligible),
            'eligible_probe_classes': [vocab[i] for i in eligible],
            'retrieval_p10': retrieval_precision_at_k(val,labels),
            'retrieval_macro_p10': macro,
            'retrieval_per_class_p10': {vocab[i]: v for i,v in per_class.items()},
            'retrieval_macro_support_rule': 'validation support >=10; all rows remain in gallery',
            'retrieval_chance': retrieval_chance(labels),
            'retrieval_chance_convention': 'Phase B sum of squared validation frequencies',
        }
    return json_safe(output)


def paired_effects(pristine, records):
    """Paired point effects; no invented pretrained training-seed intervals."""
    output = []
    hypotheses = {'scale_contraction':'H1','mean_injection':'H2',
                  'isotropic_noise':'H3','rank_truncation':'H4'}
    for record in records:
        effects = {'transform':record['transform'],'severity':record['severity'],
                   'hypothesis':hypotheses.get(record['transform']),
                   'status':'predeclared' if record['transform'] in hypotheses else 'exploratory',
                   'geometry_delta':{},'attributes':{}}
        for split in ('train','val'):
            effects['geometry_delta'][split] = {
                k:float(v-pristine['geometry'][split][k])
                for k,v in record['geometry'][split].items()}
            base = pristine['geometry'][split]['total_variance']
            effects.setdefault('variance_ratio',{})[split] = (
                record['geometry'][split]['total_variance']/base if base > 0 else None)
        for attr,scores in record['attributes'].items():
            base = pristine['attributes'][attr]
            effects['attributes'][attr] = {
                'retrieval_p10_delta':scores['retrieval_p10']-base['retrieval_p10'],
                'standardized_accuracy_delta':scores['probe_standardized']['accuracy']-
                                               base['probe_standardized']['accuracy'],
                'standardized_balanced_delta':scores['probe_standardized']['balanced_accuracy']-
                                               base['probe_standardized']['balanced_accuracy'],
                'unscaled_accuracy_delta':scores['probe_unscaled']['accuracy']-
                                          base['probe_unscaled']['accuracy'],
                'unscaled_balanced_delta':scores['probe_unscaled']['balanced_accuracy']-
                                          base['probe_unscaled']['balanced_accuracy'],
            }
        output.append(effects)
    return json_safe({'claim_label':'[measurement]','paired_effects':output,
                      'uncertainty':{'method':'repository paired training-seed Student-t',
                                     'status':'not estimable for one fixed pretrained encoder',
                                     'training_seed_replicates':0,
                                     'confidence_intervals':None}})
