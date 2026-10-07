"""Freeze metadata-only main IDs, excluding the historical pilot entirely."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from embedding_diagnostics.bdd100k import ATTRIBUTE_VOCAB
from embedding_diagnostics.bdd_lance import METADATA_COLUMNS, LanceBdd
from embedding_diagnostics.embedding_cache import atomic_json
from embedding_diagnostics.phase_c_pilot import select_sample, support

ROOT = Path('experiments/phaseC_c1_main')
PILOT = Path('experiments/phaseC_c1_pilot/sample_manifest.json')


def main():
    dataset = LanceBdd('/mnt/hdd/data/datasets/BDD100K-enriched')
    pilot = json.loads(PILOT.read_text())
    excluded = {r['image_id'] for r in pilot['selected_rows']}
    rows = []
    for batch in dataset.dataset.scanner(columns=METADATA_COLUMNS, batch_size=1024,
                                         batch_readahead=1, scan_in_order=True).to_batches():
        rows.extend(r for r in batch.to_pylist() if r['image_id'] not in excluded)
    selected = select_sample(rows, seed=1, train_count=2000, val_count=1000)
    counts = support(selected)
    eligibility = {a: [index for index, name in enumerate(vocab)
                       if counts['train'][a][name] >= 20 and counts['val'][a][name] >= 10]
                   for a, vocab in ATTRIBUTE_VOCAB.items()}
    payload = {
        'status': 'FROZEN IDS; NO MAIN INFERENCE', 'created_at': datetime.now(UTC).isoformat(),
        'dataset': dataset.identity, 'selected_rows': selected, 'support': counts,
        'eligible_probe_class_codes': eligibility,
        'eligible_probe_class_names': {a: [ATTRIBUTE_VOCAB[a][i] for i in ids]
                                      for a, ids in eligibility.items()},
        'eligibility_rule': {'training_support': 20, 'validation_support': 10},
        'protocol': {'seed': 1, 'train_count': 2000, 'val_count': 1000,
                     'selection': 'uniform without replacement from fully-labelled '
                     'image-ID-sorted rows, excluding all 512 pilot IDs; '
                     'PCG64 SeedSequence([1,split_index]), train=0,val=1; global image-ID order'},
        'pilot_manifest_sha256': hashlib.sha256(PILOT.read_bytes()).hexdigest(),
        'excluded_pilot_id_count': len(excluded), 'pilot_overlap': 0,
        'numpy_version': np.__version__,
    }
    if any(r['image_id'] in excluded for r in selected):
        raise ValueError('pilot overlap')
    path = ROOT / 'sample_manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        if any(old[k] != payload[k] for k in payload if k != 'created_at'):
            raise ValueError('frozen main sample changed')
    else:
        atomic_json(path, payload)
    print(json.dumps({'support': counts, 'eligibility': payload['eligible_probe_class_names']},
                     indent=2))


if __name__ == '__main__':
    main()
