"""Metadata-only census, followed by immutable uniform C1-pilot selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from embedding_diagnostics.bdd_lance import METADATA_COLUMNS, LanceBdd
from embedding_diagnostics.embedding_cache import atomic_json
from embedding_diagnostics.phase_c_pilot import PROTOCOL, census_rows, select_sample, support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('experiments/phaseC_c1_pilot'))
    args = parser.parse_args()
    dataset = LanceBdd(args.dataset)
    rows = []
    for batch in dataset.dataset.scanner(columns=METADATA_COLUMNS, batch_size=1024,
                                         batch_readahead=1, scan_in_order=True).to_batches():
        rows.extend(batch.to_pylist())
    census = census_rows(rows)
    census.update(dataset=dataset.identity, claim_label='[measurement]',
                  rare_definition='fewer than 100 fully-labelled rows, descriptive only')
    census_path = args.root / 'dataset_census.json'
    if census_path.exists():
        if json.loads(census_path.read_text()) != census:
            raise ValueError('frozen census changed')
    else:
        atomic_json(census_path, census)
    selected = select_sample(rows)
    sample_path = args.root / 'sample_manifest.json'
    if sample_path.exists():
        old = json.loads(sample_path.read_text())
        if old['selected_rows'] != selected or old['dataset'] != dataset.identity \
                or old['protocol'] != PROTOCOL:
            raise ValueError('frozen sample changed')
    else:
        atomic_json(sample_path, {
            'created_at': datetime.now(UTC).isoformat(), 'dataset': dataset.identity,
            'protocol': PROTOCOL, 'selected_rows': selected, 'support': support(selected),
            'numpy_version': np.__version__,
            'census_sha256': hashlib.sha256(census_path.read_bytes()).hexdigest(),
            'protocol_document_sha256': hashlib.sha256(
                Path('docs/phase-c.md').read_bytes()).hexdigest(),
        })
    print(json.dumps(support(selected), indent=2))


if __name__ == '__main__':
    main()
