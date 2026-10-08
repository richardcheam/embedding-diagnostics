# Executed source audit procedure

[ours] The following read-only source/cache checks generated the row ledger, initial audit and numerical image-inspection records. Outputs went to this separate audit directory; the contact sheet initially went to `/tmp`. Subsequent documentation added visual classifications and the upstream API checksum check; the selected-row ledger was compacted to ID, position, fragment, raw hash and cache location. Detailed records are retained separately for the 17 flagged rows. No encoder, scientific evaluator or cache writer was called.

```python
import collections
import hashlib
import io
import json
import subprocess
import urllib.request
from pathlib import Path

import lance
import numpy as np
from PIL import Image, ImageDraw
from embedding_diagnostics.bdd100k import ATTRIBUTE_VOCAB
from embedding_diagnostics.bdd_lance import METADATA_COLUMNS, _encode

ROOT = Path('experiments/phaseC_source_audit')
DATA = Path('/mnt/hdd/data/datasets/BDD100K-enriched')
SOURCE = DATA / 'data/bdd100k.lance'
CACHE = Path('experiments/phaseC_c1_main/embedding-cache')

def sha_file(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

manifest = json.loads((CACHE / 'manifest.json').read_text())
sample = json.loads(Path('experiments/phaseC_c1_main/sample_manifest.json').read_text())
assert manifest['selected_rows'] == sample['selected_rows']
assert manifest == json.loads(Path('experiments/phaseC_c1_main/extraction_manifest.json').read_text())
dataset = lance.dataset(str(SOURCE), version=55)
metadata = dataset.to_table(columns=METADATA_COLUMNS).to_pylist()
counts = collections.Counter(r['image_id'] for r in metadata)
assert max(counts.values()) == 1
positions = {r['image_id']: i for i, r in enumerate(metadata)}
fragments = []
start = 0
for f in dataset.get_fragments():
    fragments.append({'fragment_id': f.fragment_id, 'start': start,
                      'stop': start + f.count_rows(),
                      'image_data_file': f.metadata.files[0].path})
    start += f.count_rows()

records = []
inspections = []
images = []
for chunk in manifest['chunks']:
    path = CACHE / chunk['file']
    assert sha_file(path) == chunk['sha256']
    rows = manifest['selected_rows'][chunk['start']:chunk['stop']]
    with np.load(path, allow_pickle=False) as stored:
        for column in METADATA_COLUMNS:
            assert stored[column].tolist() == [r[column] for r in rows]
    raw = dataset.take([positions[r['image_id']] for r in rows],
                       columns=[*METADATA_COLUMNS, 'image_bytes']).to_pylist()
    for offset, (selected, source, recorded_hash) in enumerate(zip(rows, raw, chunk['image_sha256'], strict=True)):
        mapped = {'image_id': source['image_id'], 'split': source['split'], **_encode(source)}
        assert mapped == selected
        actual_hash = hashlib.sha256(source['image_bytes']).hexdigest()
        assert actual_hash == recorded_hash
        position = positions[selected['image_id']]
        fragment = next(f for f in fragments if f['start'] <= position < f['stop'])
        record = {'canonical_index': chunk['start'] + offset, 'image_id': selected['image_id'],
                  'canonical_metadata': selected, 'source_metadata': {k: source[k] for k in METADATA_COLUMNS},
                  'source_scan_position': position, 'source_fragment_id': fragment['fragment_id'],
                  'source_fragment_offset': position - fragment['start'],
                  'source_image_data_file': fragment['image_data_file'],
                  'cache_chunk': str(path), 'cache_offset': offset,
                  'recorded_image_sha256': recorded_hash, 'source_image_sha256': actual_hash,
                  'byte_count': len(source['image_bytes']), 'identifier_rewritten_by_repository': False}
        records.append(record)
        if selected['image_id'].startswith('synthetic_val_'):
            image = Image.open(io.BytesIO(source['image_bytes']))
            pixels = np.array(image.convert('RGB'))
            inspection = {'image_id': selected['image_id'], 'format': image.format,
                          'size': list(image.size), 'mode': image.mode,
                          'decoded_rgb_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                          'unique_rgb_colors': len(np.unique(pixels.reshape(-1, 3), axis=0)),
                          'pixel_0_0': pixels[0, 0].tolist(),
                          'pixel_center': pixels[pixels.shape[0] // 2, pixels.shape[1] // 2].tolist()}
            inspections.append(inspection)
            thumb = image.convert('RGB'); thumb.thumbnail((320, 180))
            images.append((selected['image_id'], thumb))
assert len(records) == 3000
save('selected_row_ledger.json', records)
save('image_inspection.json', {'purpose': 'Source-content inspection only, no encoder or endpoint computation',
                             'decoded_count': len(inspections), 'rows': inspections})
canvas = Image.new('RGB', (960, 6 * 205), 'white'); draw = ImageDraw.Draw(canvas)
for i, (name, image) in enumerate(images):
    x, y = (i % 3) * 320, (i // 3) * 205
    canvas.paste(image, (x, y)); draw.text((x, y + 181), name, fill='black')
canvas.save('/tmp/source-audit-contact-sheet.png')

# Source chronology is metadata-only. Compare flagged bytes with the first append.
base_ids = set(lance.dataset(str(SOURCE), version=51).to_table(columns=['image_id']).column(0).to_pylist())
first = lance.dataset(str(SOURCE), version=52)
first_metadata = first.to_table(columns=METADATA_COLUMNS).to_pylist()
new_rows = [r for r in first_metadata if r['image_id'] not in base_ids]
assert len(new_rows) == 500
first_positions = {r['image_id']: i for i, r in enumerate(first_metadata)}
flagged = [r for r in records if r['image_id'].startswith('synthetic_val_')]
first_bytes = first.take([first_positions[r['image_id']] for r in flagged], columns=['image_id', 'image_bytes']).to_pylist()
assert all(hashlib.sha256(r['image_bytes']).hexdigest() == old['source_image_sha256']
           for r, old in zip(first_bytes, flagged, strict=True))
version_timestamps = {v['version']: v['timestamp'].isoformat() for v in dataset.versions()}
by_hash = collections.defaultdict(list)
for r in records:
    by_hash[r['source_image_sha256']].append({'image_id': r['image_id'], 'split': r['canonical_metadata']['split']})
duplicates = {h: rs for h, rs in by_hash.items() if len(rs) > 1}
train_ids = {r['image_id'] for r in records if r['canonical_metadata']['split'] == 'train'}
val_ids = {r['image_id'] for r in records if r['canonical_metadata']['split'] == 'val'}
source_train_ids = {r['image_id'] for r in metadata if r['split'] == 'train'}
source_val_ids = {r['image_id'] for r in metadata if r['split'] == 'val'}
campaigns = {}
flagged_ids = {r['image_id'] for r in flagged}
for name in ['phaseC_c0', 'phaseC_c1_pilot', 'phaseC_c1_main']:
    paths = list(Path('experiments', name).glob('*manifest.json'))
    for p in paths:
        value = json.loads(p.read_text())
        if 'selected_rows' in value:
            ids = {r['image_id'] for r in value['selected_rows']}
            campaigns[str(p)] = {'main_flagged_ids_present': sorted(ids & flagged_ids),
                                 'all_synthetic_prefix_ids': sorted(i for i in ids if i.startswith('synthetic_val_'))}
summary = {'claim_label': '[ours]', 'audit_head': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
           'source': manifest['provenance']['dataset'], 'fragments': fragments,
           'metadata_rows': len(metadata), 'source_split_counts': dict(collections.Counter(r['split'] for r in metadata)),
           'source_duplicate_identifier_count': sum(n - 1 for n in counts.values()),
           'source_train_val_identifier_overlap': sorted(source_train_ids & source_val_ids),
           'selected_rows': len(records), 'selected_split_counts': {'train': len(train_ids), 'val': len(val_ids)},
           'selected_metadata_matches': 3000, 'selected_raw_byte_hash_matches': 3000,
           'selected_identifier_rewrite_count': 0, 'selected_identifier_overlap': sorted(train_ids & val_ids),
           'selected_duplicate_raw_byte_groups': duplicates,
           'selected_train_val_raw_byte_overlap_groups': {h: rs for h, rs in duplicates.items() if len({r['split'] for r in rs}) > 1},
           'chunk_checksums_verified': len(manifest['chunks']),
           'selected_fragment_counts': dict(collections.Counter(str(r['source_fragment_id']) for r in records)),
           'first_added_version': 52, 'first_added_timestamp_recorded_by_lance': version_timestamps[52],
           'first_added_row_count': len(new_rows),
           'first_added_all_prefix_and_val': all(r['image_id'].startswith('synthetic_val_') and r['split'] == 'val' for r in new_rows),
           'flagged_first_version_byte_matches': len(first_bytes), 'campaign_membership': campaigns,
           'cache_manifest_sha256': sha_file(CACHE / 'manifest.json'),
           'sample_manifest_sha256': sha_file('experiments/phaseC_c1_main/sample_manifest.json'),
           'extraction_code_identity': manifest['provenance']['code'],
           'prohibited_columns_read': [], 'images_decoded': len(inspections),
           'no_full_source_byte_deduplication': True,
           'audit_script_sha256': sha_file('/tmp/source_provenance_audit.py')}
save('audit.json', summary)
print(json.dumps({k: v for k,v in summary.items() if k not in ['extraction_code_identity','campaign_membership','fragments']}, indent=2))
```
