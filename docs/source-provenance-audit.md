# Bounded source-provenance audit

[ours] Audited accepted HEAD `720df9e0297a02dad054fae96381daf8dd38c1e7`
on 2026-10-08. **All 17 flagged main validation rows contain generated geometric
graphics: flat coloured backgrounds with coloured rectangles.** This classification
comes from inspection of their actual decoded source bytes, not their names.
Their bytes demonstrably came through the pinned enriched dataset; the producing
program, creator, intended purpose and label-assignment procedure remain unknown.
No scientific endpoint was recomputed and no historical result was replaced.

## Evidence and entry into the sample

[ours] The local source has 70,000 training and 10,500 validation rows at Lance
version 55. Version 51 has 80,000 rows; version 52 adds 500 new validation rows,
all named `synthetic_val_000000` through `synthetic_val_000499`, in fragment 2.
The first-append timestamp recorded by Lance is `2026-04-09T18:00:08.247433`
(timezone unspecified by the returned timestamp). The 17 inspected rows have the
same bytes in versions 52 and 55. This append predates the October extraction.

[ours] The local download sidecar records upstream revision
`d82c5188d392714ba8091d68014f7b9838ceadf2`. An independent read of the
[pinned upstream file metadata](https://huggingface.co/api/datasets/lance-format/BDD100K-enriched/tree/d82c5188d392714ba8091d68014f7b9838ceadf2/data/bdd100k.lance/data?expand=true)
confirms that fragment 2's image data file has SHA256
`fb6dfdee886acaf76efbd6be9a1098038eac28004e8e15b38a221d62a84f3098`,
matching the entire local 2,944,032-byte file. Its upstream upload commit is
`28f6f7b5cd1526ceec55082fcc2ef85f3ec9ad4a`, dated 2026-04-10.
Thus the appended content is present in the distributed source, rather than
merely sharing extraction-recorded hashes. The pinned upstream root contains
only `.gitattributes` and `data/`; its README request returned 404. No producer
script, original BDD filename mapping or source-caption field was recovered.
See [upstream verification](../experiments/phaseC_source_audit/upstream_verification.json).

[ours] The repository makes no ID rewrite: [LanceBdd](../src/embedding_diagnostics/bdd_lance.py#L38)
reads `image_id` verbatim and rejects duplicates. The same path handles all
3,000 selected rows, including 2,983 IDs matching the hexadecimal BDD-style
pattern and these 17 upstream IDs. [Main selection](../scripts/prepare_c1_main.py#L20)
excludes all 512 pilot IDs, then [select_sample](../src/embedding_diagnostics/phase_c_pilot.py#L56)
chooses fully labelled rows uniformly without replacement, seed 1, independently
within physical splits, and globally sorts IDs. It does not distinguish generated
content. These rows entered the frozen sample through that ordinary eligibility
and sampling rule, not through a special fallback or a later C3 insertion.

[ours] Historical code at extraction identity `fdc68c1e14004aac45e7c85fa4e05db96eeeda28`
was inspected; seven relevant loader, mapping, sampling, runner and cache files
match their extraction-recorded SHA256 values. The loader was introduced in
`3c6e8e4`; the main-selection script in `fdc68c1`. [decode](../src/embedding_diagnostics/bdd_lance.py#L68)
validates selected metadata/ID order, hashes original bytes, then opens those bytes.
[extract](../src/embedding_diagnostics/phase_c.py#L88) passes decoded images to the
encoder, propagates failures and stores the supplied image hashes. There is no
placeholder substitution. The [cache reader](../src/embedding_diagnostics/embedding_cache.py#L65)
checks chunk hashes and metadata alignment; source-byte continuity is additionally
checked during extraction resume. These checks do not authenticate photographic
origin or label validity.

[ours] Repository fixture paths inspected are confined to tests:
[test_bdd_lance](../tests/test_bdd_lance.py#L14) writes tiny red PNGs under
`tmp_path` with IDs `a`, `z`, `u`, `v`; its monkeypatch is test-local. No generator
for `synthetic_val_` was found in repository production code or Git history of
the ingestion path. The upstream fragment checksum supplies positive evidence
against those local fixtures having supplied the 17 canonical rows. It does not
identify which upstream generation program supplied them.

[ours] The unchanged [extraction manifest](../experiments/phaseC_c1_main/extraction_manifest.json)
records `google/embeddinggemma-2` revision
`914f7f89142e33e77833254d9c9b90c3cef7303b`, text + vision with audio disabled,
CPU FP32, batch 1, four threads, 768 dimensions, masked mean pooling and L2
normalization. Canonical cache-manifest SHA256 is
`c358c13999fefb49b7c3458e27231b9fe7e56834c2ec0df22d0eb73731fd04ad`;
sample-manifest SHA256 is
`62c62eec085ba09f4a056a4451e37391f680a89becc480c55b40f90cc85f6aaf`.
The audit verified input/cache continuity and recorded code identity, without
re-executing the model to independently verify those embeddings.

## Per-row classification

[ours] Every row below has physical split **val**, source fragment **2**, and
classification **4 — confirmed generated graphic content**. Source position is
the zero-based scan/take position at version 55; fragment offset is position
minus 80,000. Cache locations are under
`experiments/phaseC_c1_main/embedding-cache/`, in four-row files named
`chunk-START-STOP.npz`. The full source metadata, label codes, hashes, exact cache
paths/offsets and classification evidence are in
[flagged_rows.json](../experiments/phaseC_source_audit/flagged_rows.json).
The displayed hash prefixes are conveniences; the audit compared full SHA256 values.

| Source ID (= canonical ID) | Source position / fragment offset | Weather / scene / timeofday | Cache chunk start : offset | Raw SHA256 prefix | Class |
| --- | --- | --- | --- | --- | --- |
| `synthetic_val_000005` | 80005 / 5 | clear / gas stations / daytime | 2980 : 3 | `4393a92b5101` | 4 |
| `synthetic_val_000021` | 80021 / 21 | clear / tunnel / night | 2984 : 0 | `40b04cf732eb` | 4 |
| `synthetic_val_000029` | 80029 / 29 | rainy / gas stations / night | 2984 : 1 | `ee4abbc563d5` | 4 |
| `synthetic_val_000048` | 80048 / 48 | snowy / gas stations / daytime | 2984 : 2 | `7f0196d5bc26` | 4 |
| `synthetic_val_000067` | 80067 / 67 | rainy / highway / dawn/dusk | 2984 : 3 | `0540e71518d1` | 4 |
| `synthetic_val_000116` | 80116 / 116 | snowy / highway / night | 2988 : 0 | `3fdab78aa934` | 4 |
| `synthetic_val_000150` | 80150 / 150 | overcast / residential / daytime | 2988 : 1 | `b7a8140b99d9` | 4 |
| `synthetic_val_000189` | 80189 / 189 | snowy / tunnel / daytime | 2988 : 2 | `6af1efca0fe6` | 4 |
| `synthetic_val_000210` | 80210 / 210 | rainy / residential / dawn/dusk | 2988 : 3 | `19943199e1e6` | 4 |
| `synthetic_val_000228` | 80228 / 228 | rainy / residential / night | 2992 : 0 | `cdf1a6d0e774` | 4 |
| `synthetic_val_000234` | 80234 / 234 | snowy / highway / daytime | 2992 : 1 | `6e3c56bc9925` | 4 |
| `synthetic_val_000361` | 80361 / 361 | foggy / residential / night | 2992 : 2 | `f3c2e3662ffc` | 4 |
| `synthetic_val_000362` | 80362 / 362 | rainy / tunnel / night | 2992 : 3 | `803dd461e148` | 4 |
| `synthetic_val_000373` | 80373 / 373 | clear / city street / night | 2996 : 0 | `dbb1a4cd69ac` | 4 |
| `synthetic_val_000375` | 80375 / 375 | snowy / gas stations / dawn/dusk | 2996 : 1 | `3ca95b7d2913` | 4 |
| `synthetic_val_000437` | 80437 / 437 | snowy / tunnel / daytime | 2996 : 2 | `2b76a71f0413` | 4 |
| `synthetic_val_000481` | 80481 / 481 | overcast / residential / daytime | 2996 : 3 | `d2a95f1b4345` | 4 |

[ours] Separate [inspection records](../experiments/phaseC_source_audit/image_inspection.json)
and the [labelled contact sheet](../experiments/phaseC_source_audit/inspection-contact-sheet.png)
record the only image decoding performed: these 17 RGB JPEGs, each 640×360.
All visibly depict geometric graphics rather than captured driving scenes.
This resolves content classification but does not recover an original identifier
before the upstream append, an exact generator/seed, or the rationale for weather,
scene and time-of-day labels. The source's `image_id` is its only source identifier;
there is no additional original-ID column to follow.

## Full selected-sample checks

[ours] [Audit records](../experiments/phaseC_source_audit/audit.json) and the compact
[3,000-row ledger](../experiments/phaseC_source_audit/selected_row_ledger.json) establish:

- All 3,000 source metadata rows match frozen IDs, physical splits and mapped labels.
- All 3,000 current raw-byte SHA256 values match extraction-recorded hashes.
- All 750 cache chunk checksums and their stored ID/split/label alignment match.
- No duplicate ID occurs among all 80,500 source metadata rows; neither the source
  nor the selected sample has train/validation identifier overlap.
- No two selected rows have identical raw-byte SHA256 values, within or across
  splits. Identifier overlap and exact-byte content overlap were tested separately.
- Selected fragment counts are 2,000 from original train fragment 0, 983 from
  original validation fragment 1, and 17 from appended validation fragment 2.

[interpretation] Exact-byte uniqueness does not exclude re-encoded copies,
near-duplicates or shared scenes. The other 2,983 selected images were not decoded
or authenticated against an original BDD release; no blanket natural-image-origin
claim follows. Full-source byte deduplication outside the selected sample was not
performed. No dataset-provided embedding or virtual duplicate column was read.
The executed procedure is retained in the [audit procedure](../experiments/phaseC_source_audit/procedure.md).

## Campaign scope and unresolved effects

[ours] All 17 occur in accepted C1-main's sample and extraction manifests.
C2 and C3 bind those same cache/sample hashes, so all their validation conditions
inherit them. C1-main's training split contains none of these IDs; its PCA/rank
bases, train-derived common-offset transform and probe training fit on training
rows. This establishes where rows participate, not absence of downstream effects.
C0's recorded 32-image IDs contain no such prefix. The exploratory C1-pilot
contains six **different** IDs from the same appended fragment:
`000015`, `000052`, `000146`, `000220`, `000379`, `000415`
(all with the `synthetic_val_` prefix); pilot-based hardening inherits them.
Those six were checked by metadata membership only, not decoded here, and are not
individually assigned content classification 4 by this inspection.

[ours] In main validation, these 17 account for all four `tunnel`, all four
`gas stations` and the sole `foggy` label. Their other labels are clear 3,
overcast 2, rainy 5, snowy 6; city street 1, highway 3, residential 5;
dawn/dusk 3, daytime 7, night 7. The frozen eligible balanced-probe class sets
exclude tunnel, gas stations and foggy. Within eligible classes, flagged-row
counts are weather 16, scene 9 and timeofday 17. Raw probe accuracy includes all
validation rows; class eligibility restricts balanced/F1 scoring only.
Retrieval includes all validation rows as queries and gallery candidates,
including ineligible classes as gallery candidates for macro retrieval.

[interpretation] The accepted endpoints describe their recorded enriched-source
sample. The generated graphics and unverified attribute semantics qualify claims
about natural driving-scene utility. The magnitude and direction of their effect
on geometry, probe validation, retrieval, numerical turnover and margin analyses
are unknown without a separately declared sensitivity analysis. A 1.7% query
fraction does not bound retrieval effects to 1.7%, because these rows also appear
in other queries' galleries. This audit neither discards the entire study nor
establishes that the remaining results are unaffected. Mathematical invariances
have a different evidential status from empirical natural-workload utility claims.

## Recommendation — source defect confirmed

[interpretation] Propose the smallest **separately labelled exclusion sensitivity**:
retain the accepted cache and all historical results, use the unchanged 2,000
training vectors and the remaining 983 validation vectors, and exclude exactly
these 17 confirmed graphics **from both queries and gallery candidates**.
Predeclare this audit-based exclusion and the unchanged C1/C2/C3 conditions before
execution. Reuse train-fitted transformations and probe fits where compatible;
re-evaluate validation geometry, probe scores/support, retrieval/floors and the
existing C3 error/identity/attribute/margin endpoints on the reduced gallery.
Retain frozen probe class eligibility with explicit revised supports; declare how
C3 reference margins/bins are rebuilt by its existing deterministic algorithm.
Do not retune precision conditions, severities, quantizers or solver grids.
Report differences from historical values without replacing them or presenting
this sensitivity as an independent confirmatory replication. This addresses the
natural-validation scope of C1 H1–H4, C2 utility comparisons and C3 stability
findings. The exploratory pilot's six related IDs require separate content checks
if that historical pilot is later corrected. No correction was executed here.

[interpretation] Recovering the upstream builder/append script and label-assignment
record would explain why these graphics were distributed, but is not required to
recognize the observed mismatch with a naturally photographed driving validation
population. Independent original-source replication remains a later decision.

## Preservation and verification

[ours] A pre-audit checksum snapshot protects tracked historical artifacts,
scientific code/dependencies, the canonical cache manifest and all 750 canonical
chunks. The verification record documents unchanged bytes, with the sole
explicitly authorized exception of appending this audit to C3 source qualification.
The original disclosure remains an exact prefix. No inference, endpoint execution,
cache write, ID regeneration, sample change, dependency change, ANN or C4 occurred.
Repository test/lint outcomes are recorded in
[verification.json](../experiments/phaseC_source_audit/verification.json).
