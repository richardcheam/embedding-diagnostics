# Original COCO paired-data replication

[ours] This is a separate namespace for the authorized bounded protocol in
[image-text-replication-proposal.md](../../docs/image-text-replication-proposal.md).
Implementation `141ebce`; original-source/sample/rendered-input freeze `2eee895`
precedes inference. Exactly 1,000 original COCO2017 validation image groups,
five smallest-ID captions per group, and separate query/document caption caches.
Only native 768d and learned MRL 256d/128d are authorized.

## Source qualification

[ours] Source archives were acquired from the original publisher-named AWS S3
bucket over certificate-verified HTTPS. The publisher hostname currently fails
TLS hostname validation; the pre-freeze protocol records the same-bucket
path-style transport substitution, without bypassing certificate checks. Saved
response headers, publisher documentation, original annotation/license records
and archive/annotation SHA256 hashes establish the source chain and continuity.
No independent publisher-signed archive checksum was verified. See
[acquisition.json](acquisition.json) and [source_inventory.json](source_inventory.json).

[ours] Structural validation read all 5,000 original validation image members,
verified CRC, decoded RGB dimensions and computed raw-byte and dimension-bound
RGB-pixel hashes. Caption/instance dictionaries, unique IDs, foreign keys,
filenames and licenses agree. Raw-byte, decoded-pixel and recovered Flickr-photo
identity grouping yields 5,000 distinct groups. Fixed hash-order selection,
without replacement or quotas, retained 1,000 groups and five smallest caption
annotation IDs each; original numeric identifiers and text are preserved.

[ours] All selected thumbnails and their five-caption bundles were inspected
in 50 contact sheets; [content_inspection.json](content_inspection.json) records
exact coverage, local sheet hashes and observations. This is assistant visual
inspection, not independent human relevance adjudication or full-resolution
forensic review. Original collages, illustrations, edited photographs and caption
factual errors remain. Some displayed long captions may wrap past their panel;
complete original text is preserved in the source and input manifests.

[interpretation] Dataset-source authentication is distinct from byte-level
continuity and from caption truth. Unknown encoder pretraining overlap, incomplete
recorded positives and residual resized/near/event duplicates remain limitations.
A new evaluation sample is independent of BDD sampling, but uses the same encoder;
it is not independent encoder replication. The historical 983-row BDD sensitivity
and the pilot's six separately unresolved IDs retain their existing qualifications.

## Execution and provenance

Source/sample/input hashes and runtime/model/code identities are in
[freeze.json](freeze.json). Large vector chunks live in gitignored role-specific
`image/embedding-cache`, `query/embedding-cache` and `document/embedding-cache`.
The canonical hashes must be committed in `cache_binding.json` before endpoints.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/run_image_text_replication.py smoke
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/run_image_text_replication.py extract
uv run python scripts/run_image_text_replication.py bind
# Commit canonical cache binding before the following step.
uv run python scripts/run_image_text_replication.py evaluate
uv run python scripts/report_image_text_replication.py
```

[ours] The smoke has no retrieval endpoints. Each role's first input is repeated
with frozen FP32 tolerances and its evidence saved atomically. Extraction resumes
validated prefixes; completed extraction loads no model. Endpoint condition
records are checksummed; missing intervals can be recovered from stored parent
scores after an interrupted final append, without reranking. Completed interval
resume performs zero bootstrap calls. Historical experiment and BDD canonical
hashes are guarded at execution gates.

[interpretation] T2I uses the search-query caption prefix; I2T uses the document
caption prefix and five-positive retrieval. Directional differences cannot
isolate modality asymmetry. Category image-only relevance differs from paired
caption relevance; marginal geometry is descriptive, and three dimensions do
not validate a general predictor of representation quality.

## Completed execution

[ours] Contract smoke passed at `7cd2c17`; complete canonical hashes were committed
at `b24bf28` before endpoints. Shapes are image [1000,768], query [5000,768] and
document [5000,768], finite unit-normalized FP32. Smoke plus remaining extraction
wall time was 4.106 hours, peak RSS 3,571.05 MiB. Both caption roles were encoded
separately without truncation. Resource gates passed; source inspection time is
reported separately in [interpretation.md](interpretation.md).

[ours] [results.md](results.md) includes all three conditions, directions,
category supports, floors, paired intervals, geometry, ties and native pairing
control. [final_verification.json](final_verification.json) records completed
zero-inference/zero-endpoint resume, unchanged prior artifacts, role contracts and
protected historical/cache hash checks. [source_qualification.md](source_qualification.md)
separates completed authentication and content checks from residual limitations.
Large source archives and canonical arrays remain local and untracked.
