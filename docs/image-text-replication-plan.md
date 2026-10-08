# Independent image–text evaluation implementation plan

> **For agentic workers:** After separate authorization, use
> `superpowers:executing-plans` to implement this plan task by task. Checkboxes
> describe future work; none authorizes execution during this preparation task.

**Goal:** Implement only the reviewable three-representation paired-data protocol
in [image-text-replication-proposal.md](image-text-replication-proposal.md), after
separate execution authorization. **Status: implementation authorized; pre-inference execution gates required.**

**Architecture:** Authenticate original release files and freeze joined IDs;
produce separate resumable image/query-caption/document-caption canonical
matrices; feed them to model-independent exact retrieval and existing geometry.
The accepted BDD cache, source sensitivity and historical artifacts are read-only.

**Tech stack:** Existing locked Python, NumPy, Pillow, Torch, Transformers and
pytest/Ruff. Standard-library JSON/ZIP/hash processing; no pycocotools,
SentenceTransformers, downloader framework or new dependency is required for
these joins. Reuse C2 MRL/geometry functions and existing cache/provenance patterns;
do not adapt BDD attribute assumptions into the paired relevance evaluator.

**Spec:** [image-text-replication-proposal.md](image-text-replication-proposal.md).

## Global constraints and review focus

Exactly 1,000 image groups, five captions per group, native 768d/MRL256/MRL128;
CPU FP32/batch 1/four threads; no dependency changes, PCA, precision or ANN grid.
Source/sample freeze precedes inference; cache binding precedes endpoints.
Historical BDD artifacts and the canonical cache are read-only. The five main
review risks are dangling ID joins, duplicate image groups crossing sampling
units, caption roles sharing a cache identity, cross-modal positives mistakenly
self-excluded, and caption-level resampling overstating independence. Tasks 1–3
include explicit tests for each; no success threshold is chosen from results.

> The original preparation did not authorize execution. The subsequent user
> request authorizes this bounded implementation and execution; the source/sample
> freeze and cache-binding gates below remain mandatory.

## Proposed files and responsibilities

| File | Responsibility |
| --- | --- |
| `src/embedding_diagnostics/paired_data.py` | Original COCO JSON joins, source checks, group/sample ledger, five-caption selection and relevance maps; no HF objects or embeddings. |
| `src/embedding_diagnostics/models/embeddinggemma2.py` | Add an explicit text-role method reusing validated model/pooling; preserve image-only API and previous provenance semantics. |
| `src/embedding_diagnostics/paired_evaluation.py` | Rectangular exact scoring, deterministic total ordering, multiple positives, per-parent outputs, category comparator and conditional paired intervals. |
| `scripts/prepare_image_text_replication.py` | Source/sample/protocol identity and pre-extraction freeze; no semantic/geometry endpoints. |
| `scripts/run_image_text_replication.py` | Guarded local extraction, checksummed resume, post-extraction binding and three-condition evaluation. |
| `scripts/report_image_text_replication.py` | Tables/interpretation inputs from recorded results; no inference or retuning. |
| `tests/test_paired_data.py`, `tests/test_paired_evaluation.py` and adapter/cache tests | Synthetic source, contract, relevance, arithmetic, provenance and preservation tests. |
| `experiments/phaseC_paired_replication/` | Separate future source/sample/freeze/extraction/results namespace; large source/vector arrays ignored. |

## 1. Source contract and independently frozen sampling

- [ ] Complete original-source acquisition only when authorized; preserve official
  URLs/terms and validate archives as specified in the verification checklist.
- [ ] Write failing synthetic JSON/ZIP fixtures covering ID collisions, dangling
  caption/instance references, missing files/license records, bad dimensions,
  generated-placeholder prohibition and source-version mismatch. A detector must
  not claim to authenticate photographs automatically: selected source inspection
  remains a separately recorded human verification step.
- [ ] Implement minimal joins and separate raw-byte/pixel/source-photo groups;
  test transitive groups, numeric canonical representatives, group-only selection,
  stable hash ordering, retained aliases and first-five annotation IDs.
- [ ] Test original UTF-8 captions, duplicates/multiple positives, no replacement,
  category support and physical split identity. Corrupt fixtures must fail loudly.
- [ ] Run focused model-free tests; save real source census/authentication ledger
  without embeddings. If authentication or rights mapping remains unresolved,
  stop with the missing evidence; do not use an unverified mirror.

## 2. Text-role adapter and resumable extraction contract

- [ ] Write tests using processor/model mocks for both exact role prefixes,
  token-mask mean pooling with prompt included, unit FP32 768d output, empty text,
  over-context failure, no hidden precision/autocast change, and unchanged image
  adapter behavior. No ordinary test loads/downloads the checkpoint.
- [ ] Add the smallest explicit text method using the locked official Transformers
  interface and pinned local configs. Inspect the installed processor code before
  choosing its text-call signature; do not assume image nesting applies to text.
- [ ] Reuse or narrowly extend chunk-cache conventions to carry caption annotation
  ID, parent ID, role and rendered-input hash; separate matrices/manifests rather
  than pretending captions are BDD rows. Test resume mismatch and completed-cache
  zero-inference resume; a failure never manufactures substitute embeddings.
- [ ] Bind image preprocessing and both text-role instructions in provenance.
  Model loading stays CPU FP32/text+vision/no audio. No dependency/CUDA changes.

## 3. Paired evaluator and meaningful controls

- [ ] Write hand-checkable synthetic rankings for text→image one-positive hits,
  image→text five-positive hits versus set recall, repeated caption strings,
  numeric-ID ties and image-only self exclusion. Preserve paired matches across
  modalities. Verify row alignment and gallery-dependent random-order floors.
- [ ] Reuse native/MRL256/MRL128 derivation and geometry; test mandatory L2,
  native identity, matching dimensions and finite/zero-vector rejection. Validate
  actual FP64 score accumulation with a cancellation witness, separately from
  FP32 model output. Block scoring must match a small dense reference exactly
  under the declared ordering, including ties.
- [ ] Test category-macro memberships, all-gallery candidates, fixed support>=10,
  rare/zero-support reports and per-category `(n_c-1)/(N-1)` floors.
- [ ] Test the fixed bundle-permutation control preserves geometry/rankings but
  changes relevance; test image-group paired resampling carries all five captions,
  fixed gallery and paired conditions. No independent-caption error bars.
- [ ] Test protected historical/cache hashes and endpoint resume, including zero
  endpoint calls on completed resume. Save per-parent effects and failures.

## 4. Commit/freeze gate, then bounded execution

- [ ] After separate authorization, finalize the proposal as the execution
  protocol, including any pre-data/source-contract amendments. Record deviations
  rather than hiding them. Run `uv run pytest`, `uv run ruff check .` and
  `uv lock --check`; commit tested implementation and final protocol.
- [ ] Generate and commit source/sample/input freeze with exact 1K image-group
  and 5K caption IDs, raw/caption hash joins, category eligibility, roles, all
  protocol/code/runtime/config hashes, tie/floor/bootstrap rules, and history
  protection. No embeddings or scientific endpoints before this gate.
- [ ] Perform the 16-selected-image/text contract smoke with finite/shape/norm,
  repeat tolerance and throughput/RSS/swap checks only. Reuse its canonical
  entries. Do not inspect retrieval to optimize prompts or the sample.
- [ ] If resources allow, finish resumable 1K image/10K text-role extraction;
  validate alignment, local provenance and all caches. Record failures instead
  of dropping rows. Bind canonical cache hashes before endpoint execution.
- [ ] Execute exactly three representations, prescribed directions/comparator,
  native pairing control and conditional intervals. Check completed resume and
  unchanged historical/canonical BDD hashes before accepting the result.

## 5. Reporting and stop

- [ ] Generate absolute/delta tables and per-parent paired uncertainty, geometry
  separated by modality/role, coarse category comparison, overlap and controls.
  Report every dimension and null outcome; no best-condition selection.
- [ ] Clearly state new evaluation sampling versus same-encoder evidence,
  incomplete paired relevance, unmeasured pretraining/near-duplicate dependence,
  fixed-gallery interval scope, and source verification actually completed.
- [ ] Preserve the 983-row sensitivity as current qualified historical analysis
  and the six unresolved pilot IDs as historical qualifications. Commit new
  results separately and stop. Any next inference/model/control needs new scope.

## Preparation checks actually performed

[ours] Confirmed HEAD `5f5f991` and initially clean tree; read AGENTS.md, source
audit, sensitivity protocol/results/interpretation/source qualification, current
STATUS and research-direction review. Inspected accepted image adapter, pinned
prompt JSON and C2 MRL/geometry functions without loading a model. Read original
COCO source documentation/schema/terms and caption-collection methods; read
Flickr30k author pages and the original TACL dataset methods. This is a focused
source choice, not exhaustive novelty review. Publisher archives/checksums,
actual caption census, duplicate groups and final IDs remain future evidence.

[ours] Local inventory and current RAM/swap/disk observations are recorded in
the proposal. No dataset/model download, image decoding, inference, scientific
endpoint, sample/cache write, new dependency or accepted-result change occurred.

[ours] Preparation verification: 458 tests passed, one optional skip (37.96 s);
Ruff and `uv lock --check` passed. Verified frozen hashes for 491 historical
artifacts, sensitivity artifact hashes and all 750 canonical chunks, without
loading embedding arrays or computing endpoints. Local document links resolve;
only the two new proposal/plan documents are changed.
