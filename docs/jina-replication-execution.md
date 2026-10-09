# Jina paired replication execution record

[ours] Authorized scope: the exact accepted COCO1000 image groups and5000 original
captions, pinned Jina-CLIP-v2, CPU FP32/batch1/four threads, official512px images,
plain captions/task=None/checkpoint-trained default LoRA. Native1024 and learned
256/128 only. Historical BDD and COCO evidence is read-only.

[ours] Source correction: `b4c2c97`. Separate extraction environment resolved;
o root dependency changes. Four pinned repositories acquired selectively;
advertised checkpoint digest verified; no nested text weights acquired.
`acquisition.json`, `checkpoint_inventory.json` and `resolution_patch.json`
record file identities, header inventory and the two resolution-only source
changes. No encoder construction/forward or scientific endpoint has run during
preparation. Frozen input tokens preserve the exact original caption strings.

[ours] New model-free tests exercise1024d shape/finite/unit validation,
mandatory prefix normalization, original ID ordering, parent alignment, shared
caption views, exact source-resolution amendments, protected hashes, chunk resume,
changed provenance, zero-model/zero-endpoint completed resumes and active swap
gates. Numerical integration remains conditional on the committed contract smoke.

[ours] Fresh whole-branch review identified smoke failure/resume promotion,
unenforced caller-offline assumptions, interval rendering integrity and throughput
accounting defects before freeze. They were corrected before real model use:
saved failed checks/blockers stop resume; the adapter requires both offline flags
before author imports; executed dynamic source hashes must match pinned bytes;
interval rendering validates completion identity/checksums; native forward timing
excludes loading and extra contract calls. Active swap sampling covers loading.
Regression tests include constructor load/coverage checks, failed smoke/blocker
resumes, executed-source mutation and interval corruption/recovery.

[ours] Pre-freeze verification:529 tests passed, one optional skip (36.12s);
27 isolated-runtime model-free tests passed. Ruff, root/isolated lock checks and
`git diff --check` passed. Historical protected hashes and accepted canonical
cache hashes were verified. No encoder forward or scientific endpoint has run.

[ours] Implementation `35bdf75`; separate pre-forward freeze `8557b78`. Offline
smoke invocation failed during nested tokenizer construction, encoder forwards0,
20.623s extraction-stage time, partial-process peakRSS736.785MiB. The pinned
nested constructor at `modeling_xlm_roberta.py:479` initiates another tokenizer
lookup without code revision. Offline mode rejects it before checkpoint loading
completes. See the [preserved blocker](../experiments/phaseC_paired_jina_replication/interpretation.md).
No scientific endpoint, image decode, full cache extraction/binding or fallback
was executed. Native/prefix repeat, throughput and full-wrapper resource gates
remain untested. Completed scientific zero-work resumes are not applicable to
this blocked attempt; their model-free regression tests pass.

[ours] Final blocked-attempt verification:529 root tests passed, one optional
skip (34.24s);27 isolated unit tests passed. Ruff, both lock checks, whitespace
and new local document links passed. Frozen model/source/code/runtime/input
identities and all protected historical/cache files verified; accepted scientific
artifact and root dependency diffs from `ebe17fb` are empty. Complete canonical
rows0. Remote check finds no `phaseC-paired-replication` branch; commits remain
local and no push occurred. No full-model resource or scientific success claimed.
