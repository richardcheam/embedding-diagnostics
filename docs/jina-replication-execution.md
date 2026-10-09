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
