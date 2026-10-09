# Second-encoder paired-data replication implementation plan

> **For agentic workers:** After separate execution authorization, use
> `superpowers:executing-plans` to implement the tasks below. Checkboxes describe
> future work and authorize no execution during this preparation task.

**Goal:** Test Jina-CLIP-v2 within-encoder compression responses on the exact
accepted 1K image/five-caption COCO sample, with unchanged endpoint definitions.

**Architecture:** A separately locked author-model extraction process exports
plain native 1024d matrices and immutable input/cache identities. The accepted
project environment derives learned prefixes and runs its existing model-independent
evaluator in a new namespace. Historical source/cache/results are read-only.

**Tech stack:** Proposed extraction project: Transformers4.49.0, timm1.0.15,
einops0.8.1, accelerate1.5.2; exact accepted Torch/torchvision, NumPy/Pillow.
Its own hub/tokenizers dependency resolution; no root lock changes. Author code
and nested config revisions are pinned in the proposal. This stack is not yet
installed or tested. Analysis remains on the existing accepted lock.

**Spec:** [second-encoder-replication-proposal.md](second-encoder-replication-proposal.md).

## Global constraints

- Only `jinaai/jina-clip-v2`, native 1024d/MRL 256d/MRL 128d; no alternate model or grid.
- Exact accepted 1,000 image groups/five captions, ordering/relevance/category
  eligibility; no selection call, caption repair or additional content filtering.
- CPU FP32, batch size 1/four threads, author 512px processor, masked text mean/image CLS,
  checkpoint-trained default LoRA; `task=None` / no instruction in both directions.
- Canonical image[1000,1024], shared caption[5000,1024], separate logical roles;
  compressed leading dimensions require L2 normalization. No duplicate role inference.
- Commit tested implementation/protocol, then committed source/input/runtime freeze
  before inference; commit complete canonical hashes before scientific endpoints.
- RSS<=4,608 MiB; two consecutive>=60 s windows each>=256 MiB swap I/O or allocation
  failure stop/checkpoint. A failed full-wrapper gate does not authorize a fallback.
- Root dependencies, frozen BDD/COCO artifacts and existing endpoint functions
  remain unchanged. No PCA, precision, ANN, new sample, prompts or adjudication.
- Stop after the three-condition report and qualified interpretation.

## Review focus

1. Nested config/code loads silently following mutable `main`: fail offline;
   bind all four repository revisions and every executed file (task 1).
2. `task=None` accidentally disabling the trained LoRA or inheriting an instruction:
   assert adapter index0 is active and rendered text is original (task 1).
3. 768d-specific cache/MRL validation silently accepting/truncating1024d:
   new dimension-aware boundaries with old-path regression tests (task 2).
4. Shared caption cache mistaken for two independent samples, positives removed
   as self matches, or IDs shifted between roles: alias/relevance tests (tasks 2/3).
5. Resume succeeding with changed code, weights, dtype, tokenizer or sampling,
   or rewriting historical artifacts: explicit mismatch/hash tests (tasks 2/3).

## 1. Author contract and isolated runtime

**Files:** Future `tools/jina_extraction/pyproject.toml`, its `uv.lock`;
`src/embedding_diagnostics/models/jinaclipv2.py`;
`tests/test_jinaclipv2.py`; optional recorded resolution-only author patch in
`tools/jina_extraction/patches/`. No edits to the accepted image/text adapter.

**Interfaces:** `JinaCLIPv2(snapshot: Path, code_roots: dict[str, Path])` owns
CPU FP32 full-wrapper loading, frozen processor/tokenizer and provenance.
`encode_images(images: list[PIL.Image.Image]) -> np.ndarray` and
`encode_captions(captions: list[str]) -> np.ndarray` return finite unit FP32[N,1024].
`provenance` records all resolved files, adapter/instruction choice, dtype,
attention flags, device, thread counts and runtime versions.

- [ ] Read the spec and immutable sources. Resolve the separate project through
  ordinary uv project/lock workflow. Do not install packages into the accepted
  environment. Verify its source imports first; reject unsupported symbols.
  Proposed 4.49 pins are a starting point: if import checks require amendments,
  document and finalize them before inference and freeze, not after endpoints.
- [ ] After authorization only, acquire one pinned safetensors checkpoint plus
  required tokenizer/config/code. Verify advertised digest, tensor names/counts,
  licences and metadata; no nested v3 weights or ONNX exports. Preserve notices.
- [ ] Write failing model-free tests for exact no-instruction rendering, default
  trained adapter selection, pad/special-token mean pooling, image CLS path,
  processor 512/resize/crop/normalization, native 1024d output and FP16 rejection.
  Assert original numeric row order despite upstream internal input sorting.
  Run `uv run pytest tests/test_jinaclipv2.py`; observe failures before implementation.
- [ ] Implement the adapter using official complete-wrapper APIs, explicit
  `torch_dtype=torch.float32`, CPU, inference/eval mode, low-memory safetensors
  loading and no CUDA libraries. Pre-tokenize without truncation; excess context
  fails. No instruction search or invented passage adapter.
- [ ] Resolve nested config/code from immutable local sources. Prefer registering
  pinned classes/config locally and preloading tokenizer/processor; if the
  wrapper requires a patch to pass nested revisions, constrain it to resolution
  arguments, record its diff/hash and test resulting numerical path equivalence.
  Fail any attempted network load under offline execution. No architecture patch.
- [ ] Run focused tests to GREEN, compare root `pyproject.toml`/`uv.lock` and
  historical hashes, and record the validated extraction lock. Full-stack real
  numerical validation waits for the separately frozen contract smoke.

## 2. Native cache and input identity without historical mutation

**Files:** Future `src/embedding_diagnostics/jina_paired_cache.py`,
`src/embedding_diagnostics/jina_compression.py`,
`tests/test_jina_paired_cache.py`, `tests/test_jina_compression.py`,
`scripts/prepare_jina_paired_replication.py`.
New namespace: `experiments/phaseC_paired_jina_replication/`.

**Interfaces:** `JinaPairedCache(root, provenance, rows)` supports `.completed`,
`.append(matrix)` and context locking; `load_jina_cache(root)` returns matrix and
ordered rows. Reuse atomic/checksum utilities, not the768-only validator.
`jina_matryoshka(canonical: np.ndarray, dimension: int) -> np.ndarray` accepts
only unit native 1024d and dimensions 1024/256/128; native is an exact copy, compressed
output is FP64 prefix-renormalized. `prepare` creates new input/freeze manifests
referencing the accepted sample; it never regenerates sample IDs.

- [ ] Write tests for arbitrary finite unit FP32[*,1024] chunks, checksum/order
  alignment, zero/nonfinite/dimension rejection, atomic prefix resume, duplicate
  IDs and changed provenance. Assert completed caches invoke no model loader.
  Add tests that existing768 caches and historical files are byte-unchanged.
- [ ] Write learned-prefix tests: shape, finite/unit norms, native identity,
  determinism, mandatory post-slice normalization and unsupported dimensions.
  Run focused tests to observed RED, implement the minimal boundaries, then GREEN.
  Do not rewrite the accepted C2 768d `matryoshka` implementation.
- [ ] Create new input rows from the byte-identical accepted sample, original
  image hashes and caption text. Preserve raw/caption IDs and parent maps.
  Freeze Jina tokenizer IDs/specials without EmbeddingGemma prefixes; logical
  query/document aliases reference one shared caption identity and ordered matrix.
- [ ] Bind the accepted starting commit, all accepted ledger/cache/result hashes,
  new code/protocol/lock hashes, four model/config/code refs and local file hashes,
  full-wrapper mode, task/LoRA settings, preprocessing, scoring/interval rules,
  resource limits and actual runtime versions. Tests must reject any changed
  source/order/role/weights/dtype/tokenizer or uncommitted freeze.
- [ ] Run full required repository checks and isolated contract unit tests;
  commit implementation and finalized protocol. Then create/commit the new freeze
  **before any encoder forward**. This is future work, not this proposal commit.

## 3. Guarded extraction, unchanged evaluator and transparent report

**Files:** Future `scripts/run_jina_paired_replication.py`,
`scripts/report_jina_paired_replication.py`, `tests/test_jina_paired_workflow.py`.
Reuse existing `paired_evaluation.py`, geometry and checksum/resume utilities;
do not alter the accepted runner's paths, conditions or stored records.

**Interfaces:** Runner stages `smoke`, `extract`, `bind`, `evaluate` operate only
on the new namespace. Reader presents image/query/document logical views to the
unchanged evaluator. Reporting consumes checksummed records, with no inference
or endpoint recomputation.

- [ ] Write RED workflow tests for shared-text role identity, multiple positives,
  cross-modal no-self-exclusion, image-only ID exclusion, numeric ties, exact row
  alignment, fixed 75-category eligibility, all 80 supports, parent-unit resampling,
  gallery-dependent floors and native circular relevance shift. Reuse hand-
  checkable existing evaluator fixtures; do not mirror scores with another metric.
- [ ] Add RED tests for freeze/cache gates, protected-history write rejection,
  completed zero-model extraction resume, completed zero-endpoint/ranking/
  bootstrap resume, interrupted final-interval recovery and saved smoke repeats.
- [ ] Implement the thin workflow using the fixed APIs from tasks1–2. Inference
  lives only in the isolated extraction process; evaluate/report run under the
  accepted root environment. Logical aliases do not become duplicated statistical
  units. Store per-query/per-parent outputs and native top10 identities.
- [ ] Run focused tests to GREEN, full `uv run pytest`, `uv run ruff check .`,
  `uv lock --check`, and isolated-lock/unit checks; commit before freeze/execution.
- [ ] After the inference freeze, run exactly the first 16 accepted image groups
  and80 captions. Measure model-loading time, each modality throughput, RSS and
  active swap; persist independent first-input repeats. Compare derived 256/128
  against official direct-prefix calls at rtol1e-5/atol1e-6, and verify parameter/
  activation/normalization/output dtypes. No retrieval/geometry endpoint. Retain
  canonical entries; stop and report any contract/resource failure.
- [ ] If the declared gates pass, resume the fixed 1K/5K extraction; never select
  another loading mode from endpoint outcomes. Bind and **commit** complete native
  cache hashes before evaluating native 1024d/MRL 256d/MRL 128d.
- [ ] Evaluate existing Hit@1/5/10, set recall@10, overlap/turnover/ties, category
  supports/floors and descriptive geometry. Preserve four paired Hit@10 intervals
  with 5K parent resamples/seed 20261010; no extra CI family or thresholds. Report
  each condition and native-relative effects, then descriptive comparison with
  already accepted EmbeddingGemma records without recomputing them.
- [ ] Include both absolute dimensions and compression ratios; preserve nulls,
  original-source/pretraining limitations and the absence of causal attribution.
  Explain whether first-rank, any-positive, caption coverage, category and identity
  responses agree. Three geometry points cannot validate a utility predictor.
- [ ] Verify completed zero-work resumes and all historical hashes, run required
  checks, commit results/status/log separately, report push state and stop.
  Failure to fit/validate requires a new reviewed loading proposal, not silent
  quantization, lower resolution, sample shrinkage or another encoder.

## Preparation handoff

[ours] This plan and the proposal were written from read-only paper/code/config
inspection. No future file or experiment above has been implemented, no model
weights acquired, no dependency changed and no endpoint computed. Review the
feasibility gate and isolated-runtime choice before authorizing execution.
