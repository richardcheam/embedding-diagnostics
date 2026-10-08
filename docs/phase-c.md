# Phase C: an external pretrained reference

[measurement] Update 2026-10-08: C1-main and the cache-only C2 comparison are
complete. See the [C2 tables](../experiments/phaseC_c2/results.md) and
[qualified assessment](../experiments/phaseC_c2/interpretation.md). C3/C4 remain
deferred; the staged design and historical C0/pilot record below are preserved.

Phase A/B ask what happened in our controlled SSL training systems. Phase C
asks whether their diagnostic conclusions generalize to a strong externally
pretrained representation. EmbeddingGemma 2 is an external reference encoder;
the seven training conditions and shared Trainer are unchanged.

C0 integration infrastructure is implemented. Its smoke record is
`experiments/phaseC_c0/smoke.json`. C1-pilot is complete and exploratory; C1-main, C2, and C3 are planned;
C4 is deferred. No Phase-C scientific conclusion is available.

## C0 measured integration record

[ours] The audited 2026-10-07 run used the pinned revision
`914f7f89142e33e77833254d9c9b90c3cef7303b`, CPU FP32, batch size 1,
four CPU threads, and chunks of four. Audio was excluded; the loaded
text+vision model contained 438,760,448 parameters.

| Measurement [ours] | Value |
| --- | --- |
| Selected images | 16 train + 16 validation, ordered by image ID |
| Embedding shape | 32 × 768 |
| Finite values | all |
| L2 norm range | 0.99999988–1.00000012 |
| Extraction + cache writing | 400.47 s |
| Model loading + selection + weight hashing | 19.97 s |
| Independent repeat of all 32 images | 353.17 s |
| Repeat maximum absolute difference | 0 |
| Total CLI wall time, including repeat/diagnostics | 773.74 s |
| Peak process RSS | 3680.59 MiB (3.59 GiB) |
| Cache + manifest | 120805 bytes (117.97 KiB) |
| CUDA inference / peak VRAM | not tested: locked Torch excludes sm_61 |

[ours] Extraction overlapped CPU test execution for part of the run; the
repeat pass ran after those tests finished. These timings are integration
measurements, not a controlled device benchmark. No claim about whether CUDA
FP32 fits in four GB is available.

[interpretation] Start C1 with 512 images total, tentatively 384 train and
128 validation, using the uniform selection protocol frozen below before inference.
The C0 throughput suggests about 1.5–2 hours for extraction; rare categories
are reported rather than oversampled. This is an instrument pilot, not the main experiment.

[ours] Metadata inspection found only 11 fully-labelled training tunnel images
and 25 training gas-station images in this redistribution. Naive random
sampling will undersample them. Class quotas, any change from natural BDD
frequencies, and classes failing the existing scoring support floor must be
reported. C0's geometric readings have a 32-image sampling limit and establish
no semantic result.

[ours] Completed-cache resume reused all 32 vectors with zero new inference
under a Python TCP connection guard and HF/Transformers offline mode. Reloaded
geometry was identical. A second process's extraction also matched exactly.
See `experiments/phaseC_c0/verification.json`. The default suite finished with
384 passed and one opt-in accelerator test skipped; Ruff, lock validation,
the existing panel validation command, and the CPU environment check passed.

## C0 protocol

Freeze the first 16 fully-labelled lexicographic image IDs from each physical
train/val split, then order all 32 by image ID. This is an integration sample,
not a representative evaluation sample. Selection precedes inference. Unknown
attributes map to -1 using Phase B's vocabulary; fully-labelled selection
excludes them. Decode/inference failures stop with the affected IDs instead of
silently changing the sample.

The Lance reader projects five metadata columns when indexing and adds
`image_bytes` only for the requested batch. It never reads the virtual
`embedding`, `is_duplicate`, or annotation columns. It indexes metadata, never
the full image corpus. The source path, Lance version, row count, selected IDs,
labels, and SHA256 of every extracted image's original bytes are recorded.

The adapter uses Transformers 5.19.0: `EmbeddingGemma2Config.from_pretrained`
with `audio_config=None`, `EmbeddingGemma2Model.from_pretrained`, and
`AutoProcessor.from_pretrained`. Every load is local-only; a Hub ID therefore
requires an already cached snapshot. Explicit snapshot paths avoid dependence
on the default HF cache. Snapshot directory revisions must match the requested
40-character revision. Arbitrary local copies record the revision as a caller
assertion rather than verified Hub identity; actual weight/configuration hashes
also travel in provenance. Use the canonical downloaded HF snapshot for C0.

[established] The official API projects each token to 768 dimensions. The
pinned checkpoint declares mean pooling including unmasked tokens, then L2
normalization. The adapter follows those semantics: attention-mask mean pooling
followed by L2 normalization. Images receive no task prefix or Phase-B ImageNet
preprocessing. The official processor controls resizing/rescaling. Disabling
audio configuration excludes its weights before construction; audio processor
metadata does not imply a loaded audio tower. Sources:
[official API](https://huggingface.co/docs/transformers/model_doc/embedding_gemma2),
[pinned model card](https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/README.md).

[established] FP16 inference is unsafe for this model's activation range.
C0 accepts FP32 only, disables autocast, checks loaded parameter dtypes, and
rejects FP16 before loading. CPU is the default. CUDA is explicit and requires
an architecture preflight; visibility alone is insufficient. See the pinned
model card above.

The canonical cache uses compressed FP32 NPZ chunks with image IDs, physical
split, and integer attribute codes. It stores no image bytes. A fsynced chunk
is renamed before an atomic manifest commit. An interrupted, uncommitted chunk
can require recomputation; completed chunks remain usable. A local advisory
lock excludes concurrent writers. Resume validates chunk checksums and row
alignment, then **always** checks completed source-image bytes, even without
repeat inference. This prevents mixing datasets recreated at the same path and
version. Provenance mismatches reject resume, including model/revision, weights,
preprocessing, dtype, device, dimension, selection/order, runtime versions,
source code, batch size, or thread count. Orphan chunks have no authority
without a manifest entry. This is process-crash recovery, not a claim about
filesystem loss.

The manifest additionally records enabled modalities, parameter count, model
and processor configurations/classes, normalization policy, actual dependency
versions, code commit and source hashes including uncommitted extraction code,
batch/chunk sizes, CPU threads, and creation timestamp.

The completed cache is reloaded before calling existing `collapse_metrics`.
Its participation ratio uses centered covariance eigenvalues; RankMe uses the
raw matrix. Existing geometry arithmetic is float64; canonical inference and
vector storage are FP32. No panel thresholds or representation-quality verdicts
are added. C0 omits probes/retrieval: its 16 validation images cannot establish
the required class coverage.

## Reproduce locally

```bash
uv sync --locked
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --locked python scripts/extract_phase_c.py \
  --dataset /mnt/hdd/data/datasets/BDD100K-enriched \
  --model /mnt/hdd/data/hf/hub/models--google--embeddinggemma-2/snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b \
  --cache experiments/phaseC_c0/audited/embedding-cache \
  --result experiments/phaseC_c0/smoke.json \
  --device cpu --batch-size 1 --chunk-size 4 --threads 4 --verify-repeat
```

`--verify-repeat` independently re-extracts every selected image after cache
completion and verifies byte hashes and embedding equivalence. Tolerances are
fixed before extraction: rtol=1e-5, atol=1e-6. Failed repeats produce an error
instead of a success report. The same command automatically resumes matching
chunks. Timing distinguishes newly extracted from resumed images; preserve the
original smoke record using a different `--result` path on a resume check.

For geometry without loading the model:

```bash
uv run --locked python scripts/extract_phase_c.py \
  --cache experiments/phaseC_c0/audited/embedding-cache \
  --result /tmp/phase-c-geometry.json --diagnostics-only
```

Cache directories are gitignored. The compact smoke JSON is the integration
record. Default tests use synthetic images, real tiny Lance datasets, and a
mocked heavyweight model load; they never download a checkpoint. The existing
SIGReg accelerator test is opt-in via
`EMBEDDING_DIAGNOSTICS_ACCELERATOR_TESTS=1` on compatible hardware.

## Next steps, after C0

Use measured throughput to choose the C1 pilot. The pilot uses the uniform, deterministic sampling protocol frozen below.
Freeze main IDs before interpreting main performance.
Report class coverage and unscorable classes; a deliberately balanced pilot
must not be represented as the natural BDD distribution. The C0 CLI enforces
16–32 images to prevent accidental main extraction.

C1 needs separate train and validation matrices. Reuse `linear_probe_scores`
with fitting/C selection confined to train, and existing retrieval functions
on validation. Do not use `diagnose_transform`'s in-sample semantic evaluation
for C1. Fit data-dependent transformations once on train, then apply to both
splits; freeze the protocol before the pilot. Do not fit separate SVD bases
on validation. Reuse one canonical 768d extraction for all interventions.

Phase B configured 5000 probe-train and 8000 validation images. Compare C1's
eventual main counts against that basis and measured MSI runtime; 5000/2000 is
a candidate rather than a frozen experiment. No artificial training seeds are
created for this deterministic external reference.

C2's Matryoshka-versus-destructive-rank study, including slicing/renormalization
utilities and tests, stays deferred until C1 is stable. C3 acts on cached
vectors and keeps storage precision, similarity arithmetic, and ANN
approximation distinct from inference precision. C4 is deferred. Controlled
mathematical transformations establish sensitivities/invariances, not what
occurs naturally during training.

## Frozen C1-pilot protocol (before inference, 2026-10-07)

C1-pilot is PILOT / EXPLORATORY. The metadata-only census is saved in
`experiments/phaseC_c1_pilot/dataset_census.json`. Eligibility requires all three
canonical Phase-B labels; undefined/unknown labels are excluded consistently.
The census retains raw undefined counts and joint attribute counts.

Select 384 train and 128 validation rows uniformly without replacement from
separate fully-labelled pools sorted by image ID. Use NumPy PCG64 with
SeedSequence([0, split_index]), with train index 0 and validation index 1.
Globally sort selected rows by image ID for extraction. There are no class
quotas, coverage constraints, replacement, or changes based on embeddings.
Exact IDs and all canonical class supports are frozen in sample_manifest.json
before model loading. Absent classes remain absent; validation classes below
the existing support floor of 10 are explicitly excluded from macro scores.
Overall accuracy and micro retrieval still include every selected row.

Keep CPU FP32, four threads, batch size 1 and chunks of four. The optional
batching benchmark is omitted to retain the validated memory footprint.
Canonical vectors remain 768-dimensional, attention-mask mean pooled and
L2-normalized. Check the pinned module sequence Transformer -> Pooling ->
Normalize and its mean-pooling config explicitly, without SentenceTransformers
as a runtime dependency. Use only raw image bytes for inference.

Reuse existing geometry definitions separately on train and validation.
Do not use the Phase-B panel: absolute scale thresholds calibrated on its
unconstrained vectors do not transfer to unit-normalized EmbeddingGemma vectors.
Definitions are reusable; threshold verdicts require independent justification.
No new health thresholds or confirmatory claims are introduced.

Fit both standardized and unstandardized probes with existing ProbeConfig:
C grid 0.01 through 100000, train-only 80/20 inner selection, seed 0,
LBFGS tol 1e-8, max_iter 5000. Report selected C, raw feature scale, convergence,
iterations, train accuracy and underfit status for every endpoint. The existing
standardized implementation fits its scaler on the outer training set before
inner C selection; validation/test vectors and labels never fit either stage.
Report held-out accuracy, support-filtered balanced accuracy, majority floors,
and validation-only self-excluding cosine retrieval P@10 with chance floors.
Report all vocabulary class supports, including zero counts. A class absent
from training cannot be predicted by a fitted classifier, even if present in
validation. Report that limitation explicitly.

Reuse all five existing transformations and severities 0, .25, .5, .75, .9,
.99. Fit any data-dependent scale, mean or raw SVD basis on train exclusively;
apply the same fitted intervention to validation. Positive rank severity keeps
max(1, round(min(train.shape)*(1-severity))) training SVD directions. Severity
zero is identity on both splits, including validation directions outside the
training row span. Isotropic noise uses seed 0 and the training global standard
deviation, with validation draws continuing the training draw. These are fixed
stress-test draws, not independent experimental seeds. Do not renormalize the
transformed vectors: retrieval's cosine arithmetic and the standardized probe
retain their existing internal normalization. No neural inference is repeated.
Pure positive scaling should preserve exact cosine neighbour ordering; changes
in geometry, probe scores and retrieval are distinct observations.

Completed extraction chunks are reused under the existing strict provenance
checks. C1 needs its actual sampling description in provenance rather than the
C0 lexicographic selection description; parameterizing that description is the
only extraction-path adjustment. Preserve C0 defaults and tests.
Stop after the pilot; C1-main IDs and conclusions remain unfrozen.

## Completed C1-pilot (2026-10-07)

The frozen protocol above was executed without changes to sampling, inference
settings or severity grid. The [pilot report](../experiments/phaseC_c1_pilot/README.md)
contains the census, all class supports, pristine metrics, controlled responses
and main-run recommendations. The original document hash at ID freeze is
preserved by protocol-before-inference.md beside the manifests.

[measurement] Extraction took 5706.03 s for 512 vectors, with 3.59 GiB peak RSS.
The full 30-row sweep ran from the cache. 392 tests passed, one optional
accelerator test skipped; Ruff and lock checks passed. Completed-cache resume
checked every source hash with zero inference and identical geometry; analysis
resume performed zero endpoint calls. No source-byte duplicates or split
overlap were found.

[open] Unscaled severe contraction/interpolation reaches the C-grid ceiling;
two severe mean-injection fits fail convergence. These limits must be considered
before choosing the main probe protocol. Rare-class support and the small
validation sample constrain interpretation. Main IDs remain unfrozen; C2–C4
remain deferred. Existing Phase-A/B headline findings are unchanged.
