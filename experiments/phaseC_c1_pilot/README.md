# C1-pilot — exploratory external reference

Status: complete, PILOT / EXPLORATORY. No confirmatory result or main-run subset is frozen.

The metadata census projects only image ID, physical split and the three
scenario attributes. The dataset-provided embedding column is forbidden and
never requested. Sampling uses fully-labelled examples uniformly without
replacement, seed 0, independent physical-split RNGs, 384 train / 128 validation,
no balancing or coverage constraints. Exact IDs and supports are in
sample_manifest.json. protocol-before-inference.md preserves the document
whose hash was recorded at sample freeze.

[measurement] Training has zero gas-station and tunnel examples. Validation has
two gas-station and one tunnel examples; these are inadequate for per-class
results. Macro endpoints require 10 validation examples per class and explicitly
list exclusions. Micro endpoints include all 128 validation examples.

The cache is canonical FP32 768d pinned EmbeddingGemma 2 output, masked mean
pooled and L2-normalized. CPU / batch 1 / four threads; audio excluded. The
cache directory is gitignored. Small manifests and results are retained.
Completed four-image chunks are resumable. Extraction_manifest.json copies the
complete authoritative cache manifest, including source image SHA256 hashes.

Probes fit only training vectors/labels and use the existing corrected
train-only C selection protocol. All five controlled transformations use the
existing severity grid, with data-dependent quantities fitted on train.
Validation retrieval is exact cosine search with self-matches excluded.
The reported retrieval chance floor retains Phase B's sum(p²) convention;
its exact finite-sample self-excluding counterpart is slightly smaller.
No threshold panel is used: Phase-B absolute scale thresholds do not transfer
to unit-normalized pretrained representations.

Reproduce locally (all model loads are offline):

```bash
uv run --locked python scripts/prepare_c1_pilot.py \
  --dataset /mnt/hdd/data/datasets/BDD100K-enriched
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  uv run --locked python scripts/extract_c1_pilot.py \
  --model /mnt/hdd/data/hf/hub/models--google--embeddinggemma-2/snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  uv run --locked python scripts/evaluate_c1_pilot.py
uv run pytest
uv run ruff check .
```

The preparation command rejects a changed frozen census or sample rather than
overwriting it. Analysis checkpoints every stress row and rejects changed
source/cache/runtime provenance on resume. Severity zero reuses the pristine
measurement exactly. Both standardized and unscaled probes retain convergence,
selected regularization and feature-scale diagnostics.

C1-main, C2, C3 and C4 are not started.

## Observed run [measurement]

Extraction/cache: 5706.03 s (95.10 min), 0.089730 images/s, 11.145 s/image. Peak RSS 3679.17 MiB (3.59 GiB). Sampled process swap peaked at 1.20 GiB; monitoring began after model load and sampled every ten seconds. No batching benchmark was run.

Shape [512,768], all finite, norm range [0.9999998211860657, 1.0000001192092896], cache 1775229 bytes (1.69 MiB). CPU FP32 / batch 1 / four threads, 438,760,448 text+vision parameters, audio excluded. The pinned Transformer -> mean Pooling -> Normalize sequence was verified.

| Attribute / class | Train | Validation |
| --- | ---: | ---: |
| scene: city street | 239 | 75 |
| scene: gas stations | 0 | 2 |
| scene: highway | 93 | 35 |
| scene: parking lot | 1 | 2 |
| scene: residential | 51 | 13 |
| scene: tunnel | 0 | 1 |
| timeofday: dawn/dusk | 29 | 11 |
| timeofday: daytime | 184 | 55 |
| timeofday: night | 171 | 62 |
| weather: clear | 242 | 76 |
| weather: foggy | 1 | 0 |
| weather: overcast | 44 | 18 |
| weather: partly cloudy | 39 | 9 |
| weather: rainy | 28 | 15 |
| weather: snowy | 30 | 10 |

## Pristine geometry [measurement]

| Metric | Train (384) | Validation (128) |
| --- | ---: | ---: |
| total_variance | 0.251561397 | 0.255685585 |
| mean_feature_std | 0.0177783382 | 0.017898157 |
| min_feature_std | 0.0113250336 | 0.0112182574 |
| mean_pairwise_cosine | 0.748438604 | 0.744314413 |
| std_pairwise_cosine | 0.0715630106 | 0.075387437 |
| rankme | 194.617439 | 77.7816026 |
| participation_ratio | 34.9636414 | 30.4937722 |

RankMe is raw-matrix spectral entropy; participation ratio uses centered covariance eigenvalues. Different split sample counts impose different spectral caps. No Phase-B threshold verdict is applied.

## Held-out endpoints [measurement]

U = unscaled; S = standardized. All 128 validation rows enter accuracy and micro P@10.

| Attribute | Accuracy U/S | Balanced U/S | P@10 | Macro P@10 | Val majority | Retrieval chance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| weather | 0.757812/0.742188 | 0.674708/0.668129 | 0.560937 | 0.347303 | 0.593750 | 0.397095 |
| scene | 0.804688/0.796875 | 0.733138/0.728694 | 0.582031 | 0.517455 | 0.585938 | 0.428955 |
| timeofday | 0.882812/0.882812 | 0.668035/0.668035 | 0.763281 | 0.579306 | 0.484375 | 0.426636 |

Balanced/macro scores include weather: clear, overcast, rainy, snowy (4/6); scene: city street, highway, residential (3/6); timeofday: all three classes. Partly cloudy has nine validation examples and is excluded by the existing support floor. Foggy has no validation examples. Gas stations and tunnel are absent from training. Uniform full-vocabulary probe chance is 1/6 for weather/scene and 1/3 for timeofday. The training-majority predictor scores 0.429688 on timeofday validation; the table reports the validation-majority convention (0.484375). The retrieval floor is the Phase-B micro sum(p²) convention, not a macro floor; exact self-excluding chance is (N*sum(p²)-1)/(N-1).

All pristine final fits converged and none flagged training underfit. Selected C (U/S): weather 100/100000, scene 10/0.01, timeofday 100/0.01. Raw feature scale is 0.0177551742. Full iteration counts, inner-selection scores and warnings are in pristine_metrics.json.

## Controlled responses [measurement]

Endpoint values below are severity .99; the complete six-point grid is in stress_results.json. P@10 columns are weather / scene / timeofday.

| Intervention | Val variance | Val cosine | RankMe | Participation ratio | P@10 W/S/T |
| --- | ---: | ---: | ---: | ---: | --- |
| scale_contraction | 2.55685585e-05 | 0.744314413 | 77.781603 | 30.493772 | 0.560937/0.582031/0.763281 |
| rank_truncation | 0.0524643968 | 0.937317207 | 2.559694 | 2.667964 | 0.511719/0.582031/0.821875 |
| mean_injection | 0.255685585 | 0.999973921 | 1.479996 | 30.493772 | 0.560937/0.582812/0.764062 |
| isotropic_noise | 9852.2722 | -0.000190779 | 125.131217 | 108.714763 | 0.373437/0.416406/0.407031 |
| mean_interpolation | 2.55685585e-05 | 0.999966232 | 1.549787 | 30.493772 | 0.560156/0.582031/0.761719 |

[measurement] Scale contraction preserves every recorded P@10 and standardized probe accuracy; variance scales by (1-severity)². Mean interpolation also preserves standardized probe accuracy at every severity. Rank truncation retains 288, 192, 96, 38, then 4 train-SVD directions. At four directions, timeofday micro P@10 rises from 0.763281 to 0.821875, while weather unscaled accuracy becomes 0.593750 with balanced accuracy 0.25 and training-underfit flags on both probe variants. Scene ends at the same micro P@10 as pristine, but its macro P@10 changes; equal aggregate P@10 does not establish identical neighbour ordering.

[measurement] Mean injection preserves centered variance/participation ratio and both probe accuracies at all severities while cosine approaches one and raw RankMe falls. Weather and scene unscaled fits at severity .99 do not converge within 5000 iterations; timeofday takes 4989. Noise raises participation ratio and RankMe while held-out endpoints decrease. Severe scaling/interpolation selects C=100000 for every unscaled probe.

[interpretation] These controlled responses show diagnostic invariances and geometry/semantics dissociations in this pilot sample. They do not establish what happens naturally during training or a confirmatory external-validity finding. Positive scaling, constant translation and mean interpolation below severity one remain invertible; raw-probe changes can reflect finite regularization grids or conditioning rather than removed information.

## Verification and main-run decision

[measurement] 392 tests passed, one optional accelerator test skipped; Ruff and lock checks passed. Completed-cache resume checks all 512 raw-image hashes under a TCP guard, performs zero neural inference and reproduces geometry exactly. Analysis resume uses sentinels to prove zero endpoint or transformation calls and unchanged results. There are no duplicate source hashes or train/validation source-byte overlaps. No dataset-provided embeddings, threshold panel, MRL compression or ANN study is used. Endpoint computation took 501.23 s, excluding extraction.

[interpretation] Provisionally recommend 5000 train / 2000 validation for common-class C1-main: 21.67 extraction hours at measured throughput. Phase-B-matched 5000/8000 costs approximately 40.24 hours and offers stronger rare validation coverage; that tradeoff remains a decision before ID freeze. Expected uniform 5000/2000 supports are only 0.89 training tunnels, 2.03 training gas stations and 8.83 validation tunnels, so this recommendation cannot support claims across every rare class.

[open] Decide scale calibration/C-grid range and severe-offset convergence policy before main. The pilot does not settle these instrument limits. Rare-class results remain limited; 128 validation observations are exploratory. No main subset is frozen and no main extraction has started. C2–C4 remain deferred.
