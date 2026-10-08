# C2 protocol — frozen before endpoint computation

C2 is a predeclared analysis on the previously frozen C1-main representation
sample, not an independent new-sample replication. C1 accepted source commit:
`0c19a85`. Do not enlarge the sample or rerun EmbeddingGemma inference.

## Audit prerequisite and fixed input

[measurement] `experiments/phaseC_c2/rank_transform_audit.json` verifies C1's raw
SVD projector is fitted on TRAIN only and applied in one common ambient space
to both splits. Neither an independent validation fit nor a concatenated fit
occurred. Its severe eight-direction validation output/geometry matches the
historical record exactly. H4 is not superseded; H1–H3 remain unchanged.

Use the integrity-checked canonical 3000×768 FP32 cache, exact sample IDs and
physical split fields: 2000 train / 1000 validation. Bind cache manifest (which
contains every chunk checksum), matrix bytes, sample manifest, C1 provenance,
C2 implementation commit and source/runtime hashes in `phaseC_c2/freeze.json`.
The runner checks that the freeze is committed and unchanged before endpoints.
No image decoding, Lance scan, Hugging Face model load or neural inference.

Pinned model: google/embeddinggemma-2,
914f7f89142e33e77833254d9c9b90c3cef7303b. The local pinned model card explicitly
supports 768/512/256/128, with leading learned MRL coordinates followed by L2
renormalization. Its hash and selected configuration evidence are recorded.
Method provenance: Kusupati et al., *Matryoshka Representation Learning*,
NeurIPS 2022, arXiv:2205.13147. Tests validate plumbing, not the paper's claims.

## Representations

Exactly seven primary representations, in this order:
`native_768`, `mrl_512`, `pca_512`, `mrl_256`, `pca_256`, `mrl_128`, `pca_128`.
No random projection or new stress grid is included.

Native is an exact copy of cached FP32 vectors. Its existing near-unit rounding
is retained, to reproduce C1. For MRL D, take the learned first D coordinates
and divide each row by its L2 norm. Fail on zero or nonfinite projected norms;
never silently remove a sample. Compressed analysis arithmetic/output is FP64
for both mechanisms; this does not change canonical FP32 model inference or
constitute a C3 storage-precision experiment.

PCA is centered, unwhitened sklearn PCA with full deterministic SVD. Fit once
on the canonical TRAIN vectors only, up to 512 components. For each D use its
first D basis rows. Apply `(x - train_mean) @ basis[:D].T` to BOTH train and
validation, then L2 normalize every row. No validation fitting, independent
validation mean or basis, randomized solver, whitening, labels or quotas.
Record training mean/basis/variance parameter hashes and fitting row count;
the reproducible parameter array may be saved in gitignored `transform-cache/`.
All three PCA dimensions share a nested train-fitted coordinate system.

Centered PCA differs from MRL in both basis construction and common-offset
removal. The experiment compares these declared mechanisms; it does not isolate
centering as a separate causal factor. Standardized feature-wise regularization
can also depend on coordinate system; probing is one endpoint, not an intrinsic
rotation-invariant definition of semantic information.

The severe C1 rank-loss intervention (raw, uncentered SVD, eight directions,
reconstructed into 768d, without final L2 normalization) remains a separately
labelled supporting historical reference. It is not a matched-D PCA baseline
and is not rerun or included in C2 matched comparisons.

## Endpoints and probe protocol

For weather, scene and timeofday separately: standardized linear-probe accuracy,
20/10-eligible balanced accuracy, exact cosine retrieval P@10. Reuse C1's
training support >=20 AND validation support >=10 classes unchanged. Fit on
all training rows; overall accuracy and retrieval retain every validation row.
Retain support, majority accuracy floors, balanced constant-class floors,
retrieval chance sum(p²), supplemental macro/per-class retrieval with validation
support >=10 and full-gallery retention.

Use the corrected C1 standardized ProbeConfig unchanged: C grid .01,.1,1,10,
100,1000,10000,100000; LBFGS tol 1e-8, max_iter 5000; seed-0 train-only 80/20
inner C selection by accuracy, first-value tie breaking. The existing scaler
fits on outer train before inner selection; this inherited convention remains
explicit. Outer validation never fits scaler, probe or PCA. The unscaled C1
numerical-sensitivity probe is not a C2 primary endpoint and is not rerun here.
Store all candidate and final selected C, boundary, convergence, iterations,
max_iter flag, training/validation accuracy and balanced scores. Retain failed
fits with warnings; do not adapt grids/solver after seeing endpoints.

Geometry on train and validation: all existing canonical metrics, including raw
RankMe, covariance participation ratio, total variance and mean cosine. Also
report descriptive `RankMe/D` and `participation_ratio/D`, alongside the raw
metrics. D is nominal coordinate capacity, not a claim about fitted numerical
rank. With 2000/1000 rows, D<=768 is the relevant nominal bound (centering also
bounds covariance rank by N-1). Fractions do not substitute for historical
metrics or representation-health verdicts. The inherited RankMe epsilon may
permit tiny capacity-bound excess; fractions are not clipped or redefined.
All compressed rows are normalized;
no Phase-B absolute scale thresholds or transferred panel verdicts.

## Predeclared hypotheses and comparisons

**C2-H1:** Dimensional reduction need not imply proportional semantic loss.
For MRL 512/256, report each endpoint's paired differences relative to native,
with floors/support. Do not define a post-hoc 'substantial retention' threshold;
no pass/fail or equivalence claim is assigned from descriptive differences.

**C2-H2:** Equal nominal D need not imply equal downstream utility. Report
MRL minus PCA at 512/256/128 for every primary endpoint. No preferred winner.

**C2-H3:** 128 may expose a larger utility cost than 256/512. Report the MRL
128-minus-256 and 128-minus-512 endpoint differences and native-relative effects;
report the corresponding PCA dimension response separately. The pinned model
card's multimodal degradation is a prior, not a BDD result. No materiality
threshold is invented after observing endpoints; negative transfer is reported.

**C2-H4:** Raw rank measures are not semantic quality on their own. Compare
raw and capacity-normalized geometry with all three task endpoints across
nominal dimensions and construction mechanisms. Report whether rank and utility
orderings agree/differ descriptively, without selecting a new universal metric
threshold or attributing controlled changes to natural training dynamics.

Record native-relative and matched-D effects for all seven representations,
never best dimension or best metric. Native C2 endpoints are computed under
this freeze and compared to saved C1 pristine endpoints for continuity; any
solver last-bit differences are reported, not silently overwritten.

## Uncertainty, execution and stop

This previously examined fixed sample/encoder has no independent training-seed
replications. Repository paired training-seed Student-t intervals are unavailable.
Report paired point effects, convergence limitations and class support; do not
manufacture seeds, post-hoc confidence intervals or significance verdicts.
Attribute-specific optima and any further mechanism conjectures are exploratory.

Implementation and tests commit first; then prepare and commit the freeze
before any C2 semantic endpoint computation. Resume result records only when
freeze/source/cache/parameter hashes match exactly. Generated result-file
changes must not change resume identity. Cache arrays remain gitignored.

```bash
uv run pytest
uv run ruff check .
uv lock --check
uv run --locked python scripts/run_c2.py --freeze
# Commit the freeze before running endpoints:
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  uv run --locked python scripts/run_c2.py --run
```

Return measurements, geometry/semantic tables, native/matched deltas,
convergence diagnostics, H1–H4 assessments and limitations. STOP after C2;
C3 numerical/ANN robustness and C4 cross-modal retrieval remain deferred.
