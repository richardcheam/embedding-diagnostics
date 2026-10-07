# C1-main protocol, prepared before main inference

This is the protocol-hardening and freeze stage. Main extraction has not begun.
The historical `experiments/phaseC_c1_pilot/` results remain unchanged. Improved
fit diagnostics are investigated separately in `experiments/phaseC_c1_hardening/`.
The accepted C0 extraction/cache/adapter remain unchanged.

## Frozen population and sample

Source: local BDD100K-enriched Lance version 55. Fully-labelled rows use the
canonical Phase-B vocabulary; unknown/undefined labels are excluded. Project
only image_id, split, weather, scene and timeofday for selection, adding
image_bytes only at extraction. Never request the dataset-provided embedding.

Exclude every one of the 512 pilot image IDs before sampling. Within each
physical split, sort eligible rows by image ID and sample uniformly without
replacement: 2000 train, 1000 validation. NumPy PCG64 uses
SeedSequence([1, split_index]), train=0 and validation=1. Globally order the
selected rows by image ID. No quotas, coverage constraints, replacement or
selection based on embeddings/endpoints. The exact IDs and counts are in
`experiments/phaseC_c1_main/sample_manifest.json` and cannot change after freeze.

A class enters balanced probe/F1 scoring iff its frozen training support is
at least 20 AND validation support is at least 10. All qualifying classes are
included, without performance-based exclusion. This is a pragmatic support
rule, not a power guarantee. It retains weather clear, overcast, partly cloudy,
rainy and snowy; scene city street, highway and residential; all three timeofday
classes. Train balanced accuracy uses the same eligible classes. Fit on every
training row, including rare labels; evaluate overall accuracy on every
validation row. No validation vector/label fits the scaler or classifier.

Retrieval uses every validation query and gallery row, exact cosine arithmetic,
P@10 with self-matches excluded. Report micro P@10 and its Phase-B sum(p²)
chance convention. Supplemental per-class/macro retrieval uses validation
support >=10, with every row retained in the gallery; this rule is distinct
from the two-split balanced-probe rule. No rare-category claims are targets.

## Canonical extraction

Pinned google/embeddinggemma-2 revision
914f7f89142e33e77833254d9c9b90c3cef7303b; text+vision, audio disabled.
CPU FP32, batch 1, four Torch/BLAS threads, chunks of four. Official processor,
attention-mask mean pooling, L2 normalization, 768d FP32 cached once.
No FP16 model inference, stack changes, quantization or dimension compression.
Existing strict resumable cache checks remain authoritative. Main commands
require a committed, unchanged freeze file and check frozen source/runtime
hashes before any model load. Resume identity uses the commit containing the
freeze, never result-file dirtiness or later artifact commits. Evaluation checks
cache extraction freeze provenance as well as exact IDs. Decode/inference
failures stop; no silent removal.

## Probe protocol and numerical diagnostics

Primary semantic probe: standardized existing logistic probe. Secondary:
unscaled raw features, explicitly a numerical-sensitivity diagnostic.
Standardized grid remains 0.01,0.1,1,10,100,1000,10000,100000.
Unscaled grid adds 1000000 and stops there. Both retain LBFGS, tol=1e-8,
max_iter=5000, train-only 80/20 inner selection with seed 0, accuracy selection
and first-grid-value tie breaking. The inherited standardized implementation
fits its scaler on outer training before inner selection; outer validation
never participates. Preserve this Phase-B convention explicitly.

Every candidate and final fit reports C, convergence, iteration count,
max_iter reached, train accuracy/balanced accuracy and validation accuracy/
balanced accuracy. Candidate scores concern the inner split; final validation
aliases concern the outer physical validation split. Candidate balanced scores
are descriptive recalls over present classes, with support counts; final
balanced scores use frozen class eligibility. Report C-grid boundary flags and
whether selection was fixed-C, inner selection or fallback. The historical
underfit_train flag remains descriptive: failing to beat training majority is
not a unique solver-failure detector. Nonconverged/boundary fits are retained
and qualified, not interpreted automatically as lost representation utility.

[interpretation] If X'=aX and w'=w/a, the same penalized logistic objective
requires C'=C/a². Pilot contraction a=.01 maps pristine weather C=100 to 1e6;
this restores its result without standardization. Values through 1e9 were
investigated; higher Cs can cause nonconvergence and are not adopted merely to
remove a boundary flag. The 1e6 cap is supported by the cached investigation,
not a guarantee that every main fit will find an interior optimum. Grid limits
remain explicit. Scale-aware C rescaling would align objective families more
cleanly, but would replace the inherited fixed-grid numerical-sensitivity
protocol. It is discussed here and NOT adopted silently.

[interpretation] Large common offsets preserve separability with an intercept
but worsen conditioning. Centering only restores convergence at fixed C and
unchanged feature scale in the cached investigation; standardized fits already
converge. Centered-only fits are investigative controls, not replacements for
the secondary raw probe. max_iter is not increased after seeing warnings.

[interpretation] Four-direction weather fits converge in equivalent
orthonormal coordinates and plateau under weak regularization without
recovering full-vector training performance. Their underfit flag therefore
cannot simply be dismissed as the original grid/solver failure; it also cannot
establish that all weather information is absent. This diagnosis is separate
from the frozen main, whose .99 rank intervention retains eight directions.

## Interventions and hypotheses

Use existing scale_contraction, rank_truncation, mean_injection, isotropic_noise
and mean_interpolation definitions; severities 0,.25,.5,.75,.9,.99. All
scales/means/raw SVD bases fit training only and apply identically to both
splits. Noise uses fixed seed 0 and training global matrix std; validation draws
continue the training draw. This seed does not define independent replications.
Do not renormalize transformed vectors; cosine retrieval and standardized
probes retain their internal arithmetic. Severity zero is exact identity.
Raw rank truncation retains round(768*(1-severity)), at least one direction:
576,384,192,77,8 for the five nonzero severities. This is destructive post-hoc
rank removal, not Matryoshka compression.

**H1 — positive scale contraction:** variance contracts as (1-severity)²;
mean cosine and exact cosine retrieval remain approximately invariant;
standardized linear decodability remains approximately invariant.

**H2 — common mean injection:** mean cosine approaches one; centered variance
and standardized linear decodability remain approximately invariant.

**H3 — isotropic noise:** participation ratio/RankMe may increase as noise
dominates while semantic retrieval deteriorates. Test their joint response,
not whether one rank measure independently classifies representation quality.

**H4 — rank truncation:** compare geometric dimensionality, standardized linear
decodability and nearest-neighbour retrieval responses. No monotonic semantic
ordering or attribute-specific optimum is assumed or preregistered.

Only H1–H4 are confirmatory, conditional on this frozen encoder/subset/protocol.
Mean interpolation, particular severity optima, attribute-specific surprises,
a particular rank improving retrieval and new metric orderings are exploratory.
The pilot-specific four-direction timeofday observation is not a main hypothesis.

## Effects, uncertainty and interpretation

Report pristine values and paired transformation-minus-pristine effects at
every predeclared severity, separately per attribute and probe variant. Include
class denominators, floors and every fit warning. No best-severity endpoints,
universal healthy/degenerate thresholds, significance verdicts or post-hoc
success thresholds. Mathematical invariance predictions have exact ideal
values; observed floating-point/solver differences remain reported.

The repository's Student-t intervals cover paired training-seed variability,
conditional on fixed evaluation images. One externally pretrained encoder has
no independent training-seed replications, so that uncertainty is unavailable;
do not create seeds to obtain intervals. Default freeze policy is descriptive
paired effects with training-seed uncertainty explicitly not estimable.
A paired validation-query bootstrap was offered as a separate option; it would
condition on fixed fitted probes/gallery and would not cover training/gallery
variability. It is not adopted without declaring that distinct scope.

Unit-normalized canonical vectors are not numerically comparable to Phase-B
unconstrained representations under absolute scale thresholds. Reuse metric
definitions, never transfer Phase-B panel verdicts. Distinguish changed geometry,
changed linear decodability and changed neighbour ordering. Equal aggregate
P@10 does not establish identical neighbours. Mathematical controlled responses
do not establish what occurs naturally during training.

## Freeze and execution

Freeze hashes cover sample/protocol/source/lock files and actual dependency
versions. Commit the freeze before extraction. No main embeddings exist at this
stage; no inference runs until that commit and its checks succeed.

```bash
uv run --locked python scripts/prepare_c1_main.py
uv run pytest
uv run ruff check .
# Only after the protocol/manifest/source freeze is committed:
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  uv run --locked python scripts/run_c1_main.py --extract
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  uv run --locked python scripts/run_c1_main.py --evaluate
```

[interpretation] Pilot throughput estimates 9.29 extraction hours for 3000
images, excluding loading and endpoint analysis. Report actual measurements
and deviations after the run. Do not start C2 automatically; review C1 first.
