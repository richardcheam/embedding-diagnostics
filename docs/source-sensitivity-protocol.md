# Post-audit source exclusion sensitivity protocol

This is an authorized post-audit sensitivity analysis on the previously examined
representation sample. It is neither independent replication nor a new
preregistration of historical hypotheses. Implementation/tests/protocol commit
first; sample and freeze commit second; only then compute scientific endpoints.
No outcomes may change conditions, C grids, quantization, bins or support rules.

## Input, exclusion and source qualification

Accepted source/audit commit: `4343421`. Use the exact canonical 3,000×768 FP32
cache, its 2,000 training and 1,000 validation rows, and accepted C1/C2/C3 records.
Exclude exactly the 17 classification-4 IDs in `phaseC_source_audit/flagged_rows.json`.
This is an explicit audited list, not a prefix filter. All exclusions are physical
validation rows. Preserve global and within-split ordering, all training bytes,
label codes and remaining 983 validation rows. Exclude from both queries and
gallery candidates. Save the complete retained selection, exclusion IDs,
supports and historical eligibility in this namespace's sample manifest.

[ours] The audit establishes retained source-fragment provenance: 2,000 selected
training rows in fragment 0 and 983 validation rows in fragment 1 of local
BDD100K-enriched version 55. It does not independently authenticate these images
against the original BDD release, validate every label or exclude re-encoded
near-duplicates. The pilot's six different appended-fragment IDs remain
separately unresolved; no pilot rerun or further decoding is in this task.

Keep every accepted artifact and the canonical cache unchanged. New artifacts
are confined to `experiments/phaseC_source_sensitivity/`; update current status
and append current-document qualification, not historical experiment files.
No model loading/inference, source-image access, new sampling/data/encoder,
dependency change, ANN, new precision condition or C4.

## Fixed representations and transformation parameters

C1: pristine plus all original 30 transform/severity records: scale_contraction,
rank_truncation, mean_injection, isotropic_noise, mean_interpolation; severities
0, .25, .5, .75, .9, .99, seed 0. Reuse the accepted `fitted_transform` definitions.
Fit any data-dependent transformation on unchanged TRAIN only; transform the
**original full** validation split before selecting retained rows, preserving
noise stream assignment and the training-fitted shared rank projector. No final
renormalization beyond the original evaluator's internal cosine/scaler behavior.
Bind transformed train, full validation and retained validation byte hashes.
Zero-severity conditions remain distinct recorded conditions despite identity.

C2: exactly native_768, mrl_512, pca_512, mrl_256, pca_256, mrl_128, pca_128.
Load the existing training-only mean, nested 512-component PCA basis and variance
arrays from accepted C2's ignored transform cache; verify array hashes against
accepted C2 results. No new PCA fit. Preserve centered, unwhitened shared
projection and row L2 normalization. MRL uses official leading coordinates and
mandatory renormalization; native preserves FP32 bytes. Analysis arithmetic is
unchanged FP64 for compression. No additional controls or rank dimensions.

C3: all eight original conditions, in their original order. Native reference,
FP16 gallery/both, train-calibrated INT8 gallery/both, native FP32 arithmetic,
mean99 reference, mean99 FP16 gallery. Reuse training-only coordinate scales
max(abs(train))/127, zero-column scale 1, ties-to-even, clipping [-127,127].
Freeze scales and their hash; never recalibrate on validation. Reuse unchanged
training-fitted .99 mean injection and round storage before row normalization.
No change to query storage, calibration, accumulation or stress severity.

## Probes and support

Use C1's standardized primary and secondary unscaled probe; C2 standardized
primary only; no C3 probes. C grids, seed-0 80/20 TRAIN selection, outer-train
scaler, LBFGS tol=1e-8/max_iter=5000 and first-grid-value ties are unchanged.

[ours] No saved classifier models are available. Historical selected C and all
inner candidate fit records *are* saved train-only fitted parameters. Reuse
those unchanged choices/records, bound to historical source/condition/training
bytes, rather than retuning them. Reproduce the final classifier fit and scaler
using the existing implementation and full original ProbeConfig. Selection is
not relabelled fixed-C: its recorded inner diagnostics remain attached with an
explicit reused-selection/refitted-final-model provenance field. Reuse results
for exactly identical train/held-out inputs and selection/configuration within
an invocation; no approximate equivalence cache. Report original versus refit
training accuracy, iterations and convergence to expose solver repeat differences.

Keep the original train>=20/validation>=10 **eligible class codes** fixed; do not
recompute eligibility after exclusion. Overall accuracy includes all 983 rows.
Balanced/F1 scoring uses eligible present classes under the inherited evaluator,
with every absent eligible class explicitly unscorable (null), the unchanged
eligible list and the scored denominator disclosed. All vocabulary supports are
reported. Tunnel, gas stations and foggy have zero validation support and no
per-class score. They were not eligible in the historical balanced probe.
Preserve selected C, boundaries, iterations/max_iter, convergence/warnings,
training and validation accuracy/BA and all original inner candidate diagnostics.
A refit or boundary warning is not representation failure.

## Geometry, retrieval and floors

Re-evaluate validation geometry with established instruments, preserving raw
RankMe and centered covariance participation ratio. C2 additionally retains
RankMe/D and participation_ratio/D as descriptive capacity fractions. No Phase-B
absolute thresholds. Training outputs may be computed by the existing evaluator,
but all input/training transformation hashes must remain historical-compatible.

C1/C2 retrieval preserves FP64 normalization, dot and NumPy argpartition top-10,
self-exclusion and support>=10 macro convention. Do not replace their tie rule
with C3 ordering. Report micro, macro, per-class supports/scores (descriptive
rare-class scores, null for zero support), majority and balanced constant-class
floors, and inherited sum(p²) chance with the revised validation frequencies.

C3 normalization/scoring preserves explicit FP32/FP64 arithmetic and cancellation
witness. Exact order is descending cosine then ascending image_id; self excluded
by aligned unique ID. Rebuild all 983-row reference neighbours, top10/top11
margins and bins independently for native and mean99. Unchanged inverted_cdf
quartiles, unique cuts below maximum, searchsorted(side=left); ties remain
unsplit and may yield fewer than four bins. Keep actual dtype/backend witnesses.
Report sum(p²) and separately exact self-excluded chance
sum n_c(n_c-1)/(N(N-1)), with N=983. Bounds remain mathematical sufficient
conditions plus empirical floating-point checks: prospective reconstruction
bounds are separate from retrospective observed score errors, and are not
rigorous floating-point certificates. Arithmetic-only prospective coverage
remains inapplicable. Retain all numerical/overlap/turnover/bin/error exceptions.

## Comparisons and supplemental gallery diagnostic

For each endpoint report historical versus reduced absolute values and their
change. Also report the original condition-minus-corresponding-baseline effect,
the reduced condition-minus-reduced-baseline effect, and the difference between
those effects. C1 baseline is pristine; C2 native_768; C3 native_ref or mean99_ref
according to condition. Preserve C2 matched-D MRL-minus-PCA comparisons and their
change. Report geometry, probe accuracy/BA/F1 and both C1 probe variants,
retrieval/macro scores, floors/support and C3 numerical/identity endpoints.

Supplement: hold the **same 983 retained queries** fixed, comparing historical
1,000-row gallery with reduced 983-row gallery. C1/C2 use the inherited retrieval
arithmetic/argpartition in a rectangular-gallery wrapper, with original-position
self-exclusion; no new tie ordering. C3 projects accepted per-query records onto
the 983 query IDs for the historical-gallery diagnostic, alongside new reduced
records; no extra precision conditions. Report query-removal and gallery-removal
P@10 components, and C3 identity-change/overlap effects under both galleries.
For rectangular search, random-neighbour chance is conditional on retained
query labels: average (gallery_count_of_query_label-1)/(gallery_N-1). Also retain
the descriptive gallery sum(p²) convention, without equating it to this
conditional floor. Rebuild reduced C3 margin bins instead of reusing old bins.
This diagnostic is supplemental, not a tuned intervention or independent result.

Distinguish neighbour identity from relevant-attribute counts, entering/departing
attribute multisets and unmeasured instance semantics. Interpret point differences
descriptively; no new equivalence threshold, confidence interval, manufactured
seed or population-robustness verdict. Keep stable/null conditions and explain
that numerical fit repeats can contribute to probe endpoint differences.

## Freeze, execution and stop

Freeze binds accepted history/audit bytes, cache/sample/matrix hashes, exclusion
list and retained order, transformed input hashes, saved PCA file/array hashes,
INT8 scales, probe configs and historical train-selection records, existing
condition/code hashes, normalization/scoring/tie/bin rules, library versions,
CPU/BLAS backend and four threads. Parameter construction/hash calculation at
freeze does not compute diagnostic or semantic endpoints. The runner requires a
committed, unchanged freeze/sample and rejects mismatches before execution.

Atomic checksummed records resume in frozen order. Completed resume must execute
zero endpoints and preserve results and original verification bytes. Verify all
historical artifact hashes against accepted Git bytes before/after, and reload
canonical chunks/matrix after the run. Guard network, image opening and encoder
construction. Tests are synthetic/model-free and exercise exclusions, alignment,
zero support, self-exclusion, gallery denominators, historical preservation,
reuse of training selection, freeze identity and zero-call resume.

Run `uv run pytest`, `uv run ruff check .`, `uv lock --check`. Commit tested
implementation/protocol, then prepare/commit sample and freeze, then run with
four CPU/BLAS threads. Generate comparison/fit/gallery/margin tables, record
exceptions, runtime/RSS and protocol deviations. Commit results/interpretation
and current-document updates. STOP; independent replication requires separately
scoped authorization, and the pilot, ANN/C4 and further precision work do not run.
