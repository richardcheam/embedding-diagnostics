# C3 protocol: storage rounding, neighbour identity and attribute utility

Prepared 2026-10-08, before endpoint computation. Finalized from the approved
[direction review](research-direction-review.md). Commit tested implementation
and this protocol, then commit `experiments/phaseC_c3/freeze.json`, then execute.
No conditions, bins, quantizer or interpretation thresholds may change afterward.

## Scope and question

Does numerical approximation change exact cosine neighbour identities, and do
those changes alter weather/scene/timeofday P@10? Can reference similarity
margins and reconstruction error help explain where changes occur?

Native 768d is primary. The sole supporting stress is the **existing** C1
severity-.99, seed-0 training-fitted mean injection. No compression dimensions,
ANN index, probe refits, encoder calls, image decoding, dependency changes,
resampling, new dataset or C4. Accepted source is
`50d03a8d5726465a49ef60d426e652905646cefb`; historical C0/C1/C2 files and the
canonical cache remain unchanged. This uses a previously examined sample, not
an independent replication or a population uncertainty study.

## Why include the single stress control

[established] Code inspection, not new endpoint computation, shows C1's
`mean_injection` draws a seed-0 Gaussian direction, normalizes it, and adds
`direction * mean_train_row_norm * .99/(1-.99)` to every training row.
`fitted_transform` adds the identical training-derived offset to validation.
The operation uses FP64; no separate validation fit or severity selection.

[interpretation] The large common offset concentrates directions, making small
angular differences more susceptible to coordinate rounding and cancellation.
This supplies a declared stress control without choosing a severity from
quantization outcomes. It is not a natural failure mode of the pretrained model.
Use only stress reference and FP16 **gallery** storage. Do not add stress INT8,
stress query rounding or an arithmetic stress sweep. Preserve the unchanged
FP64 transformed coordinates: round storage **before** cosine normalization.

## Frozen conditions (eight, in this order)

| Condition | Input | Gallery storage | Query storage | Normalization / scoring |
|---|---|---|---|---|
| native_ref | canonical FP32 | original FP32 | original FP32 | FP64 / FP64 |
| native_fp16_gallery | canonical FP32 | FP16 | original FP32 | FP64 / FP64 |
| native_fp16_both | canonical FP32 | FP16 | FP16 | FP64 / FP64 |
| native_int8_gallery | canonical FP32 | train-calibrated INT8 | original FP32 | FP64 / FP64 |
| native_int8_both | canonical FP32 | train-calibrated INT8 | train-calibrated INT8 | FP64 / FP64 |
| native_fp32_arithmetic | canonical FP32 | original FP32 | original FP32 | FP32 / FP32 |
| mean99_ref | unchanged train-fitted stress FP64 | FP64 | FP64 | FP64 / FP64 |
| mean99_fp16_gallery | unchanged train-fitted stress FP64 | FP16 | stress FP64 | FP64 / FP64 |

Storage conditions reconstruct/upcast and L2 normalize; no normalization before
packing. The reference also normalizes its input for cosine scoring. FP64
analysis is not FP64 neural inference. FP16 is storage only. INT8 both uses the
same scales on queries/gallery. Arithmetic-only changes normalization and dot
arithmetic, retaining canonical FP32 storage. It is not ANN or a storage change.
Mean-control effects are relative to mean99_ref, never mislabelled as effects
relative to native. Native and stress reference computations are C3 ordering
controls; accepted C1 endpoints are not regenerated or overwritten.

## Calibration, alignment and ordering

Only the frozen 2,000 TRAIN rows calibrate INT8: scale_j=max(abs(train_j))/127
in FP64; zero columns use scale 1. Encode with nearest, ties-to-even rounding,
clip to [-127,127], store signed INT8. No zero-point, adaptation, validation
calibration or quantization-aware training. Record held-out clipping after
rounding and coordinates outside the calibration range separately. Record
scales/hash and actual packed gallery bytes including FP64 scales. FP16 uses
IEEE conversion through NumPy. Fail on zero/nonfinite reconstructions; discard
nothing. Analysis arrays in FP64 are not evidence of operational memory savings.

Exact 1,000 validation queries/gallery rows, self excluded by unique image ID,
k=10. Cache chunk checksums and every row's image_id/split/attributes must match
accepted manifests; row order remains frozen. Order by descending cosine then
ascending image_id. Detect boundary ties explicitly. Historical retrieval uses
argpartition; any tie-rule continuity differences are disclosed, not edited in
historical files. Gallery remains natural sampled distribution including rare
classes. No train-gallery switch or support-based row exclusion.

## Dtype verification

Normalization explicitly multiplies, sums, takes square roots and divides in
the selected FP32/FP64 dtype. Scoring uses matching NumPy inputs/output and
CPU BLAS dot with four threads. Record actual normalized query/gallery and
score dtypes, BLAS/library configuration and a synthetic matrix-dot cancellation
witness that distinguishes FP32 from FP64 accumulation behaviour. Also record
residual versus FP64 dot on the actual normalized scoring inputs. Output dtype
alone does not demonstrate internal accumulation precision. The synthetic
witness is an empirical backend check, not a proof of every kernel operation;
stop if it fails to distinguish the declared arithmetic paths.

## Margin logic and deterministic bins

[established] For reference top-10/top-11 scores, gamma_i=s10-s11. If all
candidate score changes are bounded by epsilon_i and gamma_i>2epsilon_i, the
reference top-10 **set** is preserved. This strict sufficient condition neither
orders neighbours inside the set nor implies turnover when it fails.

[established] For exact unit vectors, the prospective reconstruction bound is
`B_i=norm(q'_i-q_i)+max_j norm(g'_j-g_j)`. It uses reconstructed coordinates,
not new similarity scores. Test gamma_i>2B_i empirically for storage conditions.
The maximum includes the whole gallery and may be loose. Reference conditions
have zero reconstruction error. For arithmetic-only, this bound does not cover
similarity arithmetic; mark it inapplicable rather than claiming certification.

Retrospective epsilon_i is the observed maximum absolute score change over
non-self gallery rows. Check gamma_i>2epsilon_i against observed set stability.
Separate this retrospective check from prospective bounds. No rigorous
floating-point certificate is claimed. Report prospective exceptions; a
retrospective order-condition violation stops execution for instrument audit.

Bins are computed independently for native_ref and mean99_ref reference margins,
without labels or perturbed scores. Use NumPy `quantile([.25,.5,.75],
method='inverted_cdf')`; take unique cut values strictly below the maximum.
Assign with searchsorted(side='left'), so equal margins remain together. Report
actual cut ranges/counts: ties can yield fewer than four distinct bins, and all
identical margins yield one. No random tie splitting or after-outcome merging.

## Endpoints and attribute turnover

Per query retain reference/condition neighbour IDs, set overlap@10, boundary
margin/bin, maximum score error, reconstructed unit-vector displacement,
raw storage error/norm, sufficient-condition checks and clipping context.
Aggregate mean/max score error, mean overlap, changed-query counts, prospective
coverage/exceptions, retrospective coverage, storage bytes and runtime/RSS.
Bin tables report counts, overlap, changed queries, score error, bound coverage
and each attribute P@10 delta. No fitted prediction model or new health metric.

For each query take entering IDs=`new_set - old_set` and departing IDs=`old_set
- new_set`. Count their attribute labels separately; do **not** pair entering
and departing neighbours arbitrarily. Relevant-count delta is entering counts
of the query label minus departing counts of that label, divided by 10 for
P@10. Retain full label-count dictionaries and whether the attribute multiset
is unchanged. Distinguish: identical neighbour set; unchanged attribute P@10;
unchanged full attribute counts. None establishes unmeasured instance semantics.

Report weather/scene/timeofday micro P@10, paired deltas, per-class support/scores,
macro scores for validation support>=10 with the full gallery, and gain/loss/
unchanged-query counts. This retrieval rule is distinct from C1's 20/10 balanced
probe rule; no probes are fitted. Retain sum(p²) chance convention and separately
report the exact self-excluded random-neighbour floor
`sum_c n_c(n_c-1)/(N(N-1))`. [established] With fixed k and labels,
`abs(delta P@k) <= 1-overlap@k`; count-based turnover can preserve P@10 despite
changed IDs. Test this logic on synthetic fixtures, not as a novel theorem.

## Predeclared comparisons and interpretation

- C3-H1: Examine whether smaller reference margins coincide with more identity
  turnover for each fixed storage condition. Report all bins, without assuming
  monotonicity or optimizing a prediction threshold. Contrast informative and
  loose prospective bound coverage with retrospective checks.
- C3-H2: Examine whether neighbour-set changes and attribute P@10 changes differ.
  Count changed identities with unchanged relevant-label counts, and retain
  changes to other attribute counts rather than inferring instance equivalence.
- C3-H3: Compare native FP16 gallery effects with the fixed mean99 FP16 gallery
  stress response. Different outcomes show conditional numerical sensitivity;
  no claim about spontaneous encoder degeneration follows.
- Separate native FP32 arithmetic effects from storage effects. No equivalence,
  materiality or universal health threshold is introduced. Every condition and
  attribute is reported, including stable/null outcomes and exceptions.

If every condition is stable, report bounded robustness for these operations,
this gallery and these coarse endpoints. Sufficient-bound coverage may explain
some stability; low coverage does not contradict it. Stable cases cannot test
how well margins predict actual failures, establish large-scale ANN behaviour,
or validate instance/cross-modal semantics. Do not add bit depths or severities
to manufacture instability after observing that outcome.

## Provenance, resume, checks and stop

Freeze code/protocol/source hashes, accepted source commit, exact cache manifest,
canonical matrix/sample hashes, quantizer scale/hash, conditions, dimensions,
transformation, ordering, bin rule, versions, backend and four-thread settings.
Checksum all accepted Phase-C historical files against accepted Git bytes before
and after execution, and validate canonical chunks again afterward. Guard
network connections, encoder construction and image opening during execution.
Completed records are atomically persisted with provenance and payload checksums;
resume rejects changes or reordered/duplicate conditions and computes only
remaining records. Completed resume must preserve result bytes and call no
endpoint. Large derived arrays are not committed.

Run `uv run pytest`, `uv run ruff check .`, `uv lock --check` and affected
provenance/resume checks. Tests use synthetic CPU fixtures only. Commit protocol
and tested implementation first; commit freeze second; compute endpoints third.

Deliver immutable results, generated tables, margin/exception analysis, runtime,
verification and qualified interpretation. Paired point effects are descriptive
on a previously examined sample; query dependence/unknown sequence grouping
precludes invented training-seed intervals. Stop after C3 interpretation.
Independent replication, more precise relevance judgments and multimodal
extensions remain separate future scope decisions.
