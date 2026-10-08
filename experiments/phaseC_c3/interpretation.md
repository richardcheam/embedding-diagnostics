# C3 interpretation

This is the frozen, bounded numerical study on the previously examined C1 sample.
Results are descriptive paired responses; no equivalence threshold or population
interval was introduced. Native 768d is primary. See [tables](results.md),
[macro scores](macro_results.md), [exceptions](exceptions.json) and
[protocol](../../docs/c3-protocol.md).

## What was measured

[ours] Native FP16 gallery/both storage changed 5/1,000 neighbour sets, with
mean overlap@10=.9995. Native INT8 gallery/both changed 60/75 sets, overlap
.9940/.9925. Native FP32 normalization/scoring changed zero sets, despite maximum
score error 7.123e-7 and maximum arithmetic residual 6.474e-7 relative to FP64
scoring on its actual normalized inputs. The cancellation witness distinguishes
FP32/FP64 accumulation behaviour (448 versus 766); execution checked this and
the frozen four-thread backend before endpoints.

[ours] Native weather/scene/timeofday P@10 starts at .5722/.6167/.8193.
FP16 storage changes it by −.02/+.02/−.01 percentage points. INT8 gallery changes
it by −.04/−.01/0 pp; INT8 both by −.02/−.01/−.03 pp. These are point responses,
not formal noninferiority or evidence that every query remained semantically
stable. For example, INT8 gallery has nine timeofday gains and nine losses;
the unchanged aggregate hides 18 changed per-query relevant-label counts.

[ours] The unchanged training-fitted mean99 reference yields
.5715/.6177/.8186 P@10. Rounding only its gallery to FP16 changes 195 neighbour
sets (overlap .9803), with weather/scene/timeofday deltas +.04/+.02/+.18 pp.
Of the 195 changed sets, 131/119/149 preserve attribute P@10, but only
102/116/141 preserve the full respective label-count multiset. Entering and
departing labels are counted independently; no neighbour pairing is assumed.

## C3-H1: margins and identity turnover

[ours] Reference binning produces four bins of 250 queries for each input family,
with no tied boundary scores. Native FP16 changes occur only in the lowest bin.
Native INT8 gallery changes by bin are 55/5/0/0; INT8 both 67/8/0/0.
Mean99 FP16 gallery changes are 121/55/18/1. Score-error means vary much less
across bins than identity turnover does; full bin ranges and endpoints are kept.

[interpretation] Smaller reference margins are associated with more turnover in
these fixed conditions. This supports the declared local-margin account without
establishing a universal failure predictor or isolating quantization orientation
and clipping as causal factors. The quartiles were fixed by a deterministic
reference-only rule; no bin, severity or quantizer was selected from outcomes.

[ours] Prospective reconstruction-based sufficient checks cover 592 native FP16
gallery queries and 412 native FP16-both queries, with zero observed exceptions.
They cover zero native INT8 and zero stressed FP16 queries. Retrospective maximum
score-error checks cover 938/916 FP16, 32/34 INT8 and 45 stressed FP16 queries;
all covered sets remain unchanged. Native FP32 arithmetic has retrospective
coverage 999, with all 1,000 sets unchanged. Its prospective storage bound is
explicitly inapplicable to score arithmetic.

[interpretation] A sufficient condition can explain some stability while being
too loose to explain most stable cases. The global reconstruction bound has no
coverage for INT8 or the stress, even though most sets remain stable. Failure
to satisfy a bound is not failure of the representation. Retrospective observed
error is not a prospective predictor. Neither check is rigorous floating-point
certification, and the elementary inequalities are not new research results.

## C3-H2: neighbour identity versus attribute utility

[interpretation] Identity turnover and attribute utility are distinct: many
changed neighbour sets retain the query's relevant-label count, and aggregate
gains/losses can cancel. Full-label-count preservation is stronger than unchanged
P@10 but still says nothing about unmeasured instance semantics. These coarse
BDD labels cannot tell whether replacement images differ in safety-relevant
objects, layout, driving behaviour or caption relevance.

[interpretation] Near-native aggregate retrieval under the tested storage formats
is a workload-specific numerical robustness observation. It is not a general
storage recommendation or a novel discovery that quantization can preserve
retrieval. Macro/per-class endpoints remain available so dominant labels do not
hide denominators. No rare-class inference is made from tiny class supports.

## C3-H3: fixed angular-concentration stress

[ours] Under native FP16 gallery storage, maximum score error is 6.012e-5 and
only five sets change. Under the predeclared mean99 stress it is smaller,
2.946e-7, yet 195 sets change. Native median boundary margin is about 7.38e-4;
the corresponding stress median is about 7.71e-8. Stress raw row norms are
98.913–99.049, as expected from the fixed large offset. Storage rounding happens
before normalization in both cases.

[interpretation] Smaller absolute score error can coexist with more neighbour
turnover when relevant margins are smaller. The stress demonstrates conditional
numerical sensitivity, without showing lost aggregate attribute utility or a
naturally occurring pretrained-model defect. Positive P@10 changes under the
stress are descriptive label-neighbour changes; they are not evidence that
rounding improves the encoder or a basis for choosing this stress in deployment.

## Exceptions, provenance and runtime

[ours] No prospective or retrospective sufficient-check exceptions, nonfinite
reconstructions, dropped rows or boundary ties were observed. INT8 has 397
coordinates outside the training calibration range; 377 clip after rounding,
across 263 of 1,000 gallery rows. This uses 768,000 gallery coordinates. Per-row
range/clipping context and affected IDs remain tracked; these outliers may
contribute to the loose global reconstruction bound. Calibration was not changed.

[ours] Packed gallery storage is 3,072,000 bytes for native FP32, 1,536,000 for
FP16, and 774,144 for INT8 including 6,144 scale bytes. The analysis reconstructs
FP64 working arrays; these byte counts establish packed storage size, not an
operational search-memory or speed benefit. Mean99 reference is FP64 (6,144,000
bytes), so its storage comparison must not be mistaken for native FP32 savings.

[ours] Initial complete invocation: 6.226 seconds, summed condition time 2.562
seconds, peak process RSS 744.38 MiB; CPU and four threads. Completed resume:
zero endpoint calls, 1.897 seconds, unchanged result/report/freeze/original
verification bytes. All 40 accepted historical files and canonical cache remain
unchanged. C3 native reference matches committed C1 P@10 exactly; mean99 reference
differs only by 1.11e-16 on timeofday aggregation. Guards prevented network,
encoder construction and image opening. No protocol deviations or ANN/C4 work.

[ours] Verification: 441 tests passed, one optional accelerator test skipped;
Ruff and uv lock validation passed. The final protocol/tested implementation
commit is 443459f; freeze commit is 50f6659. Large vector arrays are not stored
in the C3 record; per-query IDs, scalar diagnostics and label counts are retained.

## Limits and next decision

[interpretation] Not every condition was identity-stable, but aggregate attribute
utility changed little. If all conditions had been stable, the study would still
establish bounded robustness under these operations and this gallery; it would
not establish failure prediction, universal equivalence, ANN scalability or
cross-modal/instance relevance. The observed failures here let the margin account
be checked, while retaining the same limitations on generalization.

[interpretation] The strongest next step is an independently frozen embedding
cache with finer instance relevance, ideally an existing public artifact with
trustworthy encoder/preprocessing provenance. Use the same declared numerical
controls and decide sampling/uncertainty units before evaluation. This requires
separate scope authorization. A later multimodal study should distinguish
attribute-prototype text queries from genuine paired captions and apply identical
MRL or image-trained PCA coordinates to both modalities. Neither starts here.

[interpretation] Unresolved: how this response transfers to larger galleries,
other encoders, sequence-independent samples and finer relevance; how useful a
less conservative prospective bound would be; and whether attribute-preserving
identity changes matter for actual scenario mining. C3 stops with these questions,
without adding conditions to manufacture larger effects or proposing a new metric.

[ours] Final record-only review also found 17 inherited validation IDs prefixed
`synthetic_val_`, with zero similarly prefixed training IDs. Their origin is
unexplained in inspected repository records; names alone do not establish
fabrication. No rows, frozen conditions or endpoints were changed after this
discovery.

[interpretation] The [source qualification](source_qualification.md) limits
claims about an exclusively original BDD population.
