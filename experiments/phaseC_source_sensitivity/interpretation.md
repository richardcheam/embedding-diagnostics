# Interpretation of the separately frozen exclusion sensitivity

[ours] The implementation/protocol commit is `2feef78`; the pre-execution freeze
is `97595f3`. Exactly the 17 audit-confirmed graphics were excluded from both
validation queries and gallery candidates. All 2,000 training vectors and 983
retained validation vectors remain in their original order. All 46 conditions
completed, with no condition, severity, C grid, quantizer or bin-rule change.
Accepted C1/C2/C3 artifacts and the canonical cache were not replaced.

[interpretation] The principal diagnostic dissociations remain visible on the
retained enriched-source fragments. Absolute endpoints and some effect sizes
change, particularly weather balanced accuracy. This is evidence about a fixed
post-audit subset, not an equivalence test, independent replication or proof of
population robustness. Removing erroneous source content does not improve the
encoder; it changes the evaluated cohort and gallery.

## Measurements: absolute values and matched effects

[ours] Pristine standardized accuracy rises by 1.0351/1.0074/0.9807 percentage
points for weather/scene/timeofday. Balanced accuracy rises by
1.7921/0.3729/1.0148 points. Pristine P@10 rises by
0.6437/0.7715/0.7963 points, to 0.578637/0.624415/0.827263.
The same 983 retained queries have unchanged pristine P@10 against the old and
reduced galleries (at most aggregation rounding). Thus those pristine retrieval
increases are query-removal effects, not improvements in retained-query galleries.

[ours] Floors also change: sum(p²) is now 0.398695/0.432172/0.434058 and majority
accuracy is 0.595117/0.585961/0.490336. Fixed balanced constant-class floors remain
0.2/0.333333/0.333333. Support denominators are in results.md: foggy, gas stations
and tunnel have zero validation support and null per-class scores. Parking lot
has five retained validation rows; its descriptive P@10 is not a supported
macro/rare-category finding. Original eligible balanced classes remain present.

[ours] All absolute and baseline-relative differences are reported separately in
[comparisons.md](comparisons.md) and [comparisons.json](comparisons.json), including
both C1 probe variants, macro retrieval, floors and all validation geometry.
The [gallery diagnostic](gallery_diagnostic.md) holds the same 983 queries fixed.
Unlike pristine retrieval, the severe-noise scene endpoint has a +0.7426-point
gallery-removal component. Exclusion effects cannot generally be inferred only
from the number of removed queries.

## C1: observations that persist and effects that change

[ours] **H1:** At severity .99, variance contracts to 0.0001 of pristine while
mean cosine, standardized accuracy/BA and P@10 remain invariant. The reduced
pristine variance is 0.257033; its mean cosine is 0.742967. This is the expected
positive-scale response, not a new universal health threshold.

[ours] **H2:** Severity-.99 mean injection raises mean cosine to 0.999973793
and reduces raw RankMe to 2.008, while centered variance and participation ratio
remain 0.257033 and 38.265. Standardized probe scores remain unchanged. P@10
changes are -0.0712/+0.1017/-0.0712 points for weather/scene/timeofday. Secondary
unscaled weather/scene fits still fail convergence at 5,000 iterations.

[ours] **H3:** Severity-.99 isotropic noise raises RankMe from 270.810 to
667.313 and participation ratio from 38.265 to 430.994, while P@10 effects are
-17.8128/-19.8576/-38.2808 points. Balanced-probe effects are
-44.1754/-34.6297/-44.1455 points. Higher apparent dimensionality still coexists
with worse label-neighbour utility under this manufactured noise intervention.

[ours] **H4:** The severe eight-direction, shared train-fitted rank projector
has validation RankMe 4.618 and participation ratio 4.393. Effects remain
attribute/endpoint dependent:

| Attribute | Standardized BA effect pp | P@10 effect pp | BA effect change from historical pp | P@10 effect change pp |
| --- | --- | --- | --- | --- |
| weather | -32.2694 | -5.1373 | -2.0345 | -0.0373 |
| scene | -0.7090 | +3.7335 | +0.3873 | +0.1535 |
| timeofday | -6.0261 | +2.7772 | -0.3151 | +0.1572 |

[interpretation] Geometry, linear decodability and nearest-neighbour ordering
remain distinct endpoints. The time/scene retrieval increases are descriptive
stress outcomes already observed historically; this post-audit run does not make
an optimum or attribute-specific surprise confirmatory. Weather's rank-loss BA
contrast changes by over two points, so “nothing changed” would be inaccurate.
Noise/mean-interpolation and every intermediate severity remain reported; no
condition was selected to preserve a preferred conclusion. Controlled changes do
not establish naturally occurring degeneration of this pretrained model.

## C2: learned versus post-hoc compression

[ours] MRL 512/256 P@10 remains within -0.0916/-0.5290 weather points,
-0.2136/-0.2442 scene points and -0.0814/-0.3154 time points of native. MRL 128
costs more: -1.5361/-1.6582/-2.0651 P@10 points and
-8.1154/-2.8214/-2.2426 balanced-probe points. These are point effects, not a
predeclared equivalence tolerance or a claim that the costs are negligible.

[ours] Matched-D differences retain their historical directions for primary BA
and P@10. MRL minus PCA (weather/scene/timeofday, percentage points):

| D | Weather BA | Scene BA | Time BA | Weather P@10 | Scene P@10 | Time P@10 |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | +7.4608 | +4.6648 | +1.6281 | +1.9939 | +0.4374 | -0.4273 |
| 256 | +2.2108 | +3.5292 | +1.5715 | +1.2920 | +0.2238 | -0.9257 |
| 128 | -7.4489 | -1.6735 | -3.9808 | +0.1424 | -1.4649 | -3.3367 |

[ours] At 128 dimensions, PCA's raw RankMe is 109.669 versus MRL's 71.885,
while its participation ratio is lower, 22.942 versus 30.516. Both raw measures
and descriptive RankMe/D and participation_ratio/D are available; these cannot
be collapsed into one representation-quality ordering. PCA time P@10 increases
as its dimension decreases (0.830722/0.833367/0.839980 at 512/256/128), whereas
MRL time P@10 declines. PCA centering and coordinate-dependent standardized
regularization remain confounds of attributing this to basis choice alone.

[interpretation] C2-H1's modest MRL 512/256 point losses, C2-H2's construction/task
differences, C2-H3's larger MRL 128 costs and C2-H4's rank/utility dissociations
remain descriptive observations on the retained subset. PCA is not uniformly
worse and dimensionality is not utility. Some effect sizes change: for example,
MRL512-minus-PCA512 scene BA grows by 1.1164 points after exclusion. Neither a
small effect change nor matching signs establishes statistical robustness.

## C3: identity, attribute utility and margins

[ours] Changed neighbour sets are 5/5/60/75 for native FP16 gallery/both and
INT8 gallery/both, zero for FP32 arithmetic, and 195 for mean99 FP16 gallery.
These counts are the same as historically, now over 983 queries. Inspection of
accepted per-query neighbour records establishes that none of the 983 retained
queries retrieved an excluded ID in any historical C3 top-10 condition. Their
old-gallery and reduced-gallery top-10 sets match exactly across all eight
conditions. The supplemental analysis does not establish instance equivalence.

[ours] Reduced FP16 gallery/both maximum score errors are 5.783e-5/8.158e-5;
INT8 gallery/both remain 3.952e-3/4.817e-3; FP32 arithmetic remains 7.123e-7;
mean99 FP16 gallery remains 2.946e-7. Reference-versus-reconstructed-reference
FP64 score differences reach 8.9e-16 with no set changes, reflecting last-bit
analysis arithmetic rather than a new storage condition.

[ours] Every attribute precision-effect change from the historical baseline is
below 0.0032 percentage points. Many identity changes preserve P@10: for native
INT8 gallery, 42/35/42 of its 60 changed sets retain weather/scene/time relevant
counts. Mean99 gallery has 131/119/149 of 195 such unchanged P@10 cases. Full
attribute multisets have distinct counts, recorded alongside entering/departing
label counts. Stable aggregate attribute utility does not measure layout,
objects, caption relevance or safety-relevant instance semantics.

[ours] Rebuilt native bins have 246/246/246/245 queries. Native FP16 changes
all lie in the smallest-margin bin; INT8 gallery changes split 55/5/0/0 and
INT8 both 66/9/0/0. Mean99 FP16 changes split 121/53/20/1 across its independently
rebuilt reference-margin bins. Prospective coverage is 576 for FP16 gallery and
398 for both; still zero for INT8 and the mean stress. Arithmetic-only coverage
remains inapplicable. No prospective empirical exception or retrospective
sufficient-order violation occurs, and no boundary ties are observed.

[interpretation] The margin/turnover association, identity-versus-coarse-utility
distinction and stronger mean-stress sensitivity remain visible. Prospective
bounds are mathematical sufficient conditions evaluated empirically and can be
loose; retrospective score errors are observations, not prospective predictors.
No rigorous floating-point certification is established. Failure of a bound is
not failure of the representation, and stable cases are retained rather than
adding precision conditions to manufacture an effect.

[ours] Training calibration remains unchanged. INT8 now has 372 coordinates
outside range, 353 clipped coordinates across 246 rows, versus historical
397/377/263, out of 754,944 reduced-gallery coordinates. Packed native/FP16/INT8
gallery sizes are 3,019,776/1,509,888/761,088 bytes (INT8 includes scales).
Reconstructed FP64 working arrays remain; no operational memory/speed or ANN
benefit was tested.

## Fit diagnostics, preservation and execution

[ours] 174 distinct final probe fits supply 207 recorded final-score entries,
with exact-input reuse for repeated zero severities/native. All recorded refit
training accuracies, iteration counts and convergence statuses match their own
historical final diagnostics. Historical model coefficients/predictions were not
saved, so coefficient-level identity is not independently testable. Train-only C
selection logs are reused unchanged; no grid was retuned on the new validation set.

[ours] All standardized final fits converge. Four secondary raw fits fail at
max_iter: weather under scale .99, weather/scene under mean injection .99 and
weather under mean interpolation .99. Of 207 reported score entries, 108 primary
standardized entries select the lower C boundary; seven raw entries select the
lower boundary and two raw timeofday entries select the upper boundary under
scale/interpolation .99. Full fit diagnostics and reused candidate warnings
remain visible. A grid boundary is not evidence of representation failure or
an optimized unconstrained decodability estimate.

[ours] Execution took 731.693 seconds (12.19 minutes), peak process RSS 874.43
MiB, CPU/four threads. Completed resume took 3.721 seconds, with **zero endpoint
calls and zero probe fits**, preserving result/report/freeze/original verification
bytes. All 491 accepted historical experiment files, saved PCA and canonical
cache remain unchanged. No inference, source-image access, resampling, dependency,
ANN, C4 or pilot work occurred. No deviations from the frozen protocol are recorded.
Tests: 458 passed, one optional accelerator skip; Ruff and lock validation passed.

## Open scope and recommendation

[interpretation] The source defect changes absolute utility and certain effect
sizes; excluding it does not authenticate the remaining 2,983 source images or
labels, supply independent samples, resolve temporal/near-duplicate correlation,
or establish finer relevance. The pilot's six distinct appended-fragment IDs
remain separately unresolved and its exploratory record is preserved. Population
uncertainty and coefficient-level historical fit identity remain unavailable.

[interpretation] Recommend a separately authorized independent replication with
an authenticated source release and independently frozen query/gallery sample,
using the same focused diagnostic questions and finer relevance where feasible.
Resolve source-image/label provenance and sampling/uncertainty units before
extraction or endpoints. This task stops at the post-audit sensitivity; no new
replication, multimodal, ANN or precision condition starts automatically.
