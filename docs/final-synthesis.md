# Final synthesis: diagnostic reliability and utility

Experimental scope closed on 2026-10-09. Maintenance, preservation and transparent
corrections remain possible. No further Jina repair or experimental expansion is
part of this project. Audio–visual post-training is a separate project.

## What geometry reveals and misses

[established] Total variance measures centered spread; mean pairwise cosine measures
directional agreement. RankMe uses the raw singular spectrum; participation ratio
uses centered covariance eigenvalues. Positive scaling leaves both rank ratios and
cosine ordering invariant. A common offset changes raw geometry while preserving
centered variance. These mathematical responses are established properties.

[ours] Corrected Phase-B time-of-day probe accuracy is approximately 91.37% for the
contracted control and 92.71% for the EMA reference; retrieval P@10 is approximately
56.57% versus 67.47%. The earlier raw probe's near-majority result was a solver
artifact, withdrawn and superseded. Residual iteration-cap warnings qualify the
corrected probes. Geometry degeneration did not eliminate all decodable labels.

[ours] In the qualified pretrained analysis, severe positive contraction leaves
standardized probing and cosine retrieval unchanged; severe mean injection leaves
standardized probing unchanged. Severe noise increases RankMe while weather P@10
falls toward its chance floor. These are manufactured post-hoc interventions, not
evidence about what naturally happens during encoder training.

[interpretation] The useful contribution is the comparison of instruments and task
endpoints with their failure qualifications. No universal health threshold or new
representation-quality predictor is established. Phase-B absolute variance thresholds
do not transfer to unit-normalized pretrained outputs.

## Compression and different utility endpoints

[ours] The seven C2 representations compare native 768d with learned MRL and centered,
train-fitted PCA at 512/256/128d. Training and validation share the same fitted PCA
basis; the C1 severe rank intervention also passed the train-only/common-space audit.
MRL prefixes are obligatorily L2 renormalized. MRL 512/256 retain much of the measured
utility; 128 incurs larger losses for several endpoints. PCA and MRL have different
attribute/probe/retrieval orderings rather than one uniform winner.

[interpretation] Familiar learned compression behavior is not a novel algorithmic
finding. Raw rank and dimension-normalized capacity descriptors answer different
questions. Centering, coordinate construction, regularization and class support
limit causal interpretations of the PCA comparison. Small fixed-sample point
changes do not establish equivalence or population robustness.

## Numerical perturbation and neighbour identity

[ours] On 983 retained BDD queries, native FP16 gallery storage changes five top-10
sets; native INT8 gallery storage changes 60. FP32 normalization/scoring changes
none. Frozen severity-.99 mean injection followed by FP16 gallery storage changes
195 sets despite a smaller maximum score error than native FP16. Native FP16
changes all lie in the lowest boundary-margin bin; INT8 gallery changes distribute
55/5/0/0. All tested conditions and null results remain reported.

[established] A sufficiently large reference boundary margin relative to bounded
score perturbations is a sufficient condition for top-10 set preservation.
[ours] Prospective reconstruction-based conditions cover 576 native FP16-gallery
queries, with no empirical exception, but cover no INT8 or stressed queries.
[interpretation] These are mathematical sufficient conditions and empirical checks,
not rigorous floating-point certification; retrospective score errors are distinct.

[ours] Among 195 stressed identity changes, weather/scene/time-of-day P@10 remains
unchanged for 131/119/149 queries. Entering/departing label counts are recorded without
arbitrary neighbour pairings. [interpretation] Identity stability, coarse attribute
utility and unmeasured instance semantics must remain distinct. Reconstructed FP64
working arrays do not establish operational storage/search throughput benefits.

## Positive hits, ranking and multiple-caption coverage

[ours] The accepted COCO study uses 1,000 original-release validation image groups,
five human-authored captions each, and the same pinned EmbeddingGemma encoder. Native
768d, learned MRL 256d and 128d are the complete grid. At 128d, image→text Hit@10
falls 2.50 percentage points, Hit@1 falls 10.60 points and caption set recall@10
falls 11.34 points. At 256d the respective changes are −0.30, −0.70 and −1.74.
Identity overlap@10 is approximately .84 at 256d and .65 at 128d in both directions.

[ours] The conditional paired 95% intervals for Hit@10 differences are:

| Prefix | Text→image delta pp [interval] | Image→text delta pp [interval] |
|---|---:|---:|
| 256 | −0.20 [−0.54, +0.14] | −0.30 [−1.10, +0.50] |
| 128 | −3.34 [−4.06, −2.62] | −2.50 [−3.60, −1.40] |

[interpretation] Intervals crossing zero do not demonstrate equivalence. Preserving
at least one recorded positive does not establish unchanged first-rank retrieval or
recovery of all five captions. Coarse category and paired utility broadly decline
under stronger compression here; the study does not establish a strong opposing
ordering between category preservation and paired retrieval.

[ours] A native circular pairing control leaves vector geometry unchanged but reduces
text→image/image→text Hit@10 to 1.22%/.20%. [interpretation] Marginal distributions
cannot establish paired semantic alignment. Direction-specific caption-role prefixes
prevent isolating modality asymmetry. Three dimensionalities do not validate a
quality predictor. Original-caption relevance is incomplete; unknown pretraining
overlap and residual near/event duplicates remain unresolved.

## Sources, uncertainty and failed replication

[ours] The source audit confirmed 17 generated graphics inherited from the enriched
BDD validation fragment. The separately frozen exclusion sensitivity retains 2,000
training and 983 validation rows, removes the flagged rows as queries **and** gallery
candidates, and preserves the historical 3,000-row cache and results. This sensitivity
is the current qualified C1/C2/C3 analysis. Verified enriched-fragment continuity is
not independent original-BDD authentication. Tunnel, gas stations and foggy have zero
validation support; six pilot appended-fragment IDs remain separately unresolved.

Training-seed paired Student-t intervals (five seeds, one fixed split) belong to
Phase B. BDD Phase C uses fixed-sample descriptive effects. COCO uses conditional
paired parent-group bootstrap intervals (5,000 draws, fixed gallery). These are
neither independent encoder replication nor uncertainty over encoder training.
Phase-A probe endpoints remain provisional; the system is not a reproduction of
LeJEPA even though its SIGReg loss implementation passes reference checks.

[ours] Jina-CLIP-v2 is an incomplete replication attempt: constructor resolution was
repaired, but the subsequent full-wrapper CPU FP32 parameter contract failed before
any forward. Original and superseding freezes, failures and diagnostic reports are
preserved. No Jina embedding-quality or compression evidence exists. The exact
parameter-level cause was not persisted and remains unresolved; no further repair
is undertaken in this closure.

## Evidence and scope

The site tables and figure are generated from accepted JSON payloads verified against
`viz/evidence-lock.json`, including each embedded record checksum. They do not load
images, neural models or vectors, and do not recompute scientific endpoints. The
training display continues to consume the corrected Phase-B records exclusively.
The final LaTeX tables share the same artifact reader as the site.

- [Source audit](source-provenance-audit.md)
- [Current sensitivity results](../experiments/phaseC_source_sensitivity/results.md),
  [comparison](../experiments/phaseC_source_sensitivity/comparisons.md),
  [interpretation](../experiments/phaseC_source_sensitivity/interpretation.md)
- [COCO results](../experiments/phaseC_paired_replication/results.md),
  [interpretation](../experiments/phaseC_paired_replication/interpretation.md),
  [source qualification](../experiments/phaseC_paired_replication/source_qualification.md)
- [Jina bounded outcome](../experiments/phaseC_paired_jina_repair/README.md)
- [Research positioning](research-direction-review.md)

[interpretation] The completed portfolio supports a scoped diagnostic study: inspect
instrument definitions and numerical behavior before interpreting a surprising metric,
and evaluate the task property being claimed. It does not establish a universal collapse
detector, a new quality predictor, deployment performance or independent encoder
replication. Experimental expansion stops here.
