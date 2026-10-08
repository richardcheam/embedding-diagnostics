# Paired-data compression interpretation

[ours] This bounded analysis uses 1,000 independently sampled original COCO2017
validation image groups and five original human caption annotations per group,
with the same pinned EmbeddingGemma 2 encoder. Implementation `141ebce`,
pre-inference source/sample/input freeze `2eee895`, and completed canonical-cache
binding `b24bf28` precede scientific evaluation. Exactly native 768d, learned MRL
256d and learned MRL 128d were evaluated. See [results.md](results.md) for all
endpoints, class supports, geometry, overlaps and controls, and
[paired_intervals.json](paired_intervals.json) for unrounded intervals.

## Measurements and conditional transfer

[ours] Primary text→image Hit@10 is 97.160% native, 96.960% at 256d and 93.820%
at 128d. Paired native-relative differences are −0.200 percentage points
(conditional 95% interval [−0.540, +0.140]) and −3.340 points
([−4.060, −2.620]). At 256d, 28 caption queries gain and 38 lose their recorded
parent within the first ten; at 128d, 27 gain and 194 lose. These are reported
null and adverse outcomes, not an equivalence test or a selected best dimension.

[ours] Image→text Hit@10 is 98.400%, 98.100% and 95.900%; paired differences
are −0.300 points [−1.100, +0.500] and −2.500 points [−3.600, −1.400]. Recovery
of the full five-caption positive set is more sensitive: set recall@10 is
80.240%, 78.500% and 68.900%, with descriptive differences −1.740 and −11.340
points. The high any-positive hit endpoint does not imply recovery of all captions.
No interval was predeclared for these set-recall differences.

[ours] The 75 eligible category image-only macro P@10 values are 48.842%,
48.125% and 44.880%, against a 3.873% macro chance reference. Native-relative
changes are −0.717 and −3.963 points. All 80 category supports and endpoints
remain available; membership is multi-label and eligibility was frozen at ten
member images. Category-macro uncertainty was not predeclared.

[interpretation] Both utility families show a smaller point loss at 256d and a
larger loss at 128d. Thus the dimension-specific pattern transfers descriptively
from the qualified BDD study to recorded paired-caption retrieval on this source.
This run does not expose a clear preservation-versus-loss reversal between coarse
category utility and paired Hit@10. Their magnitudes cannot be compared as if
they measured identical semantics. Small 256d differences and intervals including
zero do not establish equivalence; the gallery is comparatively easy for Hit@10.
The larger set-recall loss also illustrates the dependence on endpoint definition.

## Identity and geometry

[ours] Native-neighbour overlap@10 is 0.83630/0.83750 (text→image/image→text)
at 256d and 0.64778/0.65240 at 128d. Neighbour sets change for 4,410/879 queries
at 256d and 4,939/987 at 128d, despite much smaller changes in positive-hit rates.
There are no text→image boundary ties; image→text ties at ranks 1/5/10 are
1/3/4 native, 1/5/9 at 256d and 1/5/1 at 128d. Numeric annotation-ID ordering
resolves them deterministically. Identical captions retain separate parent IDs.

[interpretation] Recorded-positive preservation differs from identity stability.
This does not establish utility of every replacement neighbour: unrecorded
instance-level relevance is unmeasured. Familiar MRL compression and different
endpoint sensitivities are the supported findings; neither a new dataset nor
neighbour turnover alone supplies publication novelty.

[ours] Image RankMe falls from 325.579 to 178.144 to 86.732; centered covariance
participation ratio falls from 90.691 to 72.820 to 56.228. RankMe/D is
0.424/0.696/0.678 and participation_ratio/D is 0.118/0.284/0.439. Image variance
is 0.395358/0.374912/0.250436 and mean cosine 0.604642/0.625088/0.749564.
The report retains separate query-caption and document-caption geometry.

[interpretation] Lower raw rank accompanies worse point utility here, whereas
participation ratio per available dimension increases. Raw and normalized ranks
answer different descriptive questions. This is an informative absence of a
strong rank-versus-utility reversal on these three constructions, not validation
of a general representation-quality predictor. Separate modality marginals do
not measure paired alignment; Phase-B absolute health thresholds were not used.

[ours] The fixed native relevance-bundle circular shift leaves embeddings,
rankings and geometry unchanged, but text→image Hit@10 becomes 1.220% and
image→text Hit@10 0.200%. Their uniform-order references are 1.000% and 0.996%.
Image→text scrambled set recall@10 is 0.040%, versus the 0.200% reference.

[interpretation] Native success depends on recorded correspondence, which
marginal geometry cannot establish. One deterministic relevance permutation is
an instrument control, not a permutation distribution or evidence that every
non-parent item is semantically irrelevant. Its below-floor secondary outcome
is retained without selecting another permutation.

## Resources, source checks and limits

[ours] The 16-group contract passed without retrieval endpoints and retained
16 image plus 80 query plus 80 document embeddings. All three independent first-
input repeats matched exactly. Smoke wall time was 272.823 seconds, peak RSS
3,393.33 MiB, with a conservative projected full extraction time of 4.74 hours.
The subsequent extraction completed the remaining 10,824 canonical entries in
14,507.062 seconds. Smoke plus subsequent extraction totaled 14,779.885 seconds
(4.106 hours), excluding upfront provenance-guard hashing. The image phase
averaged 12.980 seconds/new image; query and document phases averaged 0.17736
and 0.17525 seconds/new caption. Peak process RSS was 3,571.05 MiB (3.49 GiB).
One 63.66-second system-wide swap window reached 287.12 MiB; no two consecutive
windows crossed the frozen 256 MiB limit. The gate passed without condition changes.

[ours] Original-source structural verification took 57.328 seconds. The recorded
1,312.352-second contact-sheet inspection window includes interleaved development
and review; it is not dedicated inspection CPU time and is separate from encoder
extraction. The recorded condition-evaluation/interval phase took 11.348 seconds,
excluding upfront provenance checks, cache loading and native-reference scoring.

[interpretation] Original-release archive/annotation joins, source IDs, rights
records and all selected content checks improve authentication over the enriched
BDD fragments. They do not prove caption factual accuracy, absence of pretraining
overlap, or absence of resized/near/event duplicates. See the explicit
[source qualification](source_qualification.md). Evaluation sampling is independent
of BDD selection, but this is the same encoder, not independent encoder replication.

[interpretation] Parent-group intervals are conditional on the fixed gallery,
prompts and encoder; they omit gallery sampling and training uncertainty, and
assume approximate parent independence. Directions differ in caption-role prefix,
candidate count and number of positives, so directional differences cannot isolate
modality asymmetry. Relevance is recorded parent correspondence, not exhaustive
semantic relevance. No claims depend on reopening the exploratory pilot; its six
unresolved IDs and the current 983-row BDD qualification remain unchanged.

## Stop and next scoped question

[interpretation] The bounded transfer question has a usable answer: category and
paired-hit responses broadly agree, 256d has small observed losses with unresolved
equivalence, and 128d loses more, especially when recovering multiple captions.
No prompt, sample, dimension, scoring rule or uncertainty method was tuned after
results. There was one pre-inference transport amendment and explicitly qualified
assistant content inspection; no scientific-grid deviation occurred.

[interpretation] Stop here. The next useful separately authorized replication
would hold this paired-data protocol fixed and test one independently chosen
multimodal encoder with documented learned compression. That would address the
remaining encoder-specificity question rather than repeat this model on more
images. It requires a new model/source license and resource review, model-specific
prefix/pooling validation and a pre-execution freeze; nothing is started here.
If that is impractical, report the current evidence as a scoped diagnostic study
with familiar compression behavior, rather than promise a general contribution.
