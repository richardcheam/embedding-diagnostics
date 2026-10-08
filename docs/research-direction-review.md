# Research direction after C2

2026-10-08. Accepted source: `50d03a8d5726465a49ef60d426e652905646cefb`.
Status: literature review and proposed protocol; no C3/C4 implementation or execution.

## Recommendation

[interpretation] Next, ask **when storage rounding changes neighbour identities,
and when those changes actually change attribute retrieval utility**. Use the
existing native 768d cache, local top-10 similarity margins, and separate storage
and arithmetic controls. This is cheaper and more interpretable than adding
another encoder or immediately extending to text. A stable result is useful;
the experiment need not produce degradation to succeed.

[interpretation] The portfolio's strongest contribution is an auditable study
of diagnostic validity: controlled interventions, held-out evaluation, retained
solver failures, and explicit limits on claims. Neither “rank is not quality”
nor “compression can preserve utility” is new. A credible report should connect
those established cautions to a precise mechanism and independent replication,
rather than present a growing metric leaderboard. Publication novelty remains
unestablished by this focused review.

## What the accepted evidence supports

[ours] [C1-main](../experiments/phaseC_c1_main/results.md) uses a frozen,
fully-labelled, uniformly sampled 2,000/1,000 train/validation split. The
3,000×768 FP32 cache comes from pinned EmbeddingGemma 2
`914f7f89142e33e77833254d9c9b90c3cef7303b`, masked mean pooling and L2
normalization. C1's dimensional stress fits an **uncentered train-only SVD
projector** and applies it to both splits; the
[C2 audit](../experiments/phaseC_c2/rank_transform_audit.json) found neither
independent validation fitting nor concatenation leakage.

[ours] C1 contraction preserves standardized probing and cosine retrieval;
noise increases RankMe and participation ratio while P@10 deteriorates.
[C2](../experiments/phaseC_c2/results.md) compares learned MRL prefixes with
centered, train-fitted PCA at 512/256/128, with mandatory renormalization.
MRL 256 changes weather/scene/timeofday P@10 by −0.51/−0.23/−0.32 percentage
points relative to native. PCA 128 improves timeofday P@10 by 1.22 points while
reducing dimensionality. Neither method wins every endpoint.

[interpretation] These are workload-specific paired responses, not evidence for
a universal health score, formal equivalence, or an independent C2 replication.
PCA centering, basis choice, normalization and coordinate-wise probe scaling
are not isolated causal factors. All 21 final C2 probes converge, but 18 select
the lower C boundary and 19 inner candidates fail convergence; small score
differences deserve that qualification. Rare classes remain limited by the
frozen 20/10 support rule. Phase-B absolute variance thresholds do not apply to
these unit-normalized vectors. See the
[C1 protocol](c1-main-protocol.md), [C2 protocol](c2-protocol.md) and
[qualified interpretation](../experiments/phaseC_c2/interpretation.md).

## Closest literature and access record

Primary sources accessed 2026-10-08. “Full text” means methods, relevant
experiments and limitations/discussion were inspected, not an exhaustive
verification of every proof or reproduction of author code. Paper claims below
are [established] prior findings within their stated settings. No required
paper was restricted to its abstract. Venue metadata qualifications matter.

| Source and version/status | Relevant method and finding | Scope/limitation and access |
|---|---|---|
| [RankMe](https://proceedings.mlr.press/v202/garrido23a.html), ICML 2023, peer-reviewed proceedings | Raw singular-value entropy predicts linear-transfer performance sufficiently well for label-free JE-SSL hyperparameter selection in its experiments. | Full conference PDF, §3–5 and appendix. Already notes degenerate full-rank solutions and task/class-count constraints; it does not establish universal utility ordering. |
| [LiDAR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/fb0b57dfb8686c357f96e41b880ce7ed-Abstract-Conference.html), ICLR 2024, peer-reviewed | LDA-style between-instance/within-augmentation covariance separates discriminative variation from nuisance variation before computing a spectral entropy. | Full PDF, §3, §5–6. Reports negative correlations in some VICReg/high-dimensional settings. Requires multiple augmented views per instance: one cached vector per image cannot faithfully supply it. |
| [Comparative study](https://arxiv.org/abs/2608.23182v1), v1, 24 Aug 2026; author metadata reports TMLR 2026 publication | Synthetic spectral sensitivities plus 260 vision models across six downstream datasets; architecture/objective stratification and normalized rank quantities. | Full PDF/HTML, methods and discussion. Regularized least-squares classification differs from our logistic protocol; pool imbalance and vision-only tasks limit transfer. TMLR decision could not be independently inspected: OpenReview forum/API access failed, and indexed review PDF is older. |
| [LayerScope](https://arxiv.org/abs/2609.28086v2), v1 23 Sep, v2 24 Sep 2026, preprint | Layer-wise geometry alongside classification, clustering and cross-modal retrieval. Marginal sliced Wasserstein distance is contrasted with held-out Procrustes alignment and pairing permutations. | Full v2 PDF/HTML, §3–5 and implementation appendix. Model/task/layer-specific evidence, not a universal selector. Its standardized covariance entropy is not this repository's participation ratio. |
| [MMEB-V3 / OmniSET](https://arxiv.org/abs/2604.23321v2), v1 25 Apr, v2 29 Jul 2026; [COLM accepted-paper listing](https://colm.eventhosts.cc/Conferences/2026/AcceptedPapers) | Broad omni-modality evaluation; semantically equivalent tuples diagnose modality effects and instruction-conditioned target constraints. | Full v2, benchmark methods and Appendix A.2. Synthetic video-from-image and audio-from-text introduce asymmetric dependencies; OmniSET diagnostics are separate from leaderboard scoring. |
| [Closing the Modality Gap Aligns Group-Wise Semantics](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ad7e42e7b1f638e991d822724969be45-Abstract-Conference.html), ICLR 2026, peer-reviewed | True-pair alignment plus centroid uniformity targets grouping while preserving instance retrieval. Distinguishes relative rankings from joint cluster geometry. | Full conference PDF, §3–5 and A.5. Imbalance and ambiguous alignment are limitations; observed gap effects are not a theorem that arbitrary shifts preserve cosine rankings. |
| [Convergent Representation of Contrastive VLMs](https://proceedings.mlr.press/v306/yi26b.html), ICML 2026, peer-reviewed | Optimal-representation geometry and shared-subspace alignment; empirical study of 78 contrastive VLMs. | Full proceedings PDF via its linked GitHub asset, §2–5 and A.3. Main theoretical regime assumes asymptotic optimization; several results use von Mises–Fisher marginals. Causes of dimensional loss remain open. Do not assume these conditions hold for our encoder. |
| [Quantization Beyond Uniform Bit Allocation](https://arxiv.org/abs/2608.19388v1), v1 19 Aug 2026; [VecDB program](https://vecdb-ws.github.io/vldb2026/), 4 Sep | Variable allocation for Matryoshka embeddings using scalar/product quantization; tests recall under byte budgets with exact search on reconstructed vectors. | Full v1, algorithms, experiments and §5. Workshop paper, **not VLDB main-track**. Text workloads, expensive allocation search and systems costs constrain conclusions. Already separates quantization error from graph-index approximation. |
| [EmbeddingGemma 2 model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2), official documentation, accessed 8 Oct 2026 | Supported learned 768/512/256/128 outputs, prefix renormalization and a stronger multimodal warning at 128. | Full card consulted; documentation is not peer-reviewed evidence. Live card is not revision-bound; accepted local model-card/config hashes remain the experiment authority. |

[established] Two selective references resolve concrete positioning questions:
[MRL, NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html)
already establishes learned nested representations and efficient classification/
retrieval (conference full-text method and evaluation inspected).
[Anisotropic Vector Quantization, ICML 2020](https://proceedings.mlr.press/v119/guo20h.html)
already motivates score-aware error rather than reconstruction error alone
(proceedings method inspected). Neither a new compression principle nor the
importance of score error can be claimed here.

## Claim-to-prior-art audit

The prior-art column refers to the linked sources above. Proposed contributions
are hypotheses to investigate, not findings already obtained.

| Theme | What prior work establishes | What accepted experiments add | Missing control/limitation | Defensible wording |
|---|---|---|---|---|
| Geometric degeneration versus utility | RankMe qualifications, LiDAR nuisance handling, and the comparative study already challenge unconditional spectral prediction. | C1 explicitly manipulates noise, mean and scale on one frozen external representation while retaining probe/retrieval endpoints. | One encoder/sample; interventions do not identify natural training causes. | [ours] Controlled counterexamples to task-independent metric ordering on BDD attributes; not a novel general principle. |
| Raw/centered rank and scale | Raw singular entropy and centered covariance describe different objects; spectral scale cancellation is mathematical. LayerScope distinguishes preprocessing. | C1 scale/offset controls and C2 document both definitions and capacity fractions. | Centering is confounded with PCA construction; RankMe/D is not novel. | [established] Positive scaling preserves spectral proportions and exact cosine ordering. [ours] Instrument checks reproduce those invariances. |
| Learned MRL versus PCA | MRL already teaches nested compression; model documentation declares supported dimensions. | Matched-D held-out attribute utility differs by method and task, without a universal winner. | No centering-only/basis-only isolation; standardized regularization depends on coordinates. | [ours] Conditional comparison of learned MRL with centered post-hoc PCA, not proof of an intrinsically superior mechanism. |
| Retrieval versus probing/grouping | LayerScope and Closing the Modality Gap already compare endpoint families with different sensitivities. | C1/C2 separate probe BA from attribute P@10 and preserve class denominators. | No clustering result here; coarse labels and imbalance limit semantics. | [ours] Linear decoding and local attribute neighbourhoods respond differently on this workload. |
| Numerical robustness/neighbour order | AVQ treats score error as important; the VecDB paper measures retrieval recall under quantization. | No accepted storage/arithmetic/ANN experiment yet. | Margins, ties, clipping and semantic swap composition have not been measured. | [interpretation] A proposed mechanism audit of identity turnover versus coarse semantic utility; no new quantizer or theorem. |
| Marginals versus paired alignment | LayerScope's permutations already show marginal distance is correspondence-blind; OmniSET and shared-space studies assess complementary alignment/task properties. | C1/C2 contain image-only evidence, despite a multimodal encoder. | No text vectors or caption relevance judgments; shared-space assumptions untested. | [interpretation] Image compression preservation does not establish cross-modal preservation. A pairing control would validate the instrument, not establish novelty. |

## Three candidate directions

| Direction | Question and candidate contribution | Minimum controls and requirements | Limits and useful null outcome |
|---|---|---|---|
| **A. Cache-only numerical robustness — recommended** | Can local similarity margins account for neighbour turnover, and do attribute labels mask that turnover? Closest: AVQ and VecDB quantization. Contribution candidate is a transparent mechanism study joining numerical error, identity overlap and semantic utility. | Native 768d only; exact search; storage-only and arithmetic-only conditions; train-only INT8 calibration; fixed ties/self-exclusion; query-level margins and swaps. Existing cache and small CPU matrix products; no model/data additions. | Tiny gallery and coarse labels cannot establish web-scale ANN reliability. No changes at FP16/INT8 would still document conditional robustness; identity changes without semantic loss reveal endpoint granularity. |
| **B. Cross-modal utility under compression** | Does preserving image-to-image utility preserve text-to-image utility? Closest: MRL, OmniSET, LayerScope and shared-space alignment studies. Candidate is task-specific compression transfer, not discovery of a modality gap. | New pinned FP32 text embeddings and frozen prompts/relevance; same prefix or image-train PCA transform on both modalities; native cross-modal baseline. Existing image cache suffices, but new text inference needs future authorization. Genuine caption tasks also require curated/judged pairs. | Prototype attribute retrieval and paired-caption retrieval answer different questions. No compression loss would support transfer only for the frozen relevance task; disagreement would identify a boundary, not a universal encoder flaw. |
| **C. Retrieval versus semantic grouping** | Do compression mechanisms change grouping and local retrieval differently by attribute? Closest: LayerScope and Closing the Modality Gap. Candidate is an attribute-conditioned mechanism comparison. | Future clustering evaluation on cached representations; freeze distance, K, initialization, train-only centroid fit, held-out assignment, labels/support and ARI/AMI. Reuse committed probe/retrieval results as historical comparators, without rerunning them. | Cluster budget and correlated attributes can drive apparent dissociations. Agreement across endpoints is informative but would add limited novelty to established comparisons. |

For **B**, first distinguish two possible relevance tasks. A small frozen set of
class-prototype queries (“rainy weather”, for example) uses existing attribute
labels to define many relevant images. Fix templates and official text-query
instructions before encoding; report query/class support, chance floors and
prompt specificity. This is class-prototype retrieval, not paired-caption
retrieval. Genuine paired-caption retrieval needs image-specific descriptions,
independent relevance assessment, a fixed candidate pool, and rules for multiple
positives and unjudged images. Attribute triples alone cannot supply that ground
truth. Both are future proposals; no images were inspected for captions here.

Apply MRL prefix-plus-L2 to **both** image and text embeddings. Apply an
image-trained PCA baseline as `(x - image_train_mean) @ image_train_basis.T`,
then L2 normalize, to **both** modalities. Separate modality fits produce
incompatible coordinates; fitting on validation images/text leaks geometry.
Image-trained centering/projection may distort out-of-domain text: this is a
declared control limitation, not proof that MRL is universally preferable.
Caption-pair permutations can preserve marginals while breaking correspondence;
class prototypes cannot substitute for true paired captions in that control.
An image-only gallery also cannot test mixed-modality target preference.

For **C**, predeclare spherical K-means for unit-normalized representations,
train-only centroids and validation assignment. Choose K from the number of
eligible training classes per attribute if exercising known grouping tasks,
explicitly declaring this label-informed K choice. Fix seed, initialization
count, iteration cap, tolerance and empty-cluster policy; these are optimizer
settings, not independent research replications. Score held-out ARI/AMI on the
eligible rows, disclose exclusions, and retain full-gallery retrieval as a
different denominator. Do not choose K/labels or map cluster IDs using validation
accuracy after observing results. Different attribute K values also preclude a
simple absolute comparison across attributes.

## Draft next-experiment protocol: A

**Proposed, not executed or frozen.** Commit a final protocol and implementation
identity before producing new numerical or semantic endpoints. Do not expand
this into a full dimension×precision×ANN grid.

1. **Input and task.** Reuse exact canonical IDs and native 768d FP32 vectors;
   1,000 validation rows are queries and gallery, self excluded, k=10, matching
   C1. The 2,000 training vectors calibrate quantization only. Never read images
   or the Lance-provided embedding column. Cache checksum validation must pass
   before execution; mismatch stops rather than regenerates inference.
2. **Conditions.** Reference: original FP32 storage, upcast to FP64, L2 normalize,
   exhaustive cosine scoring. Storage-only: FP16 cast or symmetric per-coordinate
   INT8, reconstruct in FP64 and renormalize; original queries against compressed
   gallery are primary, with compressed queries+gallery as a matched secondary
   condition. INT8 scales are `max(abs(train[:,j]))/127`; zero axes use scale 1;
   round ties to even and clip to [-127,127]. Record validation clipping rather
   than recalibrating. Separate arithmetic-only control: original FP32 vectors
   normalized/scored in FP32 under a fixed CPU backend. No model precision change
   and no ANN index. Reference FP64 scoring is analysis of FP32 vectors, not new
   FP64 model inference. Renormalization defines cosine after reconstruction;
   it does not restore the lost coordinates.
3. **Ordering controls.** Total order: descending cosine, then image_id;
   exclude self by ID and report ties. Historical retrieval uses argpartition;
   the proposed explicit tie rule may differ at ties. Preserve historical files
   and disclose continuity differences rather than rewriting accepted P@10.
   No silently discarded rows or nonfinite/zero reconstructed vectors.
4. **Endpoints.** Per query: neighbour-set overlap@10, maximum absolute cosine
   error, reconstruction/norm error, top-10 boundary margin, clipping and
   weather/scene/timeofday P@10 differences. Aggregate mean paired effects and
   distributions; report gains, losses and unchanged queries, including
   same-label versus different-label swaps. Include class supports and macro
   retrieval under the existing validation-support≥10 rule, retaining every
   gallery row. Keep C1's sum(p²) chance convention, label it as such, and give
   the exact self-excluded random-neighbour floor separately:
   `sum_c n_c(n_c-1) / (N(N-1))`. No probe refits or new geometry verdicts needed.
5. **Mechanism checks.** Let `gamma_i = s_i,10 - s_i,11` in reference order.
   [established] If every score perturbation is at most `epsilon_i`, then
   `gamma_i > 2*epsilon_i` suffices to preserve the top-10 **set**. This elementary
   order argument is not a new theorem and does not certify within-set ordering.
   For unit vectors, a pre-search bound is
   `epsilon_i <= ||q'_i-q_i|| + max_j ||g'_j-g_j||`, in exact arithmetic.
   Audit numerical residuals separately; do not present floating computations
   as rigorous certificates without accounting for them. Observed maximum score
   error is a retrospective check, not a prospective predictor. Freeze four
   reference-margin quartiles, with ties kept together, before examining
   perturbed endpoints. Report bound coverage and turnover by bin without
   fitting a new health metric or selecting thresholds from outcomes.
6. **Semantic interpretation.** [established] For fixed labels/gallery and k,
   `abs(delta P@k) <= 1 - overlap@k`; identity turnover need not change label
   counts. Distinguish identity stability from semantic stability. A sufficient
   bound that certifies few queries can be valid but practically uninformative.
   Large clipping errors or ties can explain failures of a simple margin
   narrative; retain them rather than changing the quantizer after results.
7. **Provenance and uncertainty.** Bind accepted commit, cache manifest/chunks,
   canonical matrix and sample hashes, proposed code/protocol commit, quantizer
   scales, normalization, tie rule, backend/threading and dependency versions.
   Record wall time/RSS and actual packed bytes including scales; reconstructed
   FP64 arrays do not demonstrate search-memory savings or speedups. Use paired
   descriptive effects for this previously examined sample. Do not treat
   dependent query rows or severity conditions as independent training seeds.
   Future population intervals require a declared independent sampling unit;
   driving-sequence dependence and coarse labels remain limitations.
8. **Stopping point.** Deliver one table joining numerical error, overlap and
   three semantic endpoints, plus margin-bin analysis and exceptions. Synthetic
   tests should verify ties, clipping, no calibration leakage and the order
   bound. Stop for interpretation whether results are stable, unstable or mixed;
   no automatic ANN, extra bit depths, encoder inference or text extension.

## Path to a credible research report

First, finish this narrow instrument study under a frozen protocol, including
null results. Separate mathematical checks from empirical workload claims and
separate storage, arithmetic and search approximation throughout the report.

Second, seek an **independent replication** only with separately authorized
scope: a different frozen cache/task, ideally an existing public embedding
artifact with trustworthy preprocessing provenance. Freeze it before evaluation.
Resolve sequence/group dependence and uncertainty design before treating it as
population evidence. Additional model names alone would not supply independence
or explain a mechanism; a failed transfer is a reportable result.

Third, choose B only if the question becomes whether the mechanism transfers
across modalities. Begin with declared attribute-prototype relevance, then move
to independently judged paired captions if instance alignment is the target.
Any new text inference, dataset/encoder or layer extraction needs a new scope
decision. Do not implement multimodal distribution metrics merely because a
model supports text.

Finally, assemble a report with a claim ledger, immutable protocols, complete
response curves, solver diagnostics and failure cases. Retain the distinction:
Phase A/B studies controlled SSL training systems; Phase C tests diagnostic
responses in a strong external representation. Candidate value beyond a
benchmark is a reproducible explanation of *which instrument changes under
which intervention and why*, with explicit counterexamples and transfer limits.

## Checks performed and unresolved questions

- Checked HEAD and clean working tree before writing; accepted commit matches.
  Read repository instructions, STATUS, development log, open questions,
  Phase-C/C1/C2 protocols, accepted results and interpretation. Inspected the
  train-fitted stress/compression and retrieval implementations.
- Retrieved full texts for all eight required papers; consulted the official
  model card and two selective primary references. Verified COLM listing and
  VecDB program. TMLR publication is author-reported; direct decision access
  failed. This is a focused positioning review, not an exhaustive novelty search
  or reproduction of those papers' numerical claims.
- Canonical cache manifest and 750 declared chunk files are locally present.
  No embedding matrix was loaded or checksummed for this review; full integrity
  validation remains required before a future experiment.
- Only this brief was added. No scientific source, accepted artifact, dependency,
  sample or inference setting changed; no C1/C2 endpoint was repeated. Document
  links and whitespace checked. Tests/Ruff/lock commands were not rerun for this
  documentation-only change; earlier accepted verification remains historical.

[interpretation] Open questions are whether rounding causes enough turnover to
study on this small gallery, whether margin bounds are informative rather than
merely valid, and whether attribute labels conceal meaningful instance changes.
The proposed experiment can answer the first two conditionally; the third needs
finer relevance judgments before becoming a multimodal or instance-level claim.
