# Independent paired-data evaluation: proposed protocol

2026-10-08. Accepted starting commit: `5f5f991062f117726e5212813d46b1c689b100a8`.
**Preparation only; not an executed experiment or a frozen sample.**
Implementation plan: [image–text plan](image-text-replication-plan.md).

## Recommendation and exact question

[interpretation] Use the **original COCO 2017 validation image archive and
original caption/instance annotations**, with **1,000 distinct image groups**
and five recorded human captions per image. Evaluate the same pinned
EmbeddingGemma 2 at native 768d, learned MRL 256d and learned MRL 128d. Ask:
**Does learned compression preserve recorded image–caption retrieval, and does
its response agree with coarse image-neighbour utility and marginal geometry?**
No PCA, precision, ANN, probe, clustering, additional encoder or severity grid.
This is a proposed independent evaluation sample and cross-modal transfer test,
not an independent encoder replication or a repeat of the BDD experiment.

[ours] The current qualified BDD analysis is the
[983-row exclusion sensitivity](../experiments/phaseC_source_sensitivity/interpretation.md).
MRL 256 has small point losses in coarse attribute P@10; MRL 128 loses more.
Rank/utility comparisons depend on construction and endpoint. These are fixed
enriched-fragment observations, not original-release authentication or universal
compression guarantees. The six pilot appended-fragment IDs remain separately
unresolved; this proposal needs no claim derived from them and does not reopen it.

[interpretation] A useful contribution would be an auditable transfer boundary:
which compression response carries over when relevance becomes recorded human
image–caption correspondence, and which marginal diagnostic fails to identify
that response. A new dataset, familiar Recall@K table, or a pairing permutation
alone supplies no publication novelty. This does not replicate C2's PCA comparison
or C3's precision findings; neither mechanism is included here.

## Local inventory and two-source comparison

[ours] Read-only inspection found `/mnt/hdd/data/datasets/BDD100K-enriched` and
`khmer-speech-dataset`, with no COCO/Flickr/caption-named assets in a bounded
five-level search of `/mnt/hdd/data` and `/home/richard/projects`, or in the
Downloads listing. The configured hub root has model caches, not a discovered
paired evaluation dataset. `/home/richard/.cache/huggingface/hub` is absent.
`/media/root` could not be listed (permission denied). This is an inventory of
those locations, not proof that no paired data exists anywhere on the machine.
No source images or caption archives were downloaded or decoded.

Primary documentation consulted 2026-10-08; only these two candidates compared:

| Candidate | Original pairs and recoverable identity | Release/splits and rights | Decision |
| --- | --- | --- | --- |
| **COCO, original 2017 validation release** | [established] Human caption collection is described in Chen et al., [COCO Captions v2](https://arxiv.org/html/1504.00325), revised 3 April 2015 (technical preprint; data-collection section inspected). | [established] The [official download documentation](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm) lists the 5K/approximately 1 GB validation archive and 241 MB train/validation annotation archive. 2017 changes the image split, rather than introducing a new photograph population. | [interpretation] Recommend: original archive membership, public validation annotations and direct integer IDs make a bounded authentication trail practical. Not the Karpathy split and not a repackaged mirror. |
| **Flickr30k, author-distributed release** | [established] Young et al., [TACL 2014, §3](https://aclanthology.org/Q14-1006.pdf), describes 31,783 photographs with five independently authored captions each; original filenames identify images. | [established] The [author download page](https://hockenmaier.cs.illinois.edu/DenotationGraph/data/) limits provision to noncommercial research/education and retains image-owner rights; its [caption page](https://shannon.cs.illinois.edu/DenotationGraph/data/flickr30k.html) states CC Attribution–ShareAlike for captions. This review has not authenticated an image archive or recovered an original official retrieval split. | [interpretation] Suitable alternative, but adds access/split-provenance work without resolving a distinct question. Do not silently substitute it if COCO verification fails. |

[established] COCO's [annotation schema](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/format-data.htm)
includes image ID, filename, dimensions, license ID and source URLs; captions
have annotation ID, image ID and text. It specifies **at least five captions**,
so the proposed five-caption rule is an explicit analysis subset, not a claim
that every original image has exactly five. Object categories come from the
original instance annotations, not an encoder or generated captioner.

[established] COCO's [terms](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm)
license annotations CC BY 4.0. Image copyright remains with the owners and image
use follows the stated Flickr terms. Record each selected image's license record
and attribution metadata; do not describe all images as CC BY 4.0 or redistribute
them under the repository's MIT license. Flickr30k likewise is not a blanket
CC0 image collection.

## Source verification checklist — required before sample freeze

These are future checks, **not completed authentication**. Dataset authentication
links an original release and its annotation joins to local files; a local SHA256
alone only supplies byte continuity after acquisition.

- [ ] Obtain `val2017.zip` and `annotations_trainval2017.zip` through the publisher
  links in the official download page, using HTTPS at `images.cocodataset.org`.
  Save referring documentation, acquisition time, final URLs/redirects, byte
  sizes, response metadata, local archive SHA256 and ZIP integrity checks.
  If an independent publisher checksum is available, compare it; this review
  has not located one. Do not claim cryptographic origin certification otherwise.
- [ ] Preserve original `captions_val2017.json`, `instances_val2017.json`,
  `info`, `licenses` and archive member names. Record their hashes and the
  official documentation revision/snapshot. Do not mix 2014 annotations,
  Karpathy assignments, enriched rows or generated captions into the release.
- [ ] Check unique numeric image and annotation IDs, every caption/instance
  foreign key, image archive membership, filename/dimensions, nonempty captions,
  license references and physical `val2017` provenance. Missing/conflicting
  source identity stops selection; no guessed ID rewriting or placeholder path.
- [ ] Census caption counts and categories on the release metadata. Inventory
  all 5K validation image bytes, retaining SHA256 separately from identifiers.
  At later authorized source validation, decode to validate dimensions and hash
  contiguous RGB pixels with shape, without EXIF reorientation. This distinguishes
  byte-identical and pixel-identical content. Decoding is not authorized now.
- [ ] Group images by shared original Flickr photo ID where recoverable, equal
  raw-byte hash or equal decoded-pixel hash (transitive closure). Retain aliases
  and group reasons. Unresolved source-photo IDs remain explicit; no guessed
  creator/album groups. Exact matching does not exclude resized/near-duplicate
  photos or correlated images from one event/photographer.
- [ ] Review every selected image and its five captions before model extraction
  for release correspondence and obvious corruption/generated placeholders;
  log findings independently of performance. Do not remove difficult or unusual
  but valid images. A source defect requires a documented protocol amendment and
  new source/sample freeze before endpoints, not discretionary replacements.
- [ ] Save source authentication ledger, duplicate ledger, exact selected IDs,
  caption IDs/text hashes, category support and archive-to-file hash mapping.
  Publish small provenance records; keep source images/large caches ignored.

## Sampling, groups and relevance

Use only original **val2017**. No training split or learned transform is needed.
This is a custom 1K evaluation gallery, not the standard 5K COCO benchmark or a
Karpathy 1K test fold. All captions of an image stay in the same group; splitting
caption rows into separate training/validation units would be invalid here.

After the source census and exact-content grouping, the representative is the
smallest original numeric image ID in each group. Include only representatives
with at least five valid caption annotations and source/annotation joins passing
verification. Report the resulting eligible group population and exclusions.
There are no category quotas or selections based on text richness or embeddings.
If fewer than 1,000 eligible groups exist, stop and revise before inference.

Order groups by SHA256 of UTF-8
`embedding-diagnostics:coco2017:20261009:<representative_image_id>` (decimal ID,
no padding), breaking hash ties by numeric ID; take the first 1,000, without
replacement. This fixed hash order implements deterministic pseudorandom
selection, independent of captions/categories/endpoints. Store output rows in
ascending numeric image ID. Take the five smallest caption annotation IDs for
each representative and retain exact original UTF-8 text, including punctuation.
Record unused caption IDs; do not choose the most descriptive five. Sort captions
by `(image_id, annotation_id)`. Verify all these choices before inference.

| Direction | Queries | Gallery | Recorded positives | Aggregation |
| --- | --- | --- | --- | --- |
| **Text→image, primary** | 5,000 separately encoded human captions | 1,000 representative images | The caption's recorded parent image (one positive) | Mean five caption hits per parent, then mean 1,000 image groups |
| **Image→text, secondary** | Same 1,000 images | All 5,000 document-encoded captions | All five annotations belonging to the query image | Mean over the 1,000 image groups |
| **Image→image, coarse comparator** | Same 1,000 images | Same images, self excluded by image ID | Membership in a shared original COCO object category, scored separately by category | Category-macro P@10 under the fixed rule below |

Do not collapse five captions into a centroid or call them independent samples.
Do not remove cross-modal paired matches as “self”; self exclusion applies only
to image→image. Duplicate caption strings remain distinct annotation IDs with
their original parent links, even across images; disclose within/across-parent
duplicates and exact-score ties. Their possible additional semantic positives
are unjudged, not falsely asserted irrelevant. No union of parents is inferred
from identical strings, and no post-result relevance expansion is allowed.

[interpretation] Recorded-pair retrieval is finer than BDD attribute relevance,
but is still incomplete relevance judgment: another image can satisfy a caption,
and another image's caption can validly describe the query. These endpoints
measure recovery of annotated correspondences, not exhaustive semantic recall.
Do not claim cross-modal Recall@10 and attribute/category P@10 measure identical
semantics. This is a paired-caption task, not class-prototype prompt retrieval.

## Representations, exact scoring and endpoints

Keep `google/embeddinggemma-2` revision
`914f7f89142e33e77833254d9c9b90c3cef7303b`, CPU FP32, batch 1, four threads,
text+vision, audio disabled, eager attention, masked mean pooling including prompt
tokens and L2 normalization. Use the accepted image preprocessing unchanged.
Canonical caches are new and separate: 1,000 images, 5,000 query-role captions
and 5,000 document-role captions, each 768d FP32. Never alter the BDD cache.

[ours] Read-only inspection of the pinned local `config_sentence_transformers.json`
finds `SearchQuery = "task: search result | query: "` and
`Document = "title: none | text: "`. Prepend the former only for text→image queries
and the latter only for the image→text caption gallery. Images have no task text.
Two text-role caches avoid silently treating query instructions as document
instructions. Freeze exact rendered strings, token IDs/lengths, processor hashes
and pooling semantics; fail rather than silently truncate an over-context caption.
The existing adapter exposes images only; text processing still needs model-free
wiring tests and a later authorized contract smoke. No prompt-quality search.

[established] The [official model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2)
describes text task instructions, mean pooling, supported learned prefixes and
mandatory post-slice normalization; it warns of greater multimodal degradation
at 128d. The live card is contextual documentation; pinned local configs and
frozen rendered inputs govern reproducibility.

Exactly three representations: `native_768`, `mrl_256`, `mrl_128`. Derive both
modalities/roles from their canonical vectors by the existing C2 `matryoshka`
function; no repeated model inference for dimensions. Slice matching leading D
coordinates and L2 renormalize. Preserve native cache bytes; for exhaustive cosine
scoring upcast every condition to FP64 and normalize rows in FP64. Record actual
normalization/dot accumulation dtypes, CPU backend and threads. Rank by descending
score, then ascending original numeric image ID or caption annotation ID. Fail
on nonfinite/zero vectors; never silently remove failed examples. Score in blocks
of 64 queries, without ANN or full resident similarity matrices.

**Primary:** text→image Hit/Recall@10 (one recorded positive), with MRL256−native
and MRL128−native paired differences. **Secondary:** Recall@1/5 in that direction;
image→text Hit@1/5/10 (at least one of five positives), and fraction of five
positives recovered at 10. Explicitly name the latter set recall to avoid
confusing it with standard image→text Hit@K. Report absolute values, percentage
point deltas and positive-query gain/loss/unchanged counts. No best-dimensionality
selection, practical-equivalence verdict, post-hoc tolerance or blended score.

For the supporting image-only endpoint, a category is eligible if at least ten
selected images have that original instance category, counting each image once
regardless of object count. Freeze the eligible category list before embeddings.
For each eligible category, queries are its members, the gallery is all other
999 images, and a positive shares that category. Average per-query P@10 within
category, then equally across eligible categories. Images may contribute to
multiple categories; report each support and floor `(n_c-1)/999`, and their macro
mean. Rare categories remain reported but outside the macro; no oversampling,
80-class probe, transfer of the BDD 20/10 rule, or pooled “shares-any-label” score.

Compute existing raw RankMe, centered covariance participation ratio, total
variance and mean cosine **separately** for image, query-caption and document-
caption matrices; also descriptive RankMe/D and participation_ratio/D. Raw rank
has dimension-dependent capacity and text captions are correlated within parents.
Do not pool modalities, transfer Phase-B thresholds, or treat marginal geometry
as a correspondence test. Retain neighbour-set overlap@10 with native as an
identity diagnostic in both cross-modal directions, distinct from positive hits.

Uniform-order reference floors: text→image Hit@K = K/1,000;
image→text Hit@K = `1 - choose(4995,K)/choose(5000,K)`;
image→text positive-set recall@10 = 10/5,000. They follow the fixed IDs/positives,
not BDD class-frequency floors. Report observed ties and gallery denominators.

One native-only instrument control: circularly shift caption **parent relevance
bundles** by one image in ascending ID order, leaving vectors, rankings and
marginal geometry unchanged. Report the resulting paired-hit scores using this
explicitly scrambled relevance map. This checks correspondence dependence,
not a new model condition, realistic negative relevance or a novel theorem.
No collection of permutation seeds to manufacture error bars.

## Uncertainty and possible outcomes

Before extraction, freeze conditional paired percentile-bootstrap intervals:
5,000 resamples, NumPy PCG64 seed `20261010`, resample the 1,000 parent-image units
with replacement. Carry all five captions and both conditions together; bootstrap
saved per-parent scores, **do not duplicate/resample gallery candidates**. Report
95% intervals for the two primary deltas and secondary image→text Hit@10 deltas.
Category macro/geometry/overlap are descriptive, without an invented common CI.
No significance verdict or equivalence declaration; intervals are marginal,
not simultaneous guarantees across endpoints.

[interpretation] These intervals describe query-unit variation **conditional on
this frozen gallery, source eligibility, encoder and prompts**, under approximate
independence of parent images. They do not include gallery sampling, encoder
training uncertainty, caption-author variation, unknown pretraining overlap or
residual scene/photographer dependence. The 1K sample is independent of our BDD
evaluation selection, not proven unseen by EmbeddingGemma pretraining. If source
checks reveal broader recoverable groups, amend grouping/uncertainty before
extraction rather than pretend image independence is established.

| Possible observation | What it would establish within this task | What it would not establish |
| --- | --- | --- |
| MRL256 changes little; 128 loses more paired utility | [interpretation] Conditional transfer of the dimension-specific pattern to authenticated recorded correspondences; report loss sizes/intervals, including null results. | Formal equivalence, independent encoder generality, or a newly discovered compression principle. |
| Both compressed conditions change little | [interpretation] This gallery does not expose a utility boundary down to 128d, despite mechanically reduced rank capacity. | That 128d is universally safe, or that the documented multimodal warning is false. |
| Category utility changes little but paired retrieval falls | [interpretation] Coarse image-neighbour preservation does not assure recorded-pair preservation for these two endpoint definitions. | Equal semantic difficulty, an isolated modality-gap cause, or exhaustive instance relevance. |
| Both utility families deteriorate, including 256 | [interpretation] Utility preservation has measurable costs here; compare magnitudes rather than impose a binary transfer threshold. Report weaker transfer if supported by those effects and uncertainty. | That any nonzero loss refutes transfer, or that task, data, prompts and modality causes have been separated. |
| Direction-dependent loss or a compressed improvement | [interpretation] A boundary depends on retrieval direction/representation; inspect stored per-parent effects descriptively. | An optimal dimension, universal quality ordering, or a reason to expand the grid automatically. |
| Marginal ranks move with utility here | [interpretation] Conditional co-movement on three constructions, an informative null for a dissociation prediction. | Validation of rank as a general utility predictor; three points are not a predictive calibration set. |
| Native recovery is weak or source/adapter checks fail | [interpretation] Either transfer is weak under the declared protocol or the instrument/source is not qualified; report the distinction and stop. | Authorization to tune prompts, replace images or add models until retrieval improves. |

## Resource estimate and stopping point

[ours] Accepted main extraction measured 37,108.043 seconds for 3,000 images,
12.369 seconds/image, peak process RSS 3,510.39 MiB. The exploratory pilot measured
11.145 seconds/image and 3,679.17 MiB RSS, with sampled swap use. These are image
measurements; **no caption throughput has been measured**. Image sizes/token costs
and current memory pressure may differ on COCO.

[interpretation] Plan approximately **3.1–3.45 hours for 1,000 images**, plus
`10,000 × measured seconds/caption` for the two text roles. Illustrative text
rates of 0.1/0.5/1/2 seconds would add 0.28/1.39/2.78/5.56 hours; these are
planning scenarios, not throughput claims or a guaranteed upper bound. A later
contract smoke uses 16 frozen selected images and their captions, retains their
cache entries, measures each role/RSS/swap and projects completion time. A slow
smoke pauses the approved job for resource review; it does not shrink/resample
or choose prompts/dimensions from endpoint performance.

[ours] At preparation, `free -h` reports 7.6 GiB RAM, about 4.1 GiB available,
and 2.9 GiB swap in use; `/mnt/hdd` has about 733 GiB free. Those are transient
system measurements, not the future process budget.
[interpretation] Use one inference process, batch 1, four CPU threads, sequential
image/text phases and chunked resume. Target roughly 3.5–4 GiB process RSS based
on prior observations; text/RGB decode peaks remain unmeasured. Log process RSS,
swap and wall time; stop/checkpoint on allocation failure or sustained swapping,
without substituting examples or changing precision. CPU FP32 remains mandatory.

FP32 canonical payload: `(1000 + 5000 + 5000) × 768 × 4` = 33,792,000 bytes
(32.23 MiB), excluding metadata. Each 64×5,000 FP64 score block is 2.44 MiB.
Compressed arrays are trivially derivable and need not be stored redundantly.
Original val image plus annotation downloads are listed around 1.24 GB; reserve
5 GB for archives/extraction/ledgers/caches pending actual sizes. No training
image archive or dataset-wide encoder extraction is proposed.

[interpretation] Source inspection also requires human time: an illustrative
15–30 seconds per selected image/caption bundle is 4.2–8.3 hours for 1,000 groups,
not a measured rate or a full relevance adjudication. Archive/metadata checks
cannot replace this content sanity review, and this budget is separate from
encoder time. Record incomplete review rather than claim authentication from
checksums alone.

Commit tested implementation/final protocol, then source/sample/input freeze
before inference or endpoints; bind authenticated source ledger, exact ordering,
caption roles/positives, IDs/raw/pixel/text hashes, conditions, ties, support,
bootstrap, code/model/config/dependency/backend identities and protected history.
After canonical extraction, record its hashes in a second binding before analysis.
Resumes reject changed source bytes, roles, revision, dtype, order or protocol.
Deliver three-condition geometry/paired-utility/category tables, deltas/intervals,
overlap, pairing-control audit, runtime and failures. Stop for interpretation.
Any PCA question, finer human relevance judging, independent encoder, precision,
ANN or larger gallery is a separate proposal, never an automatic continuation.
