# Second-encoder paired-data replication proposal

2026-10-09. Accepted source: `82d7086` on `phaseC-paired-replication`.
**Preparation only: no checkpoint acquisition, dependency installation, model
execution or scientific endpoint computation is authorized by this document.**
Implementation plan: [second-encoder-replication-plan.md](second-encoder-replication-plan.md).
Read-only evidence ledger: [second-encoder-source-review.json](second-encoder-source-review.json).

## Recommendation and question

[interpretation] Recommend **`jinaai/jina-clip-v2` only**, conditional on the
compatibility and resource contract below. Ask: **Within a second encoder, do
learned prefixes affect first-rank recovery, any-positive retrieval, five-caption
coverage and neighbour identity differently, and does the response resemble the
accepted EmbeddingGemma experiment?** Preserve the exact COCO sample and evaluation
rules. This tests recurrence across two encoder implementations; it cannot isolate
architecture, training data, loss, resolution or task instructions as causes.
Different released encoders also do not establish disjoint pretraining corpora;
shared web data and model-selection exposure remain unknown.

[ours] The accepted [COCO study](../experiments/phaseC_paired_replication/results.md)
has small 256d Hit@10 point losses without equivalence evidence, larger 128d
losses, and greater image→text set-recall sensitivity. It has substantial neighbour
turnover despite smaller positive-hit changes. Category and paired-hit responses
broadly agree. These qualified observations motivate endpoint comparison, not a
requirement that Jina reproduce the same ordering. No historical endpoint or
qualification changes here.

[interpretation] Jina is **not executable in the current locked environment as-is**.
The inspected wrapper imports a removed Transformers symbol. An isolated,
separately locked extraction environment is the proposed remedy, not a downgrade
of the accepted project. CPU attention/FP32 paths are present in author source;
full-stack correctness, memory and speed remain unmeasured. No alternative was
assessed: there is no demonstrated intrinsic inability to perform the proposed
FP32 task, and the two identified feasibility gates are concrete. If those gates
fail, report a blocker before choosing another encoder or changing conditions.

## What the closest publication already establishes

[established] Koukounas et al., [arXiv:2412.08802v2](https://arxiv.org/html/2412.08802v2)
(11 December 2024 initial submission; 24 April 2025 revision), is reviewed here
as a **preprint**, not verified peer-reviewed proceedings. Methods §3–4 and the
compression appendix were read, beyond the abstract. §3.4 applies prefix losses
to both towers. Table 23 already measures COCO/Flickr30K cross-modal compression.
Its COCO Recall@5 is:

| Direction | 1024d | 256d | 128d |
| --- | ---: | ---: | ---: |
| text→image | 68.35% | 67.43% | 64.58% |
| image→text | 81.46% | 80.70% | 78.66% |

[interpretation] Small 256d losses and larger 128d losses are therefore established
model-specific prior observations, not a prospective novelty claim. Our frozen
1K gallery differs from the publication's evaluation; its percentages are not
expected reference targets. The inspected compression tables do not jointly
report five-caption coverage, native-neighbour overlap and coarse category utility.
The incremental question is their **joint response within the same fixed workload**,
including null agreement. Publication novelty is unestablished. Unknown pretraining
and repeated public-benchmark use prevent calling this an unseen-data test.

## Immutable candidate identity and licence

[ours] Hub metadata and source text were retrieved read-only on 2026-10-09.
All revisions below are full commit IDs, not mutable `main` refs. The checkpoint's
`auto_map` points outside its repository; pinning weights alone is insufficient.

| Component | Proposed immutable revision | Role |
| --- | --- | --- |
| `jinaai/jina-clip-v2` | `e10d47f5691d0454a0fb5d13f46f2199b74cb436` | Weights, tokenizer, image processor settings and CLIP overrides |
| `jinaai/jina-clip-implementation` | `39e6a55ae971b59bea6e44675d237c99762e7ee2` | Wrapper, towers, image processing and transforms |
| `jinaai/jina-embeddings-v3` | `ab036b023d30b4d1138c4c3bfa9f0c445ab455d6` | Nested text architecture **configuration only**, overridden by CLIP configuration |
| `jinaai/xlm-roberta-flash-implementation` | `bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3` | Nested text/LoRA, attention, rotary and pooling implementation |

[ours] Hub reports 865,278,476 stored F16 parameters. `model.safetensors` is
1,730,688,642 bytes with advertised LFS SHA256
`eff4c0a13ab4de71a9927a56968fef44e626920ff935e503f1bd3e6ec797062d`.
This is **publisher-hosted metadata**, not a locally verified weight checksum;
no weights were fetched. Future FP32 inference upcasts the released F16 weights;
it does not recover an earlier training-weight precision. Prefer that file, not
redundant `.bin` or ONNX exports.
At future acquisition compare the actual file digest, tensor inventory and
configuration with this ledger.

[established] The pinned [model card](https://huggingface.co/jinaai/jina-clip-v2/blob/e10d47f5691d0454a0fb5d13f46f2199b74cb436/README.md)
licenses local model use under **CC BY-NC 4.0**, rather than the repository's MIT
licence. The three nested Hub cards also declare CC BY-NC 4.0; inspect retained
upstream notices in adapted code before redistribution. Proposed use is
noncommercial research with attribution and no weight redistribution. Commercial
reuse is outside that declared grant. Original COCO image/annotation rights and
accepted source-inspection limitations remain separate.

## Verified input and output contract

[ours] The following is source inspection, not an executed numerical validation.
References are to immutable author files, with line ranges in the evidence ledger.

| Aspect | Inspected behavior | Proposed fixed contract |
| --- | --- | --- |
| Learned dimensions | [config.json](https://huggingface.co/jinaai/jina-clip-v2/blob/e10d47f5691d0454a0fb5d13f46f2199b74cb436/config.json) lists 32/64/128/256/512/768/1024; card documents both modalities down to 64. | Only native 1024d, MRL 256d and MRL 128d; no 32d experiment. |
| CPU and precision | `configuration_clip.py:288–303` selects FP32 when text FlashAttention is off or CUDA unavailable; `_resolve_attention_libs` disables CUDA-only libraries. | Explicit CPU `torch.float32`, batch size 1/four threads, eval/inference mode, autocast off. Verify actual parameter, activation and output dtypes. |
| Attention fallback | Wrapper `modeling_clip.py:122–192`, nested `mha.py:SelfAttention` and `rotary.py` provide native Torch paths; EVA `eva_model.py:288–339` has ordinary matmul/softmax attention. | Explicit `use_text_flash_attn=False`, `use_vision_xformers=False`; resolved nested `use_flash_attn=False` and `vision_config.x_attention=False`. No FlashAttention/xFormers/apex installation. |
| Image input | Processor/transform: RGB, bicubic shorter-edge resize to512, center crop512, ToTensor, CLIP mean/std. | Decode the same original bytes; use this author processor unchanged, not EmbeddingGemma's preprocessing. Freeze processor and transform hashes. |
| Pooling/projection | Text `HFTextEncoder.MeanPooler` masks pad-ID tokens. Vision uses normalized CLS output. CLIP outer projections and EVA head are identity at this checkpoint. | Call official feature/encode paths; retain special tokens and documented pad mask. Never substitute masked mean for image CLS. |
| Text roles | `encode_text(task=None)` adds no instruction; config default instruction is null. Default trained LoRA remains `retrieval.query`, independently of the instruction argument. | **`task=None` in both directions**, original caption strings unchanged, no prefix. Keep the checkpoint's one trained default LoRA active. No EmbeddingGemma prefix or invented `retrieval.passage` adapter. |
| Context | Tokenizer config maximum8194 includes specials; wrapper otherwise defaults to max512 with truncation. | Preflight all 5K token sequences, retain specials; explicit `max_length=8194, truncation=False`. Fail on excess context rather than silently shorten. |
| Normalization | Official `encode_image` and `encode_text` slice before `f.normalize`. | Cache finite unit FP32 native 1024d once; derive leading256/128 and mandatory FP64 L2 renormalization, then accepted FP64 scoring. Check agreement with direct official prefix encoding within frozen FP32 tolerance. |
| Ordering | Public encode methods sort inputs internally and invert their permutation. | Batch1 canonical order, numeric source IDs; assert output/input alignment and deterministic repeat. |

[interpretation] The no-instruction cross-modal path is a documented API choice,
not a claim that it is uniquely optimal. The model card also illustrates an
instruction-bearing search query, but its query-versus-caption example includes
text-only retrieval. This proposal fixes the plain-caption API for both modalities
before inference; no task/prompt comparison or outcome-based choice is permitted.
An instruction `task=None` does **not** disable the default trained LoRA.

[ours] Relevant code: [wrapper](https://huggingface.co/jinaai/jina-clip-implementation/blob/39e6a55ae971b59bea6e44675d237c99762e7ee2/modeling_clip.py),
[text tower/pooling](https://huggingface.co/jinaai/jina-clip-implementation/blob/39e6a55ae971b59bea6e44675d237c99762e7ee2/hf_model.py),
[image processor](https://huggingface.co/jinaai/jina-clip-implementation/blob/39e6a55ae971b59bea6e44675d237c99762e7ee2/processing_clip.py),
[image transform](https://huggingface.co/jinaai/jina-clip-implementation/blob/39e6a55ae971b59bea6e44675d237c99762e7ee2/transform.py)
and [nested LoRA](https://huggingface.co/jinaai/xlm-roberta-flash-implementation/blob/bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3/modeling_lora.py).
These source files were fetched as text to temporary research storage; none was
imported or executed. Some Hub code pages failed in the browser tool; immutable
`resolve` URLs succeeded and their text SHA256 values are recorded.

## Dependency and loading feasibility

[ours] Current project: Python3.12+, Torch2.11.0+cu128, torchvision0.26.0+cu128,
Transformers5.19.0, huggingface_hub1.33.0, NumPy2.5.1, Pillow12.3.0.
`timm`, `einops`, `sentencepiece`, FlashAttention and xFormers are absent.
The card requires Transformers, Torch, Pillow, `timm` and `einops` for the direct
API. Existing fast `tokenizer.json` permits avoiding SentencePiece provided the
actual local `AutoTokenizer(use_fast=True)` contract passes; SentenceTransformers
is unnecessary. Requests, tqdm and safetensors are transitive requirements.

[ours] There are two concrete 5.x incompatibilities from static inspection:
`modeling_clip.py:29–34` imports `clip_loss`, absent from installed
`transformers/models/clip/modeling_clip.py`; modality configs call
`PretrainedConfig._set_token_in_kwargs`, absent from installed configuration code.
The [official v4.49.0 CLIP source](https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/clip/modeling_clip.py)
and [configuration source](https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/configuration_utils.py)
retain these symbols. That checks these blockers, **not every runtime interaction**.

[interpretation] Proposed extraction-only environment: separately lock
`transformers==4.49.0`, `timm==1.0.15`, `einops==0.8.1`,
`accelerate==1.5.2` for memory-efficient loading, with the exact accepted
Torch/torchvision builds, NumPy and Pillow. PyPI release metadata was inspected:
Transformers4.49 requires hub>=0.26,<1 and tokenizers>=0.21,<0.22, so that environment
needs its own resolved hub/tokenizers pins; the project's 1.33 hub cannot be copied
blindly. No author minimum-version guarantee was found. These proposed versions
are a compatibility starting point, **not a tested lock**. Future work must use
normal `pyproject.toml`/`uv.lock` resolution in the separate extraction project;
no `uv pip install` patching and no accepted-stack downgrade. Evaluate exported
plain matrices with the unchanged accepted project environment.

[ours] The wrapper initializes both towers, but `encode_image`/`encode_text`
call independent feature paths: sequential image and caption phases are supported
without joint cross-attention. This **does not unload the unused tower**. Public
`JinaCLIPTextModel`/`JinaCLIPVisionModel` classes also exist, but subset checkpoint
loading and resolved CPU settings need validation; they are not evidence of a
validated low-memory mode.

[interpretation] Initially use the official complete wrapper with explicit
FP32 and low-memory safetensors loading, sequential modalities and batch size 1. Freeze
that mode before the contract smoke. Do not remove an encoder or replace attention
to fit RAM after seeing endpoints. If full-wrapper loading fails the resource gate,
checkpoint and report; a separately validated official modality-class loading
amendment may be proposed **before** scientific endpoints, with exact state-dict
coverage and parity tests. It is not an automatically authorized fallback here.

[ours] A second provenance issue is independent of RAM: `_build_text_tower`
sets nested revision to `None`; the non-pretrained `HFTextEncoder` path also omits
config revision. Lazy tokenizer/processor loads likewise do not inherit all
nested pins. Future implementation must constrain all four repositories offline,
using reviewed immutable local code/config resolution (or a small recorded patch
that threads revision/code_revision only). Freeze the resolution diff and all
executed files, and test that no mutable `main` or network fetch is possible.
Only CLIP weights are needed; do not download Jina-embeddings-v3 weights.

## Resource estimates, separated from measurements

[ours] At this preparation check, this MSI reports 7,779 MiB RAM, 4,213 MiB
available, 3,215 MiB occupied swap and 732 GiB free on `/mnt/hdd`. This is a
transient observation; occupied swap is not a measurement of active swap pressure.
The accepted encoder measured 12.980 seconds/new COCO image, 0.17736/0.17525
seconds/query/document caption and 3,571.05 MiB peak RSS. **None is a Jina benchmark.**

| Quantity | Estimate or measurement | Practical implication |
| --- | --- | --- |
| Jina all weights FP32 | [ours] Metadata-derived 3,461,113,904 bytes = 3.223 GiB | Lower bound, excluding buffers, runtime, loading duplication and activations. |
| Text / image tower weights | [interpretation] Approximately2.09 /1.13 GiB from published rounded parameter counts | Independent phases with both towers loaded still retain their sum. |
| Vision attention workspace | [interpretation] 512/14 gives36×36 patches plus CLS; one 16-head FP32 score array ≈102.7 MiB | Ordinary attention can need several live arrays; no inference-mode retention of all layers, but loading peaks matter. |
| Full-wrapper RSS | [interpretation] Budget roughly 4.3–5.8 GiB, not a measured bound | Marginal on8GB; the upper estimate fails the inherited4,608 MiB gate. A successful smoke is required, not promised. |
| Extraction time | [interpretation] Scenario10–40 s/image plus 0.15–0.8 s/caption gives approximately 3.0–12.2 h for 1K+5K | No parameter-count-based speed prediction; image resolution/backbone differ. Replace with smoke measurements. |
| Canonical payload | [ours] 6,000×1024×4 =24,576,000 bytes (23.44MiB) | One image and one shared plain-caption matrix; separate logical directional role aliases, no duplicate text inference. |
| Disk | [interpretation] Reserve 10 GiB including 1.73 GB checkpoint, metadata, environment and cache | Do not acquire duplicate `.bin`, ONNX or nested text weights. |

[interpretation] Proposed future 16-group contract:16 images plus 80 plain captions,
independent first-input repeats, official native/prefix parity and dtype/order/norm
checks; **no retrieval or geometry endpoints**. Record load time separately from
image/text throughput and project `1000*t_image + 5000*t_caption`, including actual
startup/cache overhead separately. These time scenarios are not an upper bound;
the smoke may project a longer run. Preserve those entries in canonical caches.
Use the accepted RSS<=4,608 MiB gate and checkpoint on two consecutive>=60 s windows
each>=256 MiB system swap-in+swap-out. Allocation failure also stops. No FP16/BF16,
CUDA experiment, lower image resolution, batching search or model substitution.
If the gate fails, the feasibility result is a blocker, not a reason to tune prompts
or shrink the frozen sample.

## Proposed scientific protocol

[ours] Reuse byte-identical accepted COCO `sample_manifest.json`, source ledgers,
1,000 image groups, five selected caption IDs/texts, parent maps, category memberships
and 75 eligible categories. No sampling call, new inspection filtering or caption
repair. Preserve original duplicates/ties, caption errors, near/event-duplicate
limits and unknown pretraining overlap. The 983-row qualified BDD sensitivity and
six unresolved pilot IDs remain unchanged. This is second-encoder replication on
an **already examined fixed evaluation sample**, not new independent sampling.

| Encoder | Native | Prefix | Retained fraction | Dimension reduction |
| --- | ---: | ---: | ---: | ---: |
| EmbeddingGemma (accepted) |768|256|33.33% (3× smaller)|66.67%|
| EmbeddingGemma (accepted) |768|128|16.67% (6× smaller)|83.33%|
| Jina (proposed) |1024|256|25.00% (4× smaller)|75.00%|
| Jina (proposed) |1024|128|12.50% (8× smaller)|87.50%|

[interpretation] These match absolute compressed dimensions, **not compression
ratios**. Do not add ratio-matched conditions or compare raw native ranks as if
capacity were equal. Main comparisons are each encoder's compressed-minus-native
responses; between-encoder differences remain descriptive and confounded.

[ours] Preserve evaluation unchanged: FP64 row normalization and cosine dot
accumulation, 64-query blocks/four threads; descending score then ascending original
numeric ID. T2I 5K queries→1K images/one parent positive; I2T 1K queries→5K captions/
five positives. Self exclusion only for the image-only comparator. No inferred
positives from duplicate strings. Plain-caption query/document views share the
same Jina cache identity; unlike EmbeddingGemma, no instruction differs by role.
Gallery sizes and positive multiplicities still confound direction comparisons.

[ours] **Primary** remains T2I Hit@10, with MRL 256d−native and MRL 128d−native
paired differences. Report existing Hit@1/5/10 in both directions (Hit@1 is first-
rank recovery), I2T positive-set recall@10, gain/loss/unchanged queries,
native-neighbour set overlap@10, changed identities and ties@1/5/10. Report all
75-category macro P@10 plus all 80 supports/floors, image/query/document marginal
geometry and raw/normalized RankMe/participation ratio; identical Jina text views
are labelled as shared, not independent samples. No new score or practical-loss
threshold. Geometry is descriptive; three dimensionalities cannot validate a
predictor of utility.

[ours] Keep conditional 95% paired percentile intervals: 5,000 PCG64 resamples,
seed 20261010, 1,000 parent units carrying five captions together, fixed galleries,
for the two T2I and two I2T Hit@10 differences. Hit@1/set-coverage/overlap/category/
geometry differences remain descriptive, without post-hoc extra intervals or
cross-encoder causal contrasts. Preserve existing random-order floors and the
one-parent ascending-ID circular relevance shift at native only. No new control
seeds, probe, PCA, precision, ANN, caption adjudication or benchmark split.

[interpretation] Proposed questions before Jina inference: (1) does 256d have
smaller point losses than 128d in positive hits? (2) do first-rank and full-caption
recovery change more visibly than any-positive Hit@10? (3) can neighbour turnover
coexist with much smaller positive-hit changes? (4) do coarse category and paired
responses agree here? These are model-specific recurrence questions. A different
ordering or null response is retained. First-rank/set metrics have different
ceilings and meanings; difference in magnitude does not prove one universal
sensitivity ranking. No confirmatory claim is assigned until the final protocol
and input/runtime identity are separately committed/frozen before extraction.

| Possible outcome | Defensible interpretation |
| --- | --- |
| Familiar256/128 loss pattern recurs across endpoints | [interpretation] Wider encoder coverage of a familiar compression response; no new principle or architecture cause. |
| Hit@10 changes little but first rank/coverage or identities change | [interpretation] Endpoint-dependent preservation recurs on this workload; unchanged hits do not certify complete caption or identity stability. |
| All utility endpoints remain stable in point values | [interpretation] No detected utility boundary under these galleries; not equivalence or universal safety. |
| Endpoint ordering differs across encoders | [interpretation] A transfer boundary; cannot assign its cause to training, architecture or prompts. |
| Category and paired responses diverge | [interpretation] Coarse grouping does not assure recorded-pair preservation here; not identical semantics or an isolated modality-gap explanation. |
| Geometry tracks utility at three dimensions | [interpretation] Descriptive agreement, not a general predictor validation. |
| Source/adapter/resource gate fails | [interpretation] Qualified feasibility failure; no scientific response inferred from an incomplete run. |

## Future gates and preparation checks

[interpretation] After separate execution authorization: test implementation and
final protocol; commit them; bind accepted/source/sample hashes plus Jina rendered
inputs, tokenizer IDs, every nested revision/code hash, loading mode, isolated lock,
actual dtype/backends, normalization, ties and interval rules; **commit freeze
before inference**. Run the no-endpoint 16-group contract. If feasible, resume
canonical native extraction, then **commit cache hashes before evaluation**.
Run exactly three conditions, report nulls/deviations, verify completed zero-work
resume and protected historical hashes, commit interpretation and stop.

[ours] Preparation performed: clean HEAD/branch check; repository AGENTS and
accepted proposal/plan, results, interpretation, qualification, verification and
research-direction review read; cache/evaluator/adapter abstractions inspected;
all 1,260 protected prior files and 11,003 accepted paired-cache files hash-verified
without decoding or loading matrices; published full-text methods/compression
results and immutable model/code/config text read; nested revision graph, dependency
metadata, installed version inventory and missing-symbol source comparison checked.
No model weights, tokenizer execution, image decode, inference, endpoint computation,
dependency installation or scientific artifact modification occurred. Version/
resource estimates are qualified above. Required pre-commit checks are recorded
in the read-only evidence ledger after completion.

[ours] Preparation verification: 497 tests passed, one optional skip (34.17 s);
Ruff and `uv lock --check` passed. Local document links resolve. All protected
historical files and paired cache hashes were checked again after preparation;
all 24 accepted paired ledgers/results are byte-identical. Only these proposal,
plan and read-only evidence-ledger documents are changed.
