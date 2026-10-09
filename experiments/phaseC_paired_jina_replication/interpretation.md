# Jina replication: blocked offline contract

[ours] The frozen replication stopped during model construction, before its
first encoder forward. This is a compatibility-gate failure, not a semantic or
resource result. No 16-group extraction, prefix/repeat numerical comparison,
canonical binding, retrieval, geometry or cross-encoder endpoint ran.

[ours] Pinned nested source
`jinaai/xlm-roberta-flash-implementation@bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3`,
`modeling_xlm_roberta.py:479–481`, calls
`AutoTokenizer.from_pretrained(self.name_or_path, trust_remote_code=True)`
inside `XLMRobertaModel.__init__`. It does not propagate the pinned code revision.
The local nested config retains its external `auto_map`; Transformers4.49.0
attempts a config-code lookup without that revision and offline mode rejects it.
The reviewed configuration file is present at the pinned revision; the failure
is a resolution path, not evidence that the pinned source was absent or corrupt.
The outer tokenizer was successfully preloaded from the CLIP checkpoint, but
that cannot prevent this constructor-internal lookup.

[ours] `compatibility_blocker.json` and `contract_failure.log` preserve the error
and traceback. Elapsed extraction-stage time before failure was20.623s, peak
process RSS736.785MiB, forward calls0. These measurements exclude identity
preflight. They are not full-wrapper loading peaks or inference throughput.
Actual loaded parameter count/coverage, native activation/output dtypes, numerical
prefix/repeat agreement, complete loading RSS, modality throughput and active
swap windows remain unmeasured. No throughput-based full-extraction estimate is
available. The checkpoint's one-element header/metadata discrepancy remains
recorded; no parameter-count conclusion follows from this partial construction.

[interpretation] The offline guard prevented an unpinned nested resolution from
silently changing the experiment. The declared continuation gate did not pass,
so no encoder compression conclusion can be added. Accepted EmbeddingGemma COCO
and qualified983-row BDD evidence are unchanged. This attempt neither supports
nor contradicts recurrence of their compression-response patterns on Jina.

[interpretation] The smallest future remedy is a reviewed resolution-only change
for this internal tokenizer path: bind its original tokenizer/config/code identity
locally and offline before constructor use, without changing tensors, pooling,
LoRA, image processing, loading mode or numerical precision. Verify that the
internal tokenizer's use in the nested `.encode` helper does not alter the
wrapper's externally tokenized forward path. This requires a new implementation
commit and superseding pre-forward freeze retaining this failed attempt. It is
proposed, not executed. No limits were relaxed or loading mode switched.

[ours] The exact accepted COCO sample, caption text, relevance maps and category
eligibility were retained. Accepted source checks are reused, not claimed as an
independent new authentication/content review. Archive/manifest hashes establish
continuity; unknown pretraining overlap, incomplete pair relevance and residual
near/event duplicates remain. Historical six pilot-ID qualifications remain.

[interpretation] A future synthesis can report the completed diagnostic study
and this qualified second-encoder feasibility blocker. It must not present this
attempt as a completed independent encoder replication. No further experiment,
ANN, prompt tuning, precision condition or multimodal extension is started here.
