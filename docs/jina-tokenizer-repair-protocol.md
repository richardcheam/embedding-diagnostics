# Bounded Jina constructor-tokenizer repair

[ours] Starting `56e9765`, one separately authorized resolution-only repair.
The failed attempt and `8557b78` freeze remain immutable in
`experiments/phaseC_paired_jina_replication/`. The superseding attempt uses
`experiments/phaseC_paired_jina_repair/`, the exact same input manifest, sample,
scientific evaluator/conditions and isolated/root locks. No scientific outcome
has been inspected during repair.

## Reproduction and resolution

[ours] Evaluating the original pinned `XLMRobertaModel.__init__` tokenizer
expression, without constructing/forwarding an encoder, reproduces its offline
failure. The constructor passes only `self.name_or_path` and
`trust_remote_code=True`. The previously selective nested snapshot lacked
`tokenizer.json`, `tokenizer_config.json` and `special_tokens_map.json`, causing
AutoTokenizer to consult external configuration/code without the wrapper's
pinned code revision. The trace is in `resolution_reproduction.json`.

[ours] The three original tokenizer files are now acquired from
`jinaai/jina-embeddings-v3@ab036b023d30b4d1138c4c3bfa9f0c445ab455d6`.
No text-model weights or new dependencies are acquired. Their exact hashes are
in `tokenizer_acquisition.json` and the extended acquisition ledger. The unchanged
nested constructor uses this immutable local snapshot path, already established
by the wrapper's existing pinned configuration resolution. Its tokenizer config
names the installed XLMRoberta tokenizer and contains no remote tokenizer
`auto_map`; it therefore resolves without any AutoConfig/config-code lookup.
The repair restores required source files, **not** a numeric author-code patch.
Prior resolution-only wrapper diffs and all four repository revisions remain.

[ours] Before construction, the adapter explicitly binds the installed
XLMRobertaTokenizerFast to the original nested files offline, rejects incomplete files/classes/remote mappings, then
verifies the actual constructed nested tokenizer's backend, specials, vocabulary,
context and original path against that binding. It never replaces it with the
outer CLIP tokenizer. Real offline expression verification forbids AutoConfig
lookup and compares all5000 captions. The outer frozen token sequences are also
verified unchanged; any equality between tokenizers is measured, not assumed.

[ours] Downstream tracing establishes that the nested tokenizer is used by the
nested `.encode` helper. The CLIP author wrapper independently tokenizes captions
and passes IDs through `HFTextEncoder.forward` to the nested model's forward;
its masked pooling/default trained LoRA remain unchanged. Nested helper token
construction, specials, padding, context/truncation defaults are checked against
its independently bound original source tokenizer; all author source bytes and
helper/forward code remain unchanged. Actual executed-file hashes remain checked
and recorded. Numerical integration is evaluated only after the committed freeze.

## Complete checkpoint inventory

[ours] Metadata-only construction of the full original architecture, using the
same `init_empty_weights()` initialization context as Transformers4.49's
low-memory loader, yields999 state keys and865,278,477 parameters. The safetensors
header has exact matching keys/shapes, with no missing/unexpected/mismatched
state. The sole zero-dimensional parameter is `logit_scale` (one element).
The published865,278,476 count equals the non-scalar total exactly. The Hub
counting implementation's cause is unverified; the complete model includes the
scalar and no tensor is discarded. `inventory_reconciliation.json` records this.
An initial diagnostic with meta buffers enabled failed on vision `linspace.item`;
matching the documented runtime's default CPU buffers fixes that diagnostic.
This did not change the real loading mode or author code.

[ours] Before accepting extraction, the actual loader must independently verify
empty missing/unexpected/mismatched/error lists, exact state-key/shape inventory,
complete parameter count and CPU FP32 parameter settings. Metadata-only coverage
is not a claim that checkpoint values have loaded or memory gates passed.

## Unchanged execution and stop rules

[ours] Commit tested implementation and this final protocol first. Then create
and commit a superseding pre-forward freeze binding the original failed attempt,
its freeze hash, all protected historical artifacts, exact tokenizer/model/code
files, input identity, runtime versions and existing scientific rules. Do not
modify the original freeze or failed-attempt cache manifest. The new cache has
its own freeze identity; no original entries were extracted.

[ours] CPU FP32/batch1/four threads, full official wrapper with low-memory
safetensors loading, official512px processor, CLS image/masked mean text pooling,
identity heads, task=None/plain captions and checkpoint default LoRA remain.
Exactly native1024/MRL256/MRL128; one shared caption cache; mandatory prefix L2
renormalization; existing FP64 scoring, numeric ties, category eligibility,
parent-unit intervals and pairing control. No retuning, prompt/model/sample
changes, ANN, PCA, precision grid or audio–visual post-training.

[ours] Run the original16-image/80-caption contract without scientific endpoints.
Check loader coverage, actual dtypes/settings, finite unit outputs, first-input
repeat and direct256/128 prefix agreement at rtol1e-5/atol1e-6. Loading and native
throughput are separate. Keep smoke entries. RSS<=4608MiB; two consecutive>=60s
swap windows each>=256MiB, allocation failure or another substantial compatibility
failure stop/checkpoint and close this attempt. Do not switch loading mode or
relax the gate. If passed, continue the same1000-image/5000-caption resumable
extraction directly, commit complete cache binding before the unchanged
three-condition evaluation, report qualified results and completed zero-work
resumes, then stop. Unknown pretraining overlap/relevance/near-duplicate limits
and historical BDD/pilot qualifications remain unchanged.
