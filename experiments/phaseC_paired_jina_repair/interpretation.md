# Bounded repair outcome: tokenizer resolved, parameter contract rejected

[ours] Implementation `4d0b322`; superseding pre-forward freeze `2079de9`,
referencing the preserved original `8557b78` freeze and `56e9765` blocked attempt.
The exact accepted sample, captions and scientific rules are unchanged.

[ours] The original tokenizer failure was reproduced, then resolved by restoring
its three original tokenizer files from the already-pinned v3 source. The
original constructor expression resolves offline with external AutoConfig
lookup forbidden. Independent source binding and all5000 original nested-helper
token inputs were checked; all5000 outer frozen tokens remain unchanged. No
outer-tokenizer substitution or new numeric author patch was introduced.
Executed author/dynamic-file hashes are recorded in the metadata architecture
check. Original four revisions and previous resolution-only patches are retained.

[ours] Complete metadata architecture coverage is999 tensors and865,278,477
parameters:560,895,564 text,304,382,912 vision and the one-element scalar
`logit_scale`. The published865,278,476 count equals the non-scalar sum. Every
checkpoint state key/shape matches; no missing/unexpected/mismatched state and
no discarded scalar. The cause in the Hub counting implementation is unverified.
The advertised checkpoint digest remains verified. This reconciles the inventory;
it does not prove all parameter values were materialized correctly.

[ours] The frozen real load proceeded past the tokenizer blocker but failed
`verify_settings` with `ValueError: all parameters must be CPU float32` before
its first encoder forward. `compatibility_blocker.json` and `contract_failure.log`
record51.199s extraction-stage elapsed time (identity preflight excluded), peak
RSS4295.492MiB, forward calls0. Image decoding, canonical rows, scientific
endpoints and full16-group contract completion are all0.

[interpretation] The traceback places the failure after the strict loader
missing/unexpected/mismatched/error guard, exact state-shape/count check, intended
nested-tokenizer check, actual source-hash check and pooling/processor checks.
These checks executed without raising, but complete adapter provenance was not
returned/saved. The failed predicate means at least one nonempty model parameter
was not CPU float32. The exception did not record offending names/devices/dtypes;
it cannot distinguish a wrong dtype from a non-CPU/unmaterialized parameter.
Do not infer FP16, CUDA use, a corrupt checkpoint or the exact loader mechanism
from this generic message. No second full load was run to investigate it.

[interpretation] This is the second substantial compatibility blocker, so the
bounded attempt is closed under its predeclared rule. The4295.492MiB observed
peak is below the4608MiB RSS ceiling, but no encoder activations or complete
resource/active-swap contract were tested. Output precision, finiteness/norms,
prefix agreement, repeat behavior, modality throughput and scientific compression
responses remain unmeasured. No throughput-based extraction projection or
cross-encoder semantic comparison can be reported.

[ours] No precision, architecture, loading mode, resource limit, prompt, sample,
condition or dependency was changed to bypass the failure. No complete native
binding or scientific result exists. Completed extraction/evaluation zero-work
resumes are not applicable; model-free completed-resume and failed-blocker
regression checks remain. The original failed attempt, frozen COCO evidence,
qualified983-row BDD analysis and six historical unresolved pilot IDs remain.

[interpretation] Retain the completed diagnostic study and report this qualified
encoder-feasibility failure transparently in its synthesis. It neither supports
nor contradicts recurrence of the observed compression responses on Jina. Further
loader/materialization work would require a separately scoped task; it is not
part of this closed attempt. Audio–visual post-training remains a separate project.
