# Bounded original-tokenizer resolution repair

[ours] Supersedes the blocked `56e9765` attempt without changing its artifacts.
The original failure and freeze remain in `../phaseC_paired_jina_replication/`.
This namespace retains exact accepted COCO input identity and scientific rules;
no independent sample or new encoder is introduced.

[ours] The original constructor failure is reproduced in
`resolution_reproduction.json`. Only its original three nested tokenizer files
were acquired; no nested model weights, dependency changes or new author numeric
patch. `tokenizer_acquisition.json`, `resolution_patch.json` and
`resolution_verification.json` record the resolution repair and independent
nested/outer token checks. `inventory_reconciliation.json` accounts for all999
checkpoint tensors, including the extra scalarlogit_scale element.

[ours] Pre-forward verification:534 root tests passed, one optional skip
(27.68s);16 focused isolated tests passed. Ruff, both lock checks and whitespace
checks passed. No encoder forward or scientific endpoint ran before the
superseding committed freeze. Execution must satisfy the unchanged contract and
resource gates in [the protocol](../../docs/jina-tokenizer-repair-protocol.md).

[ours] Outcome: the tokenizer resolution succeeds, but frozen real loading
fails the all-CPU-FP32 parameter contract before any forward. The bounded attempt
is closed without fallback. Read [interpretation.md](interpretation.md),
[compatibility_blocker.json](compatibility_blocker.json) and
[contract_failure.log](contract_failure.log). No scientific results/cache binding
were produced; the presence of inputs is not completed extraction evidence.

[ours] Final verification:534 tests passed, one optional skip (27.84s);16 focused
isolated tests passed;Ruff/both locks/links/whitespace passed. All frozen identities
and protected history verified. Re-entry guard rejects before model loading;
complete extraction/evaluation resumes are not applicable. No push occurred.
