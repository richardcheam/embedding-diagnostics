# Paired-replication execution ledger

Plan: docs/image-text-replication-plan.md. Source commit: 96a28b9.
Execution authorized by the user; exactly 1K image groups, five captions each,
three representations. Historical BDD artifacts remain read-only.

- Setup: clean accepted main checkout; created branch `phaseC-paired-replication`
  in place, retaining access to ignored historical caches. No worktree/dependency
  installation required; branch isolation preserves the accepted main ref.
- Preflight: source/sample identities feed role-specific cache identities, which
  feed rectangular relevance and grouped scores; freeze gates all inference.
- Ruling: publisher HTTPS hostname certificate fails; verified DNS and successful
  TLS on the same S3 bucket's path-style endpoint permit original-source acquisition
  there. Record transport deviation; no certificate bypass or mirror. Cost if
  source mapping is wrong: archive origin remains unqualified, preventing inference.
- Task 1: joins/group selection/archive validation implemented; 12 synthetic tests
  passed after observed missing-module/function failures. Downloaded original
  archives, validated all 5K image CRC/decodes and all caption/instance joins;
  source groups=5K, selected draft=1K, 25,014 source caption annotations.
- Task 2: text roles/cache implemented; role-prefix pooling tests observed RED then
  GREEN; 40 focused tests passed with evaluator/workflow tests. No model inference.
- Task 3: exact paired evaluator and conditional parent resampling implemented.
  Completed-resume early return added after RED regression; retrieval must not
  reconstruct reference endpoints on a completed run. Further gates/review pending.
- Task 4: source inspection, tests, implementation commit and freeze pending.
- Task 5: smoke, extraction, cache binding, evaluation and interpretation pending.

- Ruling: content inspection is assistant visual contact-sheet review, not an
  independent human relevance judgment. Original-source illustrations and factual
  caption errors are retained and disclosed. Cost if missed: thumbnail review
  does not establish exhaustive caption correctness or all possible image defects.
- Operational resource gate finalized before smoke: max RSS 4,608 MiB; two
  consecutive >=60-second windows with >=256 MiB system swap I/O checkpoint.
  Historical swap occupancy alone is not active pressure. Cost if conservative:
  unrelated system pressure can pause an otherwise feasible fixed extraction.

[ours] Pre-freeze final review found and fixed four implementation gaps: interval
finalization can recover from stored parent scores after the last condition append;
contract repeats persist atomically per role; cache binding validates the same
inference provenance as the writer; native reference scoring uses the declared
four-thread context. Added regression tests, including actual zero-model completed
extraction. Tie counts now cover 1/5/10 boundaries. Freeze reconstructs the sample
from original JSON joins and grouping rather than trusting the draft, and binds
the accepted canonical BDD manifest to its C2 frozen hash.

[ours] Selected-source contact-sheet review is complete for all 1,000 groups.
This is assistant thumbnail/caption inspection, not independent human relevance
adjudication. Original illustrations, collages and edited photos are retained;
caption factual errors are preserved. The separate inspection ledger records
coverage, observations and its interleaved wall-time window. No inference or
scientific endpoint has occurred before the forthcoming freeze.

[ours] Pre-inference implementation verification: `uv run pytest` — 497 passed,
one optional skip (31.40 s); `uv run ruff check .` and `uv lock --check` passed.
The four important fresh-review findings were repaired and covered by tests.
No accepted scientific artifact or dependency was changed.

[ours] Tasks 4–5 completed: final implementation/protocol `141ebce`, independent
original-source/sample/rendered-role-input freeze `2eee895`, passed offline
16-group contract `7cd2c17`, complete canonical binding `b24bf28` before endpoints.
All 1K image and both 5K caption-role matrices completed without substitution.
Exactly three representations evaluated; conditional intervals and pairing control
completed. Source transport amendment and assistant-inspection limitations are
explicit. No sample/prompt/grid/threshold was changed after outcomes.

[ours] Primary native/256/128 text→image Hit@10: 97.160/96.960/93.820%.
Category macro P@10: 48.842/48.125/44.880%; image→text set recall@10:
80.240/78.500/68.900%. Interpretation reports null 256d intervals and the larger
128d losses without equivalence or novelty verdicts. Completed extraction and
evaluation resume do zero model/endpoint/ranking/bootstrap work. All 1,260
protected files and 11,003 bound canonical files are checked unchanged.

[interpretation] Stop at the recorded interpretation. Independent encoder
replication is a separately scoped recommendation; no additional model,
precision, ANN, prompt or sample work is authorized by this completion.

[ours] Final verification: 497 tests passed, one optional skip (34.31 s);
Ruff, lock check, whitespace check and local document links passed. Accepted
C1/C2/C3/pilot/sensitivity artifact diffs from `96a28b9` are empty. Completed
resume preserves all 18 pre-resume artifacts and verifies all protected hashes.
