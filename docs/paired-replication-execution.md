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
