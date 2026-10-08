# Source exclusion sensitivity implementation plan

> Execute inline under the user-authorized scope. User's protocol and the audit
> recommendation are the design brief; no additional approval stage is required.

## Scope and implementation decisions

Create a separate evaluator module, runner, report generator and synthetic tests.
Do not modify the existing campaign implementations or artifacts. Work in the
clean current checkout; the canonical ignored cache remains at its accepted path.
The protocol lives in `docs/source-sensitivity-protocol.md`.

1. Test then implement exact audited-ID removal, retained order, unchanged train
   bytes, explicit zero-support rows, rectangular-gallery retrieval and floors.
2. Test then implement probe refitting using recorded train-only C selections
   and candidate diagnostics, unchanged existing scaler/final fit/scoring code.
   Models are unavailable; final fits must be reproduced. Record fit continuity.
3. Implement freeze/run wiring around accepted C1/C2/C3 evaluators. Bind all
   historical files, audit IDs, cache bytes, source versions, transformed matrix
   hashes, saved PCA parameters, INT8 scales, scoring/ties and runtime/backend.
   Transform full original splits before excluding rows. Resume checksums and
   condition order must reject mismatches and skip completed endpoints.
4. Implement deterministic tables of absolute changes and effect changes, plus
   retained-query/full-gallery diagnostic. Synthetic tests cover self-exclusion,
   supports, gallery denominators, alignment, preservation and zero-call resume.
5. Run tests/Ruff/lock check. Commit tested implementation and final protocol.
   Prepare freeze without endpoint calls and commit it separately.
6. Execute fixed conditions, generate tables, verify zero-endpoint completed
   resume and historical/cache hashes. Interpret without equivalence claims.
   Update current status and qualification in new files/current docs; preserve
   accepted historical source qualifications. Commit results and stop.

## Ledger

- Planning: accepted cache and saved training PCA present. No saved classifier
  files. Historical candidate/selected-C records are training-fitted parameters
  and will be reused; final models will be refitted with original configuration.
- Supplemental C1/C2 retrieval uses the inherited argpartition implementation's
  arithmetic/order; C3 uses accepted per-query records for old-gallery summaries.
- The audit's 17 IDs are a content-based fixed exclusion, never a name filter.

- Implementation complete: 17 added synthetic tests; 458 passed, one optional
  accelerator test skipped; Ruff and lock validation passed. Fresh review found
  no scientific blocker. Identical-fit provenance rebind and full configuration
  cache identity were tightened before freeze. Scientific endpoints not run.
