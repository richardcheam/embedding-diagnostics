# C2 cache-only implementation plan

Spec: [c2-protocol.md](c2-protocol.md), following the user's detailed C2 request.
Native execution in the current repository; no changes to accepted C1 sources.

- [x] Inspect accepted C1 source at 0c19a85, audit raw training SVD/shared validation
  projector, compare severe cached geometry, and record immutable audit.
- [x] Test and implement official MRL normalization, centered training-only full
  PCA, matching dimensions, zero/nonfinite failure, and descriptive capacity fractions.
- [x] Reuse C1 standardized probe/scoring, preserving every fit diagnostic and
  frozen support. Test split alignment, zero projection and source/hash rejection.
- [ ] Review implementation and protocol, run pytest/Ruff/lock checks, commit
  implementation. Create/commit source-cache/sample/runtime/code freeze.
- [ ] Execute only the seven representations from cached FP32 vectors; checkpoint
  endpoints with compatible freeze/parameter provenance. Inspect all fit warnings.
- [ ] Generate raw/capacity geometry, semantic/native/matched effects, qualified
  H1–H4 report and verification from machine-readable results. Preserve C1/pilot.
- [ ] Verify and commit results; STOP before C3/C4.

Failure cases covered: zero projected norm fails without row deletion; invalid
MRL/PCA dimensions fail; changing validation cannot refit PCA; uncommitted or
changed freezes fail before endpoints; mismatched canonical IDs/hash fail.
