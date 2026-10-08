# Bounded C3 record

[ours] Fixed 2,000/1,000 train/validation C1 cache, native 768d primary, eight
predeclared exact-search conditions. No inference/images, new sampling,
dependencies, ANN or C4. Accepted C1/C2 records are preserved.

- [Protocol](../../docs/c3-protocol.md); implementation 443459f, freeze 50f6659.
- [Generated comparison tables](results.md), [macro scores](macro_results.md),
  [qualified interpretation](interpretation.md).
- [Inherited source qualification](source_qualification.md): 17 accepted
  validation IDs have unexplained `synthetic_val_` prefixes; no rows were removed.
- `freeze.json` binds cache/sample/source/runtime/conditions/backend/calibration.
- `results.json` contains atomic checksummed records, neighbour IDs and per-query
  errors/margins/attribute counts; it contains no embedding matrix or image bytes.
- `verification.json`, `completion_audit.json`, `exceptions.json` and
  `resume_verification.json` record preservation, exceptions and zero-call resume.
- `pre_resume_hashes.json` records bytes before completed resume.
- `implementation_verification.json` records testing before freeze/endpoints.

Reproduce the report without vector endpoints:

```bash
uv run --locked python scripts/report_c3.py
```

Verify/reuse completed endpoints without recomputing them:

```bash
uv run --locked python scripts/run_c3.py --run
```

The committed freeze must remain unchanged. Source/backend/cache mismatches fail
rather than mix records. Further inference, data, ANN or multimodal work requires
a separate decision. The per-query record is intentionally retained for auditing
semantic turnover; derived embedding arrays are not saved.
