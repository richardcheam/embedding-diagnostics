# Post-audit source exclusion sensitivity

[ours] Implementation/protocol: `2feef78`; pre-execution sample/freeze: `97595f3`.
This separate record excludes exactly the audit's 17 content-confirmed graphics
from both validation queries and galleries. All 2,000 training vectors are
unchanged; 983 validation vectors retain their original order. Original cache
and every accepted historical experiment artifact are protected by hashes.

- [Protocol](../../docs/source-sensitivity-protocol.md), `freeze.json` and
  `sample_manifest.json` bind exclusions, inputs, conditions and evaluation rules.
- [Tables](results.md), [absolute/effect comparisons](comparisons.md) and
  [machine-readable comparisons](comparisons.json).
- [Probe diagnostics](fit_diagnostics.md), [gallery diagnostic](gallery_diagnostic.md)
  and [C3 margin/error analysis](margin_analysis.md).
- [Interpretation](interpretation.md), [current source qualification](source_qualification.md).
- `results.json` contains atomic checksummed condition records. `verification.json`
  and `resume_verification.json` record execution and zero-endpoint completed resume.
- `implementation_verification.json` records pre-freeze tests and fresh code review.

[ours] C1/C2 reuse historical train-only regularization selections and diagnostics;
final probes/scalers are refitted under their original settings because no saved
classifier models are available. C2 uses saved, hash-verified training PCA
parameters; INT8 calibration is unchanged and training-only. Full original split
transformations precede validation exclusion, preserving seeded noise assignment.

[interpretation] Retained rows have verified enriched-source fragment provenance
(train fragment 0, validation fragment 1). They are not independently authenticated
against the original BDD release. Labels, near-duplicates and finer instance
relevance remain unvalidated. The exploratory pilot's six different IDs from the
appended fragment remain separately unresolved; this task does not rerun it.
No inference, image decoding, sampling, dependencies, ANN, C4 or replication.

Re-use complete records, requiring unchanged committed freeze and historical files:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  uv run --locked python scripts/run_source_sensitivity.py --run
uv run --locked python scripts/report_source_sensitivity.py
```

Completed resume does not fit probes or compute vector endpoints. Rendering uses
recorded scalars/neighbours/labels only. This sensitivity is post-audit and is
not a new preregistration or independent confirmatory replication.
