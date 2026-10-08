# Inherited source qualification

[ours] Record-only final review found 17 of the frozen 1,000 validation image IDs
begin with `synthetic_val_`; none of the 2,000 training IDs has that prefix.
These IDs already occur in the accepted C1 sample and extraction manifests.
They were not introduced by C3. The recorded local enriched source is Lance
version 55, 80,500 rows, attributed to `lance-format/BDD100K-enriched`.
The exact affected IDs are recorded in `completion_audit.json` and `exceptions.json`.

[interpretation] An identifier prefix is not evidence establishing how image
bytes or labels were produced. Repository records inspected here do not explain
these rows' origin. No image decoding, source download, re-extraction, row
exclusion, new subgroup endpoint or sample change was performed to resolve it.
Accepted artifacts and all frozen C3 results remain unchanged.

[interpretation] The fixed-cache numerical comparisons remain comparisons on
that exact cache, but the unresolved source qualification limits claims about
an exclusively original, naturally sampled BDD validation population. Refer to
BDD100K-enriched attributes and this frozen sample rather than asserting verified
original-BDD origin for every row. A future source audit should establish image
and label origins before independent replication or stronger workload claims.
Do not silently filter these rows after observing results or rewrite C1/C2.

## Source audit addendum — 2026-10-08

[ours] The bounded [source-provenance audit](../../docs/source-provenance-audit.md)
now establishes that all 17 flagged main rows contain generated geometric graphics,
by decoding only their corresponding source images. All 3,000 selected raw-image
hashes and metadata match the accepted extraction records; all 750 cache chunks
remain intact. The 17 IDs are preserved verbatim from an upstream 500-row append
in Lance version 52. Its image data file matches the pinned distributed dataset's
LFS SHA256 at revision `d82c5188d392714ba8091d68014f7b9838ceadf2`. No repository
fixture, fallback or ID-rewriting path supplied these rows. Exact generation code
and the attribute-label assignment procedure remain unrecovered. Full per-row
evidence and the separate visual-inspection record are linked from the audit.

[ours] No selected identifiers or exact source-byte hashes overlap between train
and validation, and no selected raw-byte duplicates were found. This does not test
near-duplicates or authenticate all remaining images against the original BDD release.
The 17 rows occur in accepted C1-main, C2 and C3. The exploratory pilot has six
different IDs from the same append; those six were not decoded in this audit.

[interpretation] The source defect is now confirmed for these 17 graphics, while
its numerical effect on accepted results remains unmeasured. Fixed-cache results
remain historical comparisons on that sample; natural driving-scene utility
claims require qualification. The audit recommends a separately frozen exclusion
sensitivity removing these rows as both queries and gallery candidates, using
the existing cache and unchanged training-derived transformations. It has not
been executed. The original disclosure above is preserved as the pre-audit record.
