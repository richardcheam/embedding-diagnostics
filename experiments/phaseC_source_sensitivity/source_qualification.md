# Current source qualification after the audit

[ours] The [bounded audit](../../docs/source-provenance-audit.md) confirmed generated
geometric graphics in exactly 17 selected validation rows, using their actual
source bytes. The image data file matches the pinned upstream enriched dataset.
The producer/generator and attribute-label assignment procedure remain unknown.
This sensitivity excludes only those 17 explicit audited IDs, from both query
and gallery roles; no prefix-based filtering, resampling or ID generation occurs.
The historical C1/C2/C3 and their source disclosures remain unchanged.

[ours] The retained canonical vectors correspond to 2,000 training rows in source
fragment 0 and 983 validation rows in fragment 1 of Lance version 55. The audit
verified all selected source-image hashes against extraction records, ID uniqueness,
exact-byte uniqueness and absence of train/validation identifier/byte overlap.
The exclusions do not authenticate the remaining images or labels independently
against the original BDD release, or rule out re-encoded/near-duplicate frames.

[interpretation] Report retained enriched-source-fragment utility, not independently
verified original-BDD utility. A numerical or utility finding that persists under
this exclusion remains conditional on that sample and protocol. Point differences
are not equivalence evidence or population-robustness estimates.

[ours] Exploratory pilot IDs `synthetic_val_000015`, `synthetic_val_000052`,
`synthetic_val_000146`, `synthetic_val_000220`, `synthetic_val_000379`,
`synthetic_val_000415` came from the same appended fragment but were not among
the 17 images visually inspected in the audit. Their content/label origins remain
separately unresolved. No pilot evaluation or further image decoding occurs here.
