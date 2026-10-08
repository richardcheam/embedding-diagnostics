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
