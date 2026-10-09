# Pinned Jina paired replication attempt

[ours] Source correction `b4c2c97`; tested implementation/final protocol
`35bdf75`; committed pre-forward input/runtime freeze `8557b78`.
The offline contract stopped in nested tokenizer construction before any encoder
forward. Scientific endpoints are **not run**; no cache binding or scientific
result is implied by the presence of input manifests.

[ours] Read [interpretation.md](interpretation.md),
[compatibility_blocker.json](compatibility_blocker.json) and
[contract_failure.log](contract_failure.log) for the measured blocker.
The acquisition ledger records four pinned repositories and the verified
advertised checkpoint digest. The input manifest preserves the accepted1000
image/five-caption sample using one shared plain-caption identity. Original
snapshots and all accepted historical artifacts remain unchanged.
