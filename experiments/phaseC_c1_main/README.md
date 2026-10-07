# C1-main protocol freeze

Exact IDs and class counts are in `sample_manifest.json`; the pre-inference
protocol is [docs/c1-main-protocol.md](../../docs/c1-main-protocol.md).
`freeze.json` binds the sample, protocol, implementation and dependency versions.
The runner requires this freeze to be committed and unchanged before loading
any model. No main embeddings or main endpoints have been generated.

[measurement] Sample: 2000 train / 1000 validation, seed 1, uniformly sampled
without replacement from fully labelled rows, excluding all 512 pilot IDs.
Pilot overlap is zero. Balanced-probe eligibility requires training support
>=20 and validation support >=10. Overall accuracy/fitting/retrieval keep all rows.

| Attribute/class | Train | Validation | Balanced-probe eligible |
|---|---:|---:|---|
| weather: clear | 1172 | 588 | yes |
| weather: foggy | 3 | 1 | no |
| weather: overcast | 309 | 152 | yes |
| weather: partly cloudy | 164 | 81 | yes |
| weather: rainy | 182 | 92 | yes |
| weather: snowy | 170 | 86 | yes |
| scene: city street | 1243 | 577 | yes |
| scene: gas stations | 1 | 4 | no |
| scene: highway | 537 | 254 | yes |
| scene: parking lot | 7 | 5 | no |
| scene: residential | 212 | 156 | yes |
| scene: tunnel | 0 | 4 | no |
| timeofday: dawn/dusk | 146 | 78 | yes |
| timeofday: daytime | 960 | 489 | yes |
| timeofday: night | 894 | 433 | yes |

[interpretation] Only predeclared H1–H4 are confirmatory. The standardized probe
is primary; the unscaled probe is secondary numerical sensitivity. Main's raw
C grid ends at 1e6; warnings/boundary fits remain visible. No Phase-B absolute
health thresholds apply. Training-seed intervals are unavailable for this fixed
encoder; paired effects are descriptive, without manufactured replications.

[open] The initial freeze precedes all main inference. Pilot throughput estimates
approximately 9.3 extraction hours. Execution measurements and main endpoints
will be recorded separately, without modifying this protocol record.
Large arrays will live in the ignored resumable `embedding-cache/` directory.
