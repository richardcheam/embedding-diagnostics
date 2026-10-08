# C1-main completion report

[measurement] Frozen protocol commit: `fdc68c1e14004aac45e7c85fa4e05db96eeeda28`.
All 30 intervention/severity records completed. No C2/C3/C4 work began.

## Population, extraction and audit

[measurement] Uniform seed-1 sample: 2000 train / 1000 validation, fully labelled,
without replacement or quotas; zero overlap with the 512-image pilot. Exact IDs
and every class denominator remain in `sample_manifest.json`; the class-count
table is in the pre-inference `README.md`. Eligible balanced-probe classes:
weather clear/overcast/partly cloudy/rainy/snowy; scene city street/highway/
residential; all three timeofday classes. Foggy, gas stations, parking lot and
tunnel are excluded from balanced-probe scoring by the frozen 20/10 rule, while
all rows remain in fitting, overall accuracy and natural-distribution retrieval.

[measurement] CPU FP32, batch 1, four threads, pinned text+vision encoder with
audio disabled. Extraction/loading/cache: 37133.49 seconds (10.315 hours).
Extraction alone: 37108.04 seconds, 12.369 seconds/image, 0.08085 images/second.
Peak process RAM: 3510.39 MiB (3.428 GiB); CUDA was not used.
Cache: 10,351,032 bytes. Shape [3000,768], all finite, norms
[0.9999998212,1.0000001192]; 3000 distinct recorded image-byte SHA256 hashes.
Frozen source/runtime/provenance and exact sample alignment checks passed.
Historical pilot SHA256 checks passed. No recorded protocol deviations.

[measurement] Summed stress endpoint evaluation wall time was 6425.29 seconds
(1.785 hours), excluding pristine baseline and process/setup overhead.

## Pristine endpoints

[measurement] Validation geometry: total variance 0.260056909; mean cosine
0.739943092; RankMe 271.900193; participation ratio 39.361108.
Train geometry: variance 0.260231255; mean cosine 0.739768745;
RankMe 291.990218; participation ratio 39.594032. Full geometry is retained.

[measurement] Standardized probe is primary. Balanced scores use eligible
classes; overall accuracy uses every validation row. Retrieval P@10 and chance
use every validation row. Majority is the validation majority accuracy floor.

| Attribute | Primary accuracy | Balanced accuracy | Raw accuracy | P@10 | Retrieval chance | Majority | Balanced constant-class floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| weather | 0.7750 | 0.6123 | 0.7840 | 0.5722 | 0.391270 | 0.588 | 0.2000 |
| scene | 0.7590 | 0.6915 | 0.7550 | 0.6167 | 0.421838 | 0.577 | 0.3333 |
| timeofday | 0.9200 | 0.7542 | 0.9130 | 0.8193 | 0.432694 | 0.489 | 0.3333 |

## Predeclared paired responses

[measurement] H1: at severity .99, variance ratio is 1e-4. Across the entire
scale sweep, mean cosine, all three P@10 endpoints, and standardized accuracy
and balanced accuracy are unchanged. At .99 the secondary weather fit reaches
5000 iterations and fails convergence, despite unchanged accuracy.

[interpretation] These observations match the predeclared scale invariances;
contracted variance alone does not establish lost linear or retrieval utility.

[measurement] H2: severe mean injection changes mean cosine from 0.739943 to
0.999973; centered variance differs by at most 5.56e-17 across the sweep.
Primary accuracy/balanced accuracy are unchanged. Largest absolute P@10 change
across attributes/severities is 0.0024. Severe secondary weather/scene fits
reach 5000 iterations and do not converge.

[interpretation] Directional concentration and raw RankMe changes coexist with
preserved standardized linear decodability, as predicted; raw solver warnings
require the conditioning qualification established during hardening.

[measurement] H3: at severe noise, participation ratio rises 39.361→434.065
and RankMe 271.900→669.672. Weather/scene/timeofday P@10 fall respectively
0.5722→0.3936, 0.6167→0.4132, 0.8193→0.4451. Paired differences are
-0.1786, -0.2035 and -0.3742; endpoints approach their reported chance floors.

[interpretation] Higher dimensionality accompanies worse semantic neighbourhoods
in this controlled noise intervention, matching the predeclared joint response.

[measurement] H4: eight retained directions at severity .99 yield participation
ratio 4.4274 and RankMe 4.6079. Primary accuracy changes (weather, scene,
timeofday): -0.171, +0.004, -0.005; balanced changes: -0.30235, -0.01096,
-0.05711. P@10 changes: -0.0510, +0.0358, +0.0262.

[interpretation] Geometric dimensionality, linear decodability and retrieval
respond differently. Specific scene/timeofday retrieval increases at this rank
are exploratory observations, not separately confirmatory findings. No severity
was selected as a best endpoint; full paired effects are in `hypothesis_effects.json`.

[measurement] Exploratory mean interpolation preserves primary scores while
contracting variance and increasing mean cosine; its severe secondary weather
fit fails convergence. This transformation is outside confirmatory H1–H4.

## Fits, uncertainty and limitations

[measurement] Saved final-fit records: standardized 93/93 converged; 90 select
a grid boundary (lower). Unscaled: four nonconverged, nine boundary selections.
Inner candidate records include 34 standardized and 80 unscaled nonconvergences.
Counts include the five severity-zero copies of pristine fits; they are not
independent replications. Every candidate/final C, iterations, warnings, scores
and support context is retained in `fit_diagnostics.json` and endpoint files.

[interpretation] Primary convergence does not establish an interior grid optimum.
Lower-bound primary selections and nonconverged candidate fits remain explicit
limitations; no post-result grid change or max_iter increase was made.

[open] Repository paired training-seed Student-t intervals are not estimable for
one fixed pretrained encoder. Results report paired descriptive effects, without
invented seeds, significance verdicts or unregistered bootstrap intervals.
Confirmatory scope is restricted to H1–H4 conditional on this frozen sample and
protocol. Natural training dynamics, general encoder-family uncertainty and
rare-class effects remain untested. No Phase-B absolute health thresholds apply.

[measurement] Final verification: 401 tests passed, one optional accelerator
test skipped; Ruff passed. Large embedding arrays remain gitignored.

[open] C1 is ready for review; proceeding to Matryoshka is a separate decision.
