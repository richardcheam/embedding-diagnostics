# C2 — learned MRL versus centered post-hoc PCA

[measurement] Seven representations computed from the accepted C1 3000×768
FP32 cache; 2000 train / 1000 validation. No new sample, neural inference or
image decoding. This is a predeclared analysis on a previously examined
sample, not an independent new-sample replication.

[measurement] C1 rank-transform audit passed: training-only raw SVD defines
a common projector for both splits. Severe validation output/geometry
matches the historical record exactly. H4 is not superseded; no C1 endpoints
were changed. The severe eight-direction rank-loss reference is separate
from these matched-D PCA baselines.

[measurement] C2 code: `5e3f3a51592187a74f5baccb5f1103a16b83c964`; freeze
SHA256: `9791d04fd60c20336dccdd5ee7db99964d89b170af122718e709ec02fd560d26`. See `freeze.json` and
`docs/c2-protocol.md` for fitting, eligibility, arithmetic and hypotheses.

## Geometry

[measurement] Raw historical metrics and descriptive nominal-D capacity
fractions are retained separately. PCA removes the training mean before
projecting; both compressed mechanisms normalize every row. These are
different constructions, so variance/cosine changes are not solely a
consequence of dimension. RankMe epsilon may marginally exceed its ideal
nominal bound; fractions remain unclipped.

### train

| Representation | D | Variance | Mean cosine | RankMe | PR | RankMe/D | PR/D |
|---|---:|---:|---:|---:|---:|---:|---:|
| native_768 | 768 | 0.260231 | 0.739769 | 291.990 | 39.594 | 0.3802 | 0.0516 |
| mrl_512 | 512 | 0.255261 | 0.744739 | 269.510 | 39.352 | 0.5264 | 0.0769 |
| pca_512 | 512 | 0.998180 | 0.001820 | 369.586 | 35.523 | 0.7218 | 0.0694 |
| mrl_256 | 256 | 0.234169 | 0.765831 | 155.945 | 35.398 | 0.6092 | 0.1383 |
| pca_256 | 256 | 0.998106 | 0.001894 | 214.382 | 32.073 | 0.8374 | 0.1253 |
| mrl_128 | 128 | 0.158380 | 0.841620 | 73.629 | 30.983 | 0.5752 | 0.2421 |
| pca_128 | 128 | 0.998226 | 0.001774 | 113.076 | 25.323 | 0.8834 | 0.1978 |

### val

| Representation | D | Variance | Mean cosine | RankMe | PR | RankMe/D | PR/D |
|---|---:|---:|---:|---:|---:|---:|---:|
| native_768 | 768 | 0.260057 | 0.739943 | 271.900 | 39.361 | 0.3540 | 0.0513 |
| mrl_512 | 512 | 0.255528 | 0.744472 | 252.012 | 39.007 | 0.4922 | 0.0762 |
| pca_512 | 512 | 0.995252 | 0.004748 | 349.317 | 34.567 | 0.6823 | 0.0675 |
| mrl_256 | 256 | 0.234129 | 0.765871 | 150.567 | 35.146 | 0.5882 | 0.1373 |
| pca_256 | 256 | 0.995384 | 0.004616 | 204.963 | 30.016 | 0.8006 | 0.1173 |
| mrl_128 | 128 | 0.156928 | 0.843072 | 72.184 | 31.316 | 0.5639 | 0.2447 |
| pca_128 | 128 | 0.995726 | 0.004274 | 109.659 | 23.395 | 0.8567 | 0.1828 |

## Semantic endpoints

[measurement] BA uses frozen train>=20 AND val>=10 eligibility: five
weather classes, three scene classes, all three timeofday classes.
Every row remains in fitting, overall accuracy and the validation
retrieval gallery. All original support counts are included in results.

| Representation | Weather BA | Scene BA | Time BA | Weather P@10 | Scene P@10 | Time P@10 |
|---|---:|---:|---:|---:|---:|---:|
| native_768 | 0.6123 | 0.6915 | 0.7542 | 0.5722 | 0.6167 | 0.8193 |
| mrl_512 | 0.6098 | 0.6974 | 0.7614 | 0.5713 | 0.6147 | 0.8185 |
| pca_512 | 0.5379 | 0.6619 | 0.7421 | 0.5517 | 0.6104 | 0.8228 |
| mrl_256 | 0.5931 | 0.6901 | 0.7596 | 0.5671 | 0.6144 | 0.8161 |
| pca_256 | 0.5725 | 0.6614 | 0.7449 | 0.5545 | 0.6121 | 0.8254 |
| mrl_128 | 0.5337 | 0.6632 | 0.7321 | 0.5572 | 0.6004 | 0.7988 |
| pca_128 | 0.6068 | 0.6763 | 0.7715 | 0.5560 | 0.6145 | 0.8315 |

| Representation | Weather accuracy | Scene accuracy | Time accuracy |
|---|---:|---:|---:|
| native_768 | 0.7750 | 0.7590 | 0.9200 |
| mrl_512 | 0.7790 | 0.7680 | 0.9250 |
| pca_512 | 0.7290 | 0.7410 | 0.9180 |
| mrl_256 | 0.7710 | 0.7700 | 0.9070 |
| pca_256 | 0.7610 | 0.7470 | 0.9220 |
| mrl_128 | 0.7440 | 0.7540 | 0.9190 |
| pca_128 | 0.7720 | 0.7660 | 0.9190 |

[measurement] Floors (weather/scene/time): majority accuracy
0.588/0.577/0.489; balanced constant-class 0.2/0.333333/0.333333;
retrieval chance 0.391270/0.421838/0.432694. No representation-specific
class exclusion or after-result threshold was applied.

## Paired differences

[measurement] Entries below are percentage-point differences. Full
accuracy/BA/P@10 deltas are saved in `paired_effects.json`.

| Relative to native | Weather BA | Scene BA | Time BA | Weather P@10 | Scene P@10 | Time P@10 |
|---|---:|---:|---:|---:|---:|---:|
| mrl_128 | -7.857 | -2.828 | -2.214 | -1.500 | -1.630 | -2.050 |
| mrl_256 | -1.926 | -0.135 | +0.542 | -0.510 | -0.230 | -0.320 |
| mrl_512 | -0.255 | +0.593 | +0.718 | -0.090 | -0.200 | -0.080 |
| native_768 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| pca_128 | -0.554 | -1.520 | +1.728 | -1.620 | -0.220 | +1.220 |
| pca_256 | -3.980 | -3.003 | -0.932 | -1.770 | -0.460 | +0.610 |
| pca_512 | -7.444 | -2.955 | -1.214 | -2.050 | -0.630 | +0.350 |

| MRL minus PCA | Weather BA | Scene BA | Time BA | Weather P@10 | Scene P@10 | Time P@10 |
|---|---:|---:|---:|---:|---:|---:|
| 128 | -7.304 | -1.308 | -3.942 | +0.120 | -1.410 | -3.270 |
| 256 | +2.054 | +2.868 | +1.474 | +1.260 | +0.230 | -0.930 |
| 512 | +7.189 | +3.548 | +1.932 | +1.960 | +0.430 | -0.430 |

| Difference in accuracy (percentage points) | Weather | Scene | Time |
|---|---:|---:|---:|
| mrl_128 minus native | -3.100 | -0.500 | -0.100 |
| mrl_256 minus native | -0.400 | +1.100 | -1.300 |
| mrl_512 minus native | +0.400 | +0.900 | +0.500 |
| native_768 minus native | +0.000 | +0.000 | +0.000 |
| pca_128 minus native | -0.300 | +0.700 | -0.100 |
| pca_256 minus native | -1.400 | -1.200 | +0.200 |
| pca_512 minus native | -4.600 | -1.800 | -0.200 |
| MRL minus PCA 128 | -2.800 | -1.200 | +0.000 |
| MRL minus PCA 256 | +1.000 | +2.300 | -1.500 |
| MRL minus PCA 512 | +5.000 | +2.700 | +0.700 |

## Fit diagnostics and runtime

[measurement] Standardized probe remains primary. Every inner candidate
and final fit is recorded with C, boundary, convergence, iteration count,
training/validation scores and warnings; final summaries follow.

| Representation | Attribute | C | Boundary | Converged | Iterations | Train accuracy | Train BA |
|---|---|---:|---|---|---:|---:|---:|
| native_768 | weather | 0.01 | lower | True | 286 | 0.9115 | 0.8458 |
| native_768 | scene | 0.01 | lower | True | 581 | 0.8760 | 0.8164 |
| native_768 | timeofday | 0.01 | lower | True | 103 | 0.9745 | 0.9164 |
| mrl_512 | weather | 0.01 | lower | True | 226 | 0.8985 | 0.8214 |
| mrl_512 | scene | 0.01 | lower | True | 345 | 0.8610 | 0.7875 |
| mrl_512 | timeofday | 0.01 | lower | True | 73 | 0.9655 | 0.8733 |
| pca_512 | weather | 0.01 | lower | True | 41 | 0.9300 | 0.8683 |
| pca_512 | scene | 0.01 | lower | True | 57 | 0.8905 | 0.8220 |
| pca_512 | timeofday | 0.01 | lower | True | 20 | 0.9800 | 0.9241 |
| mrl_256 | weather | 0.01 | lower | True | 173 | 0.8470 | 0.7227 |
| mrl_256 | scene | 0.01 | lower | True | 318 | 0.8390 | 0.7452 |
| mrl_256 | timeofday | 0.1 | interior | True | 95 | 0.9740 | 0.9103 |
| pca_256 | weather | 0.01 | lower | True | 34 | 0.8595 | 0.7346 |
| pca_256 | scene | 0.01 | lower | True | 57 | 0.8415 | 0.7352 |
| pca_256 | timeofday | 0.01 | lower | True | 18 | 0.9635 | 0.8642 |
| mrl_128 | weather | 0.01 | lower | True | 96 | 0.7805 | 0.6158 |
| mrl_128 | scene | 0.01 | lower | True | 145 | 0.8155 | 0.6997 |
| mrl_128 | timeofday | 0.01 | lower | True | 39 | 0.9420 | 0.7757 |
| pca_128 | weather | 0.1 | interior | True | 77 | 0.8415 | 0.7328 |
| pca_128 | scene | 0.01 | lower | True | 44 | 0.8130 | 0.6894 |
| pca_128 | timeofday | 0.1 | interior | True | 35 | 0.9600 | 0.8637 |

[measurement] Initial run: 336.93 seconds; PCA fit: 0.36 seconds. All projected vectors finite;
compressed norms approximately one. Native probe continuity against C1:
`{"scene": {"accuracy": 0.0, "balanced_accuracy": 0.0}, "timeofday": {"accuracy": 0.0, "balanced_accuracy": 0.0}, "weather": {"accuracy": 0.0, "balanced_accuracy": 0.0}}`.

[measurement] Fit-record audit: `{"final": {"fit_records": 21, "grid_boundary": {"interior": 3, "lower": 18}, "max_iter_reached": 0, "nonconverged": 0}, "inner_C_selection": {"fit_records": 168, "grid_boundary": {"boundary": 42, "interior": 126}, "max_iter_reached": 19, "nonconverged": 19}}`.
Network/model-construction/image-decode guards passed. Completed resume
performed zero semantic endpoint calls and preserved results/parameters.

[measurement] Tests: 415 passed, 1 skipped; Ruff and lock checks passed.

## Hypotheses and limits

See `interpretation.md` for the qualified C2-H1–H4 assessment.

[open] Paired training-seed intervals are unavailable for one fixed encoder;
no invented seeds, post-hoc uncertainty or significance claims. The same
sample was inspected in C1. Rare categories are outside balanced-probe
eligibility. Coordinate-wise standardization and PCA centering differ
between mechanisms; rank alone cannot separate their causal contributions.
No pass/fail retention tolerance was declared; paired differences remain
descriptive. C3 and C4 have not begun.
