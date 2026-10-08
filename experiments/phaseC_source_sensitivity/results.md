# Post-audit source exclusion sensitivity

[ours] Exact 17-row exclusion; unchanged 2,000 training vectors, 983 validation
queries/gallery rows. This is a post-audit sensitivity, not independent replication.
All tables are generated from frozen checksummed results. Per-class zero supports
are unscorable; historical balanced-probe eligibility is unchanged.

## Pristine endpoints

| Attribute | Accuracy | Balanced accuracy | P@10 | Macro P@10 | sum(p²) floor | Majority floor |
| --- | --- | --- | --- | --- | --- | --- |
| weather | 0.785351 | 0.630239 | 0.578637 | 0.371181 | 0.398695 | 0.595117 |
| scene | 0.769074 | 0.695197 | 0.624415 | 0.577353 | 0.432172 | 0.585961 |
| timeofday | 0.929807 | 0.764365 | 0.827263 | 0.663795 | 0.434058 | 0.490336 |

## All retained class supports

| Attribute | Class | Train | Historical val | Retained val | Fixed BA eligible | P@10 |
| --- | --- | --- | --- | --- | --- | --- |
| weather | clear | 1172 | 588 | 585 | True | 0.778291 |
| weather | foggy | 3 | 1 | 0 | False | unscorable |
| weather | overcast | 309 | 152 | 150 | True | 0.362000 |
| weather | partly cloudy | 164 | 81 | 81 | True | 0.220988 |
| weather | rainy | 182 | 92 | 87 | True | 0.247126 |
| weather | snowy | 170 | 86 | 80 | True | 0.247500 |
| scene | city street | 1243 | 577 | 576 | True | 0.697743 |
| scene | gas stations | 1 | 4 | 0 | False | unscorable |
| scene | highway | 537 | 254 | 251 | True | 0.554183 |
| scene | parking lot | 7 | 5 | 5 | False | 0.060000 |
| scene | residential | 212 | 156 | 151 | True | 0.480132 |
| scene | tunnel | 0 | 4 | 0 | False | unscorable |
| timeofday | dawn/dusk | 146 | 78 | 75 | True | 0.230667 |
| timeofday | daytime | 960 | 489 | 482 | True | 0.818465 |
| timeofday | night | 894 | 433 | 426 | True | 0.942254 |

## Severe C1 conditions

[ours] Values are reduced-sample condition-minus-pristine effects in percentage
points; every severity and absolute/effect change is in comparisons.md/json.

| Transform (.99) | Attribute | BA effect pp | P@10 effect pp | BA effect change pp | P@10 effect change pp |
| --- | --- | --- | --- | --- | --- |
| scale_contraction | weather | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| scale_contraction | scene | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| scale_contraction | timeofday | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| rank_truncation | weather | -32.2694 | -5.1373 | -2.0345 | -0.0373 |
| rank_truncation | scene | -0.7090 | +3.7335 | +0.3873 | +0.1535 |
| rank_truncation | timeofday | -6.0261 | +2.7772 | -0.3151 | +0.1572 |
| mean_injection | weather | +0.0000 | -0.0712 | +0.0000 | -0.0012 |
| mean_injection | scene | +0.0000 | +0.1017 | +0.0000 | +0.0017 |
| mean_injection | timeofday | +0.0000 | -0.0712 | +0.0000 | -0.0012 |
| isotropic_noise | weather | -44.1754 | -17.8128 | -1.6705 | +0.0472 |
| isotropic_noise | scene | -34.6297 | -19.8576 | -0.3448 | +0.4924 |
| isotropic_noise | timeofday | -44.1455 | -38.2808 | -1.3164 | -0.8608 |
| mean_interpolation | weather | +0.0000 | +0.0814 | +0.0000 | +0.0014 |
| mean_interpolation | scene | +0.0000 | -0.0102 | +0.0000 | -0.0002 |
| mean_interpolation | timeofday | +0.0000 | -0.1322 | +0.0000 | -0.0022 |

## C2 reduced representations

| Representation | Weather BA | Scene BA | Time BA | Weather P@10 | Scene P@10 | Time P@10 |
| --- | --- | --- | --- | --- | --- | --- |
| native_768 | 0.630239 | 0.695197 | 0.764365 | 0.578637 | 0.624415 | 0.827263 |
| mrl_512 | 0.628169 | 0.705570 | 0.767313 | 0.577721 | 0.622279 | 0.826450 |
| pca_512 | 0.553562 | 0.658922 | 0.751032 | 0.557782 | 0.617904 | 0.830722 |
| mrl_256 | 0.610656 | 0.697964 | 0.770296 | 0.573347 | 0.621974 | 0.824110 |
| pca_256 | 0.588547 | 0.662672 | 0.754581 | 0.560427 | 0.619736 | 0.833367 |
| mrl_128 | 0.549085 | 0.666983 | 0.741939 | 0.563276 | 0.607833 | 0.806612 |
| pca_128 | 0.623574 | 0.683718 | 0.781747 | 0.561851 | 0.622482 | 0.839980 |

## C3 error, identity and attribute effects

| Condition | Max score error | Overlap | Changed / 983 | Prospective covered | Weather delta pp | Scene delta pp | Time delta pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native_ref | 8.882e-16 | 1.000000 | 0 | 983 | +0.00000 | +0.00000 | +0.00000 |
| native_fp16_gallery | 5.783e-05 | 0.999491 | 5 | 576 | -0.02035 | +0.02035 | -0.01017 |
| native_fp16_both | 8.158e-05 | 0.999491 | 5 | 398 | -0.02035 | +0.02035 | -0.01017 |
| native_int8_gallery | 3.952e-03 | 0.993896 | 60 | 0 | -0.04069 | -0.01017 | +0.00000 |
| native_int8_both | 4.817e-03 | 0.992370 | 75 | 0 | -0.02035 | -0.01017 | -0.03052 |
| native_fp32_arithmetic | 7.123e-07 | 1.000000 | 0 | None | +0.00000 | +0.00000 | +0.00000 |
| mean99_ref | 8.882e-16 | 1.000000 | 0 | 983 | +0.00000 | +0.00000 | +0.00000 |
| mean99_fp16_gallery | 2.946e-07 | 0.979959 | 195 | 0 | +0.04069 | +0.02035 | +0.18311 |

## Execution

[ours] `{"canonical_cache_preserved": true, "claim_label": "[ours]", "conditions": 46, "endpoint_calls": 46, "excluded_validation_count": 17, "final_probe_fits_this_invocation": 174, "guards": "network/image opening/encoder construction blocked", "historical_files_preserved": 491, "identity": {"freeze_sha256": "e083fd6f246d6dec8bc001da7211f512f661762d391a28bd24d95bdab9a3ee59", "implementation_commit": "2feef78208c41d2b3338631dbe022b005e1e9e9b"}, "peak_rss_mib": 874.43359375, "pilot_not_rerun": true, "protocol_deviations": [], "train_count": 2000, "training_bytes_unchanged": true, "validation_count": 983, "wall_seconds": 731.692881628056}`.

[interpretation] See interpretation.md for qualified findings. Absolute/effect
changes are not equivalence tests. The source qualification remains conditional
on enriched-fragment provenance; instance semantics and near-duplicates are untested.
