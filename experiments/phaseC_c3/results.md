# C3: numerical error, neighbour identities and attribute utility

[ours] Fixed cache-only exact-search experiment. Native 768d is primary;
the sole supporting stress is unchanged training-fitted mean injection .99.
Mean conditions are compared with their own reference. No inference, image
decoding, ANN or C4. All tables are generated from results.json.

Freeze SHA256: `ba3e6db606fc0852d8d38b22f38a5a9adbc0ff99725b12f5a84c3082cdcee44d`.

## Numerical error and identity stability

[ours] 1000 queries; k=10.

| Condition | Mean max score error | Max error | Overlap@10 | Changed sets | Boundary ties | Prospective covered / exceptions | Retrospective covered | Gallery bytes | Score dtype |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| native_ref | 0.000e+00 | 0.000e+00 | 1.000000 | 0 | 0 | 1000 / 0 | 1000 | 3072000 | float64 |
| native_fp16_gallery | 4.314e-05 | 6.012e-05 | 0.999500 | 5 | 0 | 592 / 0 | 938 | 1536000 | float64 |
| native_fp16_both | 5.286e-05 | 1.021e-04 | 0.999500 | 5 | 0 | 412 / 0 | 916 | 1536000 | float64 |
| native_int8_gallery | 2.356e-03 | 3.952e-03 | 0.994000 | 60 | 0 | 0 / 0 | 32 | 774144 | float64 |
| native_int8_both | 2.429e-03 | 4.817e-03 | 0.992500 | 75 | 0 | 0 / 0 | 34 | 774144 | float64 |
| native_fp32_arithmetic | 5.123e-07 | 7.123e-07 | 1.000000 | 0 | 0 | n/a | 999 | 3072000 | float32 |
| mean99_ref | 0.000e+00 | 0.000e+00 | 1.000000 | 0 | 0 | 1000 / 0 | 1000 | 6144000 | float64 |
| mean99_fp16_gallery | 2.032e-07 | 2.946e-07 | 0.980300 | 195 | 0 | 0 / 0 | 45 | 1536000 | float64 |

## Attribute utility

[ours] Micro P@10; delta in percentage points, relative to same-input reference.

| Condition | Weather P@10 | Delta pp | Scene P@10 | Delta pp | Time P@10 | Delta pp |
|---|---:|---:|---:|---:|---:|---:|
| native_ref | 0.572200 | +0.0000 | 0.616700 | +0.0000 | 0.819300 | +0.0000 |
| native_fp16_gallery | 0.572000 | -0.0200 | 0.616900 | +0.0200 | 0.819200 | -0.0100 |
| native_fp16_both | 0.572000 | -0.0200 | 0.616900 | +0.0200 | 0.819200 | -0.0100 |
| native_int8_gallery | 0.571800 | -0.0400 | 0.616600 | -0.0100 | 0.819300 | +0.0000 |
| native_int8_both | 0.572000 | -0.0200 | 0.616600 | -0.0100 | 0.819000 | -0.0300 |
| native_fp32_arithmetic | 0.572200 | +0.0000 | 0.616700 | +0.0000 | 0.819300 | +0.0000 |
| mean99_ref | 0.571500 | +0.0000 | 0.617700 | +0.0000 | 0.818600 | +0.0000 |
| mean99_fp16_gallery | 0.571900 | +0.0400 | 0.617900 | +0.0200 | 0.820400 | +0.1800 |

## Turnover counts

[ours] Entering/departing label counts, with no arbitrary neighbour pairing.

| Condition | Attribute | Gain / loss / unchanged | Changed sets with unchanged P@10 | Changed sets with unchanged full label counts |
|---|---|---|---:|---:|
| native_ref | weather | 0 / 0 / 1000 | 0 | 0 |
| native_ref | scene | 0 / 0 / 1000 | 0 | 0 |
| native_ref | timeofday | 0 / 0 / 1000 | 0 | 0 |
| native_fp16_gallery | weather | 0 / 2 / 998 | 3 | 2 |
| native_fp16_gallery | scene | 3 / 1 / 996 | 1 | 1 |
| native_fp16_gallery | timeofday | 0 / 1 / 999 | 4 | 4 |
| native_fp16_both | weather | 0 / 2 / 998 | 3 | 2 |
| native_fp16_both | scene | 3 / 1 / 996 | 1 | 1 |
| native_fp16_both | timeofday | 0 / 1 / 999 | 4 | 4 |
| native_int8_gallery | weather | 7 / 11 / 982 | 42 | 32 |
| native_int8_gallery | scene | 12 / 13 / 975 | 35 | 32 |
| native_int8_gallery | timeofday | 9 / 9 / 982 | 42 | 42 |
| native_int8_both | weather | 12 / 14 / 974 | 49 | 37 |
| native_int8_both | scene | 15 / 16 / 969 | 44 | 40 |
| native_int8_both | timeofday | 9 / 12 / 979 | 54 | 52 |
| native_fp32_arithmetic | weather | 0 / 0 / 1000 | 0 | 0 |
| native_fp32_arithmetic | scene | 0 / 0 / 1000 | 0 | 0 |
| native_fp32_arithmetic | timeofday | 0 / 0 / 1000 | 0 | 0 |
| mean99_ref | weather | 0 / 0 / 1000 | 0 | 0 |
| mean99_ref | scene | 0 / 0 / 1000 | 0 | 0 |
| mean99_ref | timeofday | 0 / 0 / 1000 | 0 | 0 |
| mean99_fp16_gallery | weather | 34 / 30 / 936 | 131 | 102 |
| mean99_fp16_gallery | scene | 39 / 37 / 924 | 119 | 116 |
| mean99_fp16_gallery | timeofday | 32 / 14 / 954 | 149 | 141 |

## Support and floors

[ours] Full natural-distribution gallery. Macro retrieval includes classes with
validation support>=10; all gallery rows remain. Full per-class deltas and macro
scores are retained in results.json.

| Attribute | Class | Validation support | Reference P@10 |
|---|---|---:|---:|
| weather | clear | 588 | 0.774830 |
| weather | foggy | 1 | 0.000000 |
| weather | overcast | 152 | 0.357237 |
| weather | partly cloudy | 81 | 0.220988 |
| weather | rainy | 92 | 0.246739 |
| weather | snowy | 86 | 0.252326 |
| scene | city street | 577 | 0.696534 |
| scene | gas stations | 4 | 0.200000 |
| scene | highway | 254 | 0.549606 |
| scene | parking lot | 5 | 0.060000 |
| scene | residential | 156 | 0.469231 |
| scene | tunnel | 4 | 0.225000 |
| timeofday | dawn/dusk | 78 | 0.226923 |
| timeofday | daytime | 489 | 0.811452 |
| timeofday | night | 433 | 0.934873 |

| Attribute | sum(p²) chance | Exact self-excluded chance |
|---|---:|---:|
| weather | 0.391270 | 0.390661 |
| scene | 0.421838 | 0.421259 |
| timeofday | 0.432694 | 0.432126 |

## Margin analysis

[ours] Tied reference margins stay together; bins are separately defined for
native and mean99. Coverage is an empirical sufficient-condition check,
not rigorous floating-point certification.

| Condition | Bin | N | Margin min / max | Overlap | Changed | Prospective / retrospective covered | Weather / scene / time delta pp |
|---|---:|---:|---|---:|---:|---|---|
| native_ref | 0 | 250 | 7.899e-07 / 2.982e-04 | 1.000000 | 0 | 250 / 250 | +0.0000 / +0.0000 / +0.0000 |
| native_ref | 1 | 250 | 2.984e-04 / 7.382e-04 | 1.000000 | 0 | 250 / 250 | +0.0000 / +0.0000 / +0.0000 |
| native_ref | 2 | 250 | 7.396e-04 / 1.609e-03 | 1.000000 | 0 | 250 / 250 | +0.0000 / +0.0000 / +0.0000 |
| native_ref | 3 | 250 | 1.612e-03 / 1.816e-02 | 1.000000 | 0 | 250 / 250 | +0.0000 / +0.0000 / +0.0000 |
| native_fp16_gallery | 0 | 250 | 7.899e-07 / 2.982e-04 | 0.998000 | 5 | 0 / 188 | -0.0800 / +0.0800 / -0.0400 |
| native_fp16_gallery | 1 | 250 | 2.984e-04 / 7.382e-04 | 1.000000 | 0 | 92 / 250 | +0.0000 / +0.0000 / +0.0000 |
| native_fp16_gallery | 2 | 250 | 7.396e-04 / 1.609e-03 | 1.000000 | 0 | 250 / 250 | +0.0000 / +0.0000 / +0.0000 |
| native_fp16_gallery | 3 | 250 | 1.612e-03 / 1.816e-02 | 1.000000 | 0 | 250 / 250 | +0.0000 / +0.0000 / +0.0000 |
| native_fp16_both | 0 | 250 | 7.899e-07 / 2.982e-04 | 0.998000 | 5 | 0 / 166 | -0.0800 / +0.0800 / -0.0400 |
| native_fp16_both | 1 | 250 | 2.984e-04 / 7.382e-04 | 1.000000 | 0 | 0 / 250 | +0.0000 / +0.0000 / +0.0000 |
| native_fp16_both | 2 | 250 | 7.396e-04 / 1.609e-03 | 1.000000 | 0 | 162 / 250 | +0.0000 / +0.0000 / +0.0000 |
| native_fp16_both | 3 | 250 | 1.612e-03 / 1.816e-02 | 1.000000 | 0 | 250 / 250 | +0.0000 / +0.0000 / +0.0000 |
| native_int8_gallery | 0 | 250 | 7.899e-07 / 2.982e-04 | 0.978000 | 55 | 0 / 0 | -0.1200 / +0.0400 / -0.1200 |
| native_int8_gallery | 1 | 250 | 2.984e-04 / 7.382e-04 | 0.998000 | 5 | 0 / 0 | -0.0400 / -0.0800 / +0.1200 |
| native_int8_gallery | 2 | 250 | 7.396e-04 / 1.609e-03 | 1.000000 | 0 | 0 / 0 | +0.0000 / +0.0000 / +0.0000 |
| native_int8_gallery | 3 | 250 | 1.612e-03 / 1.816e-02 | 1.000000 | 0 | 0 / 32 | +0.0000 / +0.0000 / +0.0000 |
| native_int8_both | 0 | 250 | 7.899e-07 / 2.982e-04 | 0.973200 | 67 | 0 / 0 | -0.0800 / +0.1200 / -0.2400 |
| native_int8_both | 1 | 250 | 2.984e-04 / 7.382e-04 | 0.996800 | 8 | 0 / 0 | +0.0000 / -0.1600 / +0.1200 |
| native_int8_both | 2 | 250 | 7.396e-04 / 1.609e-03 | 1.000000 | 0 | 0 / 0 | +0.0000 / +0.0000 / +0.0000 |
| native_int8_both | 3 | 250 | 1.612e-03 / 1.816e-02 | 1.000000 | 0 | 0 / 34 | +0.0000 / +0.0000 / +0.0000 |
| native_fp32_arithmetic | 0 | 250 | 7.899e-07 / 2.982e-04 | 1.000000 | 0 | None / 249 | +0.0000 / +0.0000 / +0.0000 |
| native_fp32_arithmetic | 1 | 250 | 2.984e-04 / 7.382e-04 | 1.000000 | 0 | None / 250 | +0.0000 / +0.0000 / +0.0000 |
| native_fp32_arithmetic | 2 | 250 | 7.396e-04 / 1.609e-03 | 1.000000 | 0 | None / 250 | +0.0000 / +0.0000 / +0.0000 |
| native_fp32_arithmetic | 3 | 250 | 1.612e-03 / 1.816e-02 | 1.000000 | 0 | None / 250 | +0.0000 / +0.0000 / +0.0000 |
| mean99_ref | 0 | 250 | 2.011e-10 / 3.505e-08 | 1.000000 | 0 | 250 / 250 | +0.0000 / +0.0000 / +0.0000 |
| mean99_ref | 1 | 250 | 3.527e-08 / 7.709e-08 | 1.000000 | 0 | 250 / 250 | +0.0000 / +0.0000 / +0.0000 |
| mean99_ref | 2 | 250 | 7.729e-08 / 1.611e-07 | 1.000000 | 0 | 250 / 250 | +0.0000 / +0.0000 / +0.0000 |
| mean99_ref | 3 | 250 | 1.616e-07 / 1.889e-06 | 1.000000 | 0 | 250 / 250 | +0.0000 / +0.0000 / +0.0000 |
| mean99_fp16_gallery | 0 | 250 | 2.011e-10 / 3.505e-08 | 0.950800 | 121 | 0 / 0 | +0.4000 / -0.1200 / +0.5200 |
| mean99_fp16_gallery | 1 | 250 | 3.527e-08 / 7.709e-08 | 0.978000 | 55 | 0 / 0 | -0.0400 / +0.2800 / +0.2400 |
| mean99_fp16_gallery | 2 | 250 | 7.729e-08 / 1.611e-07 | 0.992800 | 18 | 0 / 0 | -0.1600 / -0.0800 / -0.0800 |
| mean99_fp16_gallery | 3 | 250 | 1.616e-07 / 1.889e-06 | 0.999600 | 1 | 0 / 45 | -0.0400 / +0.0000 / +0.0400 |

## Runtime and exceptions

[ours] Completing invocation 6.226s; summed condition time 2.562s; peak completing-process RSS 744.38 MiB; 40 historical files preserved.

| Condition | Wall seconds | Clipped coordinates / rows | Outside range | Max arithmetic residual |
|---|---:|---|---:|---:|
| native_ref | 0.332 | 0 / 0 | 0 | 0.000e+00 |
| native_fp16_gallery | 0.367 | 0 / 0 | 0 | 0.000e+00 |
| native_fp16_both | 0.336 | 0 / 0 | 0 | 0.000e+00 |
| native_int8_gallery | 0.319 | 377 / 263 | 397 | 0.000e+00 |
| native_int8_both | 0.322 | 377 / 263 | 397 | 0.000e+00 |
| native_fp32_arithmetic | 0.279 | 0 / 0 | 0 | 6.474e-07 |
| mean99_ref | 0.296 | 0 / 0 | 0 | 0.000e+00 |
| mean99_fp16_gallery | 0.312 | 0 / 0 | 0 | 0.000e+00 |

[ours] C3 references minus committed C1 P@10 (ordering continuity):

| Reference | Weather | Scene | Time |
|---|---:|---:|---:|
| mean99_ref | +0.000000000 | +0.000000000 | +0.000000000 |
| native_ref | +0.000000000 | +0.000000000 | +0.000000000 |

[interpretation] Identity stability, unchanged attribute counts and unmeasured
instance semantics are distinct. Numerical bounds are sufficient and can be loose.
Storage bytes exclude reconstructed analysis arrays and do not imply operational
search speedups. No query-seed intervals or post-hoc success threshold is supplied.
See interpretation.md for assessment and verification.json for checks.
