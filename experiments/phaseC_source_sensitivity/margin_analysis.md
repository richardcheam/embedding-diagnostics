# Post-audit source exclusion sensitivity

[ours] Exact 17-row exclusion; unchanged 2,000 training vectors, 983 validation
queries/gallery rows. This is a post-audit sensitivity, not independent replication.
All tables are generated from frozen checksummed results. Per-class zero supports
are unscorable; historical balanced-probe eligibility is unchanged.

## C3 rebuilt margins and empirical sufficient checks

[established] Strict gamma > 2 epsilon is a sufficient set-order condition;
failure to satisfy it does not imply instability. Prospective reconstruction
bounds are separate from retrospective score errors; neither is rigorous
floating-point certification. Arithmetic-only prospective checks are n/a.

| Condition | Bin | N | Margin min/max | Overlap | Changed | Prospective covered | Retrospective covered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native_ref | 0 | 246 | 7.899e-07 / 2.909e-04 | 1.000000 | 0 | 246 | 246 |
| native_ref | 1 | 246 | 2.936e-04 / 7.248e-04 | 1.000000 | 0 | 246 | 246 |
| native_ref | 2 | 246 | 7.298e-04 / 1.588e-03 | 1.000000 | 0 | 246 | 246 |
| native_ref | 3 | 245 | 1.589e-03 / 1.170e-02 | 1.000000 | 0 | 245 | 245 |
| native_fp16_gallery | 0 | 246 | 7.899e-07 / 2.909e-04 | 0.997967 | 5 | 0 | 184 |
| native_fp16_gallery | 1 | 246 | 2.936e-04 / 7.248e-04 | 1.000000 | 0 | 85 | 246 |
| native_fp16_gallery | 2 | 246 | 7.298e-04 / 1.588e-03 | 1.000000 | 0 | 246 | 246 |
| native_fp16_gallery | 3 | 245 | 1.589e-03 / 1.170e-02 | 1.000000 | 0 | 245 | 245 |
| native_fp16_both | 0 | 246 | 7.899e-07 / 2.909e-04 | 0.997967 | 5 | 0 | 166 |
| native_fp16_both | 1 | 246 | 2.936e-04 / 7.248e-04 | 1.000000 | 0 | 0 | 246 |
| native_fp16_both | 2 | 246 | 7.298e-04 / 1.588e-03 | 1.000000 | 0 | 153 | 246 |
| native_fp16_both | 3 | 245 | 1.589e-03 / 1.170e-02 | 1.000000 | 0 | 245 | 245 |
| native_int8_gallery | 0 | 246 | 7.899e-07 / 2.909e-04 | 0.977642 | 55 | 0 | 0 |
| native_int8_gallery | 1 | 246 | 2.936e-04 / 7.248e-04 | 0.997967 | 5 | 0 | 0 |
| native_int8_gallery | 2 | 246 | 7.298e-04 / 1.588e-03 | 1.000000 | 0 | 0 | 0 |
| native_int8_gallery | 3 | 245 | 1.589e-03 / 1.170e-02 | 1.000000 | 0 | 0 | 20 |
| native_int8_both | 0 | 246 | 7.899e-07 / 2.909e-04 | 0.973171 | 66 | 0 | 0 |
| native_int8_both | 1 | 246 | 2.936e-04 / 7.248e-04 | 0.996341 | 9 | 0 | 0 |
| native_int8_both | 2 | 246 | 7.298e-04 / 1.588e-03 | 1.000000 | 0 | 0 | 0 |
| native_int8_both | 3 | 245 | 1.589e-03 / 1.170e-02 | 1.000000 | 0 | 0 | 23 |
| native_fp32_arithmetic | 0 | 246 | 7.899e-07 / 2.909e-04 | 1.000000 | 0 | None | 245 |
| native_fp32_arithmetic | 1 | 246 | 2.936e-04 / 7.248e-04 | 1.000000 | 0 | None | 246 |
| native_fp32_arithmetic | 2 | 246 | 7.298e-04 / 1.588e-03 | 1.000000 | 0 | None | 246 |
| native_fp32_arithmetic | 3 | 245 | 1.589e-03 / 1.170e-02 | 1.000000 | 0 | None | 245 |
| mean99_ref | 0 | 246 | 2.011e-10 / 3.465e-08 | 1.000000 | 0 | 246 | 246 |
| mean99_ref | 1 | 246 | 3.466e-08 / 7.565e-08 | 1.000000 | 0 | 246 | 246 |
| mean99_ref | 2 | 246 | 7.611e-08 / 1.580e-07 | 1.000000 | 0 | 246 | 246 |
| mean99_ref | 3 | 245 | 1.587e-07 / 1.162e-06 | 1.000000 | 0 | 245 | 245 |
| mean99_fp16_gallery | 0 | 246 | 2.011e-10 / 3.465e-08 | 0.950000 | 121 | 0 | 0 |
| mean99_fp16_gallery | 1 | 246 | 3.466e-08 / 7.565e-08 | 0.978455 | 53 | 0 | 0 |
| mean99_fp16_gallery | 2 | 246 | 7.611e-08 / 1.580e-07 | 0.991870 | 20 | 0 | 0 |
| mean99_fp16_gallery | 3 | 245 | 1.587e-07 / 1.162e-06 | 0.999592 | 1 | 0 | 34 |

## Attribute turnover, zero supports and floors

| Condition | Attribute | Gain | Loss | Unchanged | Changed set/same P@10 | Changed set/same label counts | P@10 | Macro | sum(p²) | Self-excluded chance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native_ref | weather | 0 | 0 | 983 | 0 | 0 | 0.578637 | 0.371181 | 0.398695 | 0.398083 |
| native_ref | scene | 0 | 0 | 983 | 0 | 0 | 0.624415 | 0.577353 | 0.432172 | 0.431594 |
| native_ref | timeofday | 0 | 0 | 983 | 0 | 0 | 0.827263 | 0.663795 | 0.434058 | 0.433481 |
| native_fp16_gallery | weather | 0 | 2 | 981 | 3 | 2 | 0.578433 | 0.371113 | 0.398695 | 0.398083 |
| native_fp16_gallery | scene | 3 | 1 | 979 | 1 | 1 | 0.624619 | 0.577619 | 0.432172 | 0.431594 |
| native_fp16_gallery | timeofday | 0 | 1 | 982 | 4 | 4 | 0.827162 | 0.663717 | 0.434058 | 0.433481 |
| native_fp16_both | weather | 0 | 2 | 981 | 3 | 2 | 0.578433 | 0.371113 | 0.398695 | 0.398083 |
| native_fp16_both | scene | 3 | 1 | 979 | 1 | 1 | 0.624619 | 0.577619 | 0.432172 | 0.431594 |
| native_fp16_both | timeofday | 0 | 1 | 982 | 4 | 4 | 0.827162 | 0.663717 | 0.434058 | 0.433481 |
| native_int8_gallery | weather | 7 | 11 | 965 | 42 | 32 | 0.578230 | 0.370734 | 0.398695 | 0.398083 |
| native_int8_gallery | scene | 12 | 13 | 958 | 35 | 32 | 0.624313 | 0.577370 | 0.432172 | 0.431594 |
| native_int8_gallery | timeofday | 9 | 9 | 965 | 42 | 42 | 0.827263 | 0.664527 | 0.434058 | 0.433481 |
| native_int8_both | weather | 12 | 14 | 957 | 49 | 37 | 0.578433 | 0.370706 | 0.398695 | 0.398083 |
| native_int8_both | scene | 15 | 16 | 952 | 44 | 40 | 0.624313 | 0.577608 | 0.432172 | 0.431594 |
| native_int8_both | timeofday | 9 | 12 | 962 | 54 | 52 | 0.826958 | 0.664302 | 0.434058 | 0.433481 |
| native_fp32_arithmetic | weather | 0 | 0 | 983 | 0 | 0 | 0.578637 | 0.371181 | 0.398695 | 0.398083 |
| native_fp32_arithmetic | scene | 0 | 0 | 983 | 0 | 0 | 0.624415 | 0.577353 | 0.432172 | 0.431594 |
| native_fp32_arithmetic | timeofday | 0 | 0 | 983 | 0 | 0 | 0.827263 | 0.663795 | 0.434058 | 0.433481 |
| mean99_ref | weather | 0 | 0 | 983 | 0 | 0 | 0.577925 | 0.370320 | 0.398695 | 0.398083 |
| mean99_ref | scene | 0 | 0 | 983 | 0 | 0 | 0.625432 | 0.578007 | 0.432172 | 0.431594 |
| mean99_ref | timeofday | 0 | 0 | 983 | 0 | 0 | 0.826551 | 0.664025 | 0.434058 | 0.433481 |
| mean99_fp16_gallery | weather | 34 | 30 | 919 | 131 | 102 | 0.578332 | 0.371638 | 0.398695 | 0.398083 |
| mean99_fp16_gallery | scene | 39 | 37 | 907 | 119 | 116 | 0.625636 | 0.577972 | 0.432172 | 0.431594 |
| mean99_fp16_gallery | timeofday | 32 | 14 | 937 | 149 | 141 | 0.828383 | 0.665682 | 0.434058 | 0.433481 |

## Numerical errors and exceptions

| Condition | Mean max error | Arithmetic residual | Clipped coords | Clipped rows | Outside range | Ties | Prospective exceptions | Retrospective covered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native_ref | 9.035e-19 | 0.000e+00 | 0 | 0 | 0 | 0 | 0 | 983 |
| native_fp16_gallery | 4.269e-05 | 0.000e+00 | 0 | 0 | 0 | 0 | 0 | 921 |
| native_fp16_both | 5.114e-05 | 0.000e+00 | 0 | 0 | 0 | 0 | 0 | 903 |
| native_int8_gallery | 2.363e-03 | 0.000e+00 | 353 | 246 | 372 | 0 | 0 | 20 |
| native_int8_both | 2.434e-03 | 0.000e+00 | 353 | 246 | 372 | 0 | 0 | 23 |
| native_fp32_arithmetic | 5.117e-07 | 7.053e-07 | 0 | 0 | 0 | 0 | None | 982 |
| mean99_ref | 9.035e-19 | 0.000e+00 | 0 | 0 | 0 | 0 | 0 | 983 |
| mean99_fp16_gallery | 2.027e-07 | 0.000e+00 | 0 | 0 | 0 | 0 | 0 | 34 |
