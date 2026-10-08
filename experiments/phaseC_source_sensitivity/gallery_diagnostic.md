# Post-audit source exclusion sensitivity

[ours] Exact 17-row exclusion; unchanged 2,000 training vectors, 983 validation
queries/gallery rows. This is a post-audit sensitivity, not independent replication.
All tables are generated from frozen checksummed results. Per-class zero supports
are unscorable; historical balanced-probe eligibility is unchanged.

## Fixed retained queries: query versus gallery removal

[ours] Same 983 queries in middle/final columns. Effects use the relevant
reference under each gallery. C3 old-gallery values reuse accepted per-query
records, rather than recomputing old endpoints. Conditional chance is the
retained-query mean (matching gallery labels minus self)/(gallery N minus 1).

| Condition | Attribute | Historical 1000/1000 | Retained 983 / old 1000 | Reduced 983/983 | Query removal pp | Gallery removal pp |
| --- | --- | --- | --- | --- | --- | --- |
| c1_pristine | weather | 0.572200 | 0.578637 | 0.578637 | +0.64368 | +0.00000 |
| c1_pristine | scene | 0.616700 | 0.624415 | 0.624415 | +0.77151 | +0.00000 |
| c1_pristine | timeofday | 0.819300 | 0.827263 | 0.827263 | +0.79635 | +0.00000 |
| c1_scale_contraction_0 | weather | 0.572200 | 0.578637 | 0.578637 | +0.64368 | +0.00000 |
| c1_scale_contraction_0 | scene | 0.616700 | 0.624415 | 0.624415 | +0.77151 | +0.00000 |
| c1_scale_contraction_0 | timeofday | 0.819300 | 0.827263 | 0.827263 | +0.79635 | +0.00000 |
| c1_scale_contraction_0.25 | weather | 0.572200 | 0.578637 | 0.578637 | +0.64368 | +0.00000 |
| c1_scale_contraction_0.25 | scene | 0.616700 | 0.624415 | 0.624415 | +0.77151 | +0.00000 |
| c1_scale_contraction_0.25 | timeofday | 0.819300 | 0.827263 | 0.827263 | +0.79635 | +0.00000 |
| c1_scale_contraction_0.5 | weather | 0.572200 | 0.578637 | 0.578637 | +0.64368 | +0.00000 |
| c1_scale_contraction_0.5 | scene | 0.616700 | 0.624415 | 0.624415 | +0.77151 | +0.00000 |
| c1_scale_contraction_0.5 | timeofday | 0.819300 | 0.827263 | 0.827263 | +0.79635 | +0.00000 |
| c1_scale_contraction_0.75 | weather | 0.572200 | 0.578637 | 0.578637 | +0.64368 | +0.00000 |
| c1_scale_contraction_0.75 | scene | 0.616700 | 0.624415 | 0.624415 | +0.77151 | +0.00000 |
| c1_scale_contraction_0.75 | timeofday | 0.819300 | 0.827263 | 0.827263 | +0.79635 | +0.00000 |
| c1_scale_contraction_0.9 | weather | 0.572200 | 0.578637 | 0.578637 | +0.64368 | +0.00000 |
| c1_scale_contraction_0.9 | scene | 0.616700 | 0.624415 | 0.624415 | +0.77151 | +0.00000 |
| c1_scale_contraction_0.9 | timeofday | 0.819300 | 0.827263 | 0.827263 | +0.79635 | +0.00000 |
| c1_scale_contraction_0.99 | weather | 0.572200 | 0.578637 | 0.578637 | +0.64368 | +0.00000 |
| c1_scale_contraction_0.99 | scene | 0.616700 | 0.624415 | 0.624415 | +0.77151 | +0.00000 |
| c1_scale_contraction_0.99 | timeofday | 0.819300 | 0.827263 | 0.827263 | +0.79635 | +0.00000 |
| c1_rank_truncation_0 | weather | 0.572200 | 0.578637 | 0.578637 | +0.64368 | +0.00000 |
| c1_rank_truncation_0 | scene | 0.616700 | 0.624415 | 0.624415 | +0.77151 | +0.00000 |
| c1_rank_truncation_0 | timeofday | 0.819300 | 0.827263 | 0.827263 | +0.79635 | +0.00000 |
| c1_rank_truncation_0.25 | weather | 0.572200 | 0.578637 | 0.578637 | +0.64368 | +0.00000 |
| c1_rank_truncation_0.25 | scene | 0.616700 | 0.624415 | 0.624415 | +0.77151 | +0.00000 |
| c1_rank_truncation_0.25 | timeofday | 0.819300 | 0.827263 | 0.827263 | +0.79635 | +0.00000 |
| c1_rank_truncation_0.5 | weather | 0.571400 | 0.577620 | 0.577620 | +0.62195 | +0.00000 |
| c1_rank_truncation_0.5 | scene | 0.616800 | 0.624517 | 0.624517 | +0.77168 | +0.00000 |
| c1_rank_truncation_0.5 | timeofday | 0.821300 | 0.829400 | 0.829400 | +0.80998 | +0.00000 |
| c1_rank_truncation_0.75 | weather | 0.568200 | 0.574262 | 0.574262 | +0.60625 | +0.00000 |
| c1_rank_truncation_0.75 | scene | 0.619900 | 0.627670 | 0.627670 | +0.77704 | +0.00000 |
| c1_rank_truncation_0.75 | timeofday | 0.828500 | 0.836623 | 0.836623 | +0.81226 | -0.00000 |
| c1_rank_truncation_0.9 | weather | 0.564700 | 0.570295 | 0.570295 | +0.55950 | +0.00000 |
| c1_rank_truncation_0.9 | scene | 0.627300 | 0.635097 | 0.635504 | +0.77966 | +0.04069 |
| c1_rank_truncation_0.9 | timeofday | 0.837600 | 0.845677 | 0.845778 | +0.80765 | +0.01017 |
| c1_rank_truncation_0.99 | weather | 0.521200 | 0.526653 | 0.527263 | +0.54531 | +0.06104 |
| c1_rank_truncation_0.99 | scene | 0.652500 | 0.660834 | 0.661750 | +0.83342 | +0.09156 |
| c1_rank_truncation_0.99 | timeofday | 0.845500 | 0.854425 | 0.855036 | +0.89252 | +0.06104 |
| c1_mean_injection_0 | weather | 0.572200 | 0.578637 | 0.578637 | +0.64368 | +0.00000 |
| c1_mean_injection_0 | scene | 0.616700 | 0.624415 | 0.624415 | +0.77151 | +0.00000 |
| c1_mean_injection_0 | timeofday | 0.819300 | 0.827263 | 0.827263 | +0.79635 | +0.00000 |
| c1_mean_injection_0.25 | weather | 0.572900 | 0.579349 | 0.579349 | +0.64489 | +0.00000 |
| c1_mean_injection_0.25 | scene | 0.618000 | 0.625738 | 0.625738 | +0.77375 | +0.00000 |
| c1_mean_injection_0.25 | timeofday | 0.819600 | 0.827569 | 0.827569 | +0.79687 | +0.00000 |
| c1_mean_injection_0.5 | weather | 0.571700 | 0.578128 | 0.578128 | +0.64282 | +0.00000 |
| c1_mean_injection_0.5 | scene | 0.619100 | 0.626857 | 0.626857 | +0.77566 | +0.00000 |
| c1_mean_injection_0.5 | timeofday | 0.818800 | 0.826755 | 0.826755 | +0.79548 | +0.00000 |
| c1_mean_injection_0.75 | weather | 0.573100 | 0.579552 | 0.579552 | +0.64524 | +0.00000 |
| c1_mean_injection_0.75 | scene | 0.618600 | 0.626348 | 0.626348 | +0.77479 | +0.00000 |
| c1_mean_injection_0.75 | timeofday | 0.820100 | 0.828077 | 0.828077 | +0.79773 | +0.00000 |
| c1_mean_injection_0.9 | weather | 0.571400 | 0.577823 | 0.577823 | +0.64230 | +0.00000 |
| c1_mean_injection_0.9 | scene | 0.618300 | 0.626043 | 0.626043 | +0.77427 | +0.00000 |
| c1_mean_injection_0.9 | timeofday | 0.819400 | 0.827365 | 0.827365 | +0.79652 | +0.00000 |
| c1_mean_injection_0.99 | weather | 0.571500 | 0.577925 | 0.577925 | +0.64247 | +0.00000 |
| c1_mean_injection_0.99 | scene | 0.617700 | 0.625432 | 0.625432 | +0.77323 | +0.00000 |
| c1_mean_injection_0.99 | timeofday | 0.818600 | 0.826551 | 0.826551 | +0.79514 | +0.00000 |
| c1_isotropic_noise_0 | weather | 0.572200 | 0.578637 | 0.578637 | +0.64368 | +0.00000 |
| c1_isotropic_noise_0 | scene | 0.616700 | 0.624415 | 0.624415 | +0.77151 | +0.00000 |
| c1_isotropic_noise_0 | timeofday | 0.819300 | 0.827263 | 0.827263 | +0.79635 | +0.00000 |
| c1_isotropic_noise_0.25 | weather | 0.565900 | 0.572126 | 0.572126 | +0.62261 | +0.00000 |
| c1_isotropic_noise_0.25 | scene | 0.608200 | 0.615666 | 0.615666 | +0.74663 | +0.00000 |
| c1_isotropic_noise_0.25 | timeofday | 0.815200 | 0.823093 | 0.823093 | +0.78926 | -0.00000 |
| c1_isotropic_noise_0.5 | weather | 0.506100 | 0.510682 | 0.510682 | +0.45816 | +0.00000 |
| c1_isotropic_noise_0.5 | scene | 0.525500 | 0.531740 | 0.531740 | +0.62396 | -0.00000 |
| c1_isotropic_noise_0.5 | timeofday | 0.726600 | 0.733266 | 0.733367 | +0.66655 | +0.01017 |
| c1_isotropic_noise_0.75 | weather | 0.420000 | 0.423601 | 0.425941 | +0.36012 | +0.23398 |
| c1_isotropic_noise_0.75 | scene | 0.426400 | 0.431536 | 0.435707 | +0.51361 | +0.41709 |
| c1_isotropic_noise_0.75 | timeofday | 0.516000 | 0.518413 | 0.519736 | +0.24130 | +0.13225 |
| c1_isotropic_noise_0.9 | weather | 0.398400 | 0.401628 | 0.403662 | +0.32277 | +0.20346 |
| c1_isotropic_noise_0.9 | scene | 0.415700 | 0.420855 | 0.426653 | +0.51545 | +0.57986 |
| c1_isotropic_noise_0.9 | timeofday | 0.455800 | 0.456256 | 0.455748 | +0.04564 | -0.05086 |
| c1_isotropic_noise_0.99 | weather | 0.393600 | 0.397050 | 0.400509 | +0.34498 | +0.34588 |
| c1_isotropic_noise_0.99 | scene | 0.413200 | 0.418413 | 0.425839 | +0.52130 | +0.74262 |
| c1_isotropic_noise_0.99 | timeofday | 0.445100 | 0.445371 | 0.444456 | +0.02713 | -0.09156 |
| c1_mean_interpolation_0 | weather | 0.572200 | 0.578637 | 0.578637 | +0.64368 | +0.00000 |
| c1_mean_interpolation_0 | scene | 0.616700 | 0.624415 | 0.624415 | +0.77151 | +0.00000 |
| c1_mean_interpolation_0 | timeofday | 0.819300 | 0.827263 | 0.827263 | +0.79635 | +0.00000 |
| c1_mean_interpolation_0.25 | weather | 0.573100 | 0.579552 | 0.579552 | +0.64524 | +0.00000 |
| c1_mean_interpolation_0.25 | scene | 0.616500 | 0.624212 | 0.624212 | +0.77116 | +0.00000 |
| c1_mean_interpolation_0.25 | timeofday | 0.819200 | 0.827162 | 0.827162 | +0.79617 | -0.00000 |
| c1_mean_interpolation_0.5 | weather | 0.574000 | 0.580468 | 0.580468 | +0.64680 | -0.00000 |
| c1_mean_interpolation_0.5 | scene | 0.617500 | 0.625229 | 0.625229 | +0.77289 | -0.00000 |
| c1_mean_interpolation_0.5 | timeofday | 0.817700 | 0.825636 | 0.825636 | +0.79358 | -0.00000 |
| c1_mean_interpolation_0.75 | weather | 0.573100 | 0.579552 | 0.579552 | +0.64524 | +0.00000 |
| c1_mean_interpolation_0.75 | scene | 0.617400 | 0.625127 | 0.625127 | +0.77272 | +0.00000 |
| c1_mean_interpolation_0.75 | timeofday | 0.817400 | 0.825331 | 0.825331 | +0.79306 | +0.00000 |
| c1_mean_interpolation_0.9 | weather | 0.572700 | 0.579145 | 0.579145 | +0.64455 | +0.00000 |
| c1_mean_interpolation_0.9 | scene | 0.617100 | 0.624822 | 0.624822 | +0.77220 | +0.00000 |
| c1_mean_interpolation_0.9 | timeofday | 0.817800 | 0.825738 | 0.825738 | +0.79375 | -0.00000 |
| c1_mean_interpolation_0.99 | weather | 0.573000 | 0.579451 | 0.579451 | +0.64507 | +0.00000 |
| c1_mean_interpolation_0.99 | scene | 0.616600 | 0.624313 | 0.624313 | +0.77133 | +0.00000 |
| c1_mean_interpolation_0.99 | timeofday | 0.818000 | 0.825941 | 0.825941 | +0.79410 | -0.00000 |
| c2_native_768 | weather | 0.572200 | 0.578637 | 0.578637 | +0.64368 | +0.00000 |
| c2_native_768 | scene | 0.616700 | 0.624415 | 0.624415 | +0.77151 | +0.00000 |
| c2_native_768 | timeofday | 0.819300 | 0.827263 | 0.827263 | +0.79635 | +0.00000 |
| c2_mrl_512 | weather | 0.571300 | 0.577721 | 0.577721 | +0.64213 | +0.00000 |
| c2_mrl_512 | scene | 0.614700 | 0.622279 | 0.622279 | +0.75787 | -0.00000 |
| c2_mrl_512 | timeofday | 0.818500 | 0.826450 | 0.826450 | +0.79496 | -0.00000 |
| c2_pca_512 | weather | 0.551700 | 0.557782 | 0.557782 | +0.60823 | +0.00000 |
| c2_pca_512 | scene | 0.610400 | 0.617904 | 0.617904 | +0.75044 | +0.00000 |
| c2_pca_512 | timeofday | 0.822800 | 0.830722 | 0.830722 | +0.79223 | +0.00000 |
| c2_mrl_256 | weather | 0.567100 | 0.573347 | 0.573347 | +0.62469 | +0.00000 |
| c2_mrl_256 | scene | 0.614400 | 0.621974 | 0.621974 | +0.75736 | +0.00000 |
| c2_mrl_256 | timeofday | 0.816100 | 0.824110 | 0.824110 | +0.80099 | -0.00000 |
| c2_pca_256 | weather | 0.554500 | 0.560427 | 0.560427 | +0.59273 | -0.00000 |
| c2_pca_256 | scene | 0.612100 | 0.619736 | 0.619736 | +0.76355 | -0.00000 |
| c2_pca_256 | timeofday | 0.825400 | 0.833367 | 0.833367 | +0.79672 | -0.00000 |
| c2_mrl_128 | weather | 0.557200 | 0.563276 | 0.563276 | +0.60757 | +0.00000 |
| c2_mrl_128 | scene | 0.600400 | 0.607833 | 0.607833 | +0.74332 | -0.00000 |
| c2_mrl_128 | timeofday | 0.798800 | 0.806612 | 0.806612 | +0.78124 | +0.00000 |
| c2_pca_128 | weather | 0.556000 | 0.561750 | 0.561851 | +0.57497 | +0.01017 |
| c2_pca_128 | scene | 0.614500 | 0.622380 | 0.622482 | +0.78805 | +0.01017 |
| c2_pca_128 | timeofday | 0.831500 | 0.839878 | 0.839980 | +0.83779 | +0.01017 |
| c3_native_ref | weather | 0.572200 | 0.578637 | 0.578637 | +0.64368 | +0.00000 |
| c3_native_ref | scene | 0.616700 | 0.624415 | 0.624415 | +0.77151 | +0.00000 |
| c3_native_ref | timeofday | 0.819300 | 0.827263 | 0.827263 | +0.79635 | +0.00000 |
| c3_native_fp16_gallery | weather | 0.572000 | 0.578433 | 0.578433 | +0.64334 | +0.00000 |
| c3_native_fp16_gallery | scene | 0.616900 | 0.624619 | 0.624619 | +0.77185 | +0.00000 |
| c3_native_fp16_gallery | timeofday | 0.819200 | 0.827162 | 0.827162 | +0.79617 | +0.00000 |
| c3_native_fp16_both | weather | 0.572000 | 0.578433 | 0.578433 | +0.64334 | +0.00000 |
| c3_native_fp16_both | scene | 0.616900 | 0.624619 | 0.624619 | +0.77185 | +0.00000 |
| c3_native_fp16_both | timeofday | 0.819200 | 0.827162 | 0.827162 | +0.79617 | +0.00000 |
| c3_native_int8_gallery | weather | 0.571800 | 0.578230 | 0.578230 | +0.64299 | +0.00000 |
| c3_native_int8_gallery | scene | 0.616600 | 0.624313 | 0.624313 | +0.77133 | +0.00000 |
| c3_native_int8_gallery | timeofday | 0.819300 | 0.827263 | 0.827263 | +0.79635 | +0.00000 |
| c3_native_int8_both | weather | 0.572000 | 0.578433 | 0.578433 | +0.64334 | +0.00000 |
| c3_native_int8_both | scene | 0.616600 | 0.624313 | 0.624313 | +0.77133 | +0.00000 |
| c3_native_int8_both | timeofday | 0.819000 | 0.826958 | 0.826958 | +0.79583 | +0.00000 |
| c3_native_fp32_arithmetic | weather | 0.572200 | 0.578637 | 0.578637 | +0.64368 | +0.00000 |
| c3_native_fp32_arithmetic | scene | 0.616700 | 0.624415 | 0.624415 | +0.77151 | +0.00000 |
| c3_native_fp32_arithmetic | timeofday | 0.819300 | 0.827263 | 0.827263 | +0.79635 | +0.00000 |
| c3_mean99_ref | weather | 0.571500 | 0.577925 | 0.577925 | +0.64247 | +0.00000 |
| c3_mean99_ref | scene | 0.617700 | 0.625432 | 0.625432 | +0.77323 | +0.00000 |
| c3_mean99_ref | timeofday | 0.818600 | 0.826551 | 0.826551 | +0.79514 | +0.00000 |
| c3_mean99_fp16_gallery | weather | 0.571900 | 0.578332 | 0.578332 | +0.64316 | +0.00000 |
| c3_mean99_fp16_gallery | scene | 0.617900 | 0.625636 | 0.625636 | +0.77358 | +0.00000 |
| c3_mean99_fp16_gallery | timeofday | 0.820400 | 0.828383 | 0.828383 | +0.79825 | +0.00000 |

## Conditional retained-query chance

| Attribute | 983 queries / 1000 gallery | 983 queries / 983 gallery |
| --- | --- | --- |
| weather | 0.394333 | 0.398083 |
| scene | 0.426371 | 0.431594 |
| timeofday | 0.432806 | 0.433481 |

## C3 identity effects with retained queries under both galleries

| Condition | Old gallery overlap | Old gallery changed / 983 | Reduced gallery overlap | Reduced gallery changed / 983 |
| --- | --- | --- | --- | --- |
| native_ref | 1.0 | 0 | 1.0 | 0 |
| native_fp16_gallery | 0.9994913530010173 | 5 | 0.9994913530010173 | 5 |
| native_fp16_both | 0.9994913530010173 | 5 | 0.9994913530010173 | 5 |
| native_int8_gallery | 0.9938962360122076 | 60 | 0.9938962360122076 | 60 |
| native_int8_both | 0.9923702950152594 | 75 | 0.9923702950152594 | 75 |
| native_fp32_arithmetic | 1.0 | 0 | 1.0 | 0 |
| mean99_ref | 1.0 | 0 | 1.0 | 0 |
| mean99_fp16_gallery | 0.9799593082400815 | 195 | 0.9799593082400815 | 195 |
