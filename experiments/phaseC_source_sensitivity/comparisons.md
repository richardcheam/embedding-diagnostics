# Post-audit source exclusion sensitivity

[ours] Exact 17-row exclusion; unchanged 2,000 training vectors, 983 validation
queries/gallery rows. This is a post-audit sensitivity, not independent replication.
All tables are generated from frozen checksummed results. Per-class zero supports
are unscorable; historical balanced-probe eligibility is unchanged.

## Absolute and effect changes

[ours] Percentage points.

| Condition | Attribute | Endpoint | Historical | Reduced | Absolute Δ pp | Old effect pp | New effect pp | Effect change pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c1_pristine | weather | p10 | 0.572200 | 0.578637 | +0.6437 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | weather | macro_p10 | 0.370424 | 0.371181 | +0.0757 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | weather | probe_unscaled_accuracy | 0.784000 | 0.794507 | +1.0507 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | weather | probe_unscaled_balanced_accuracy | 0.606946 | 0.624657 | +1.7711 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | weather | probe_unscaled_macro_f1 | 0.647616 | 0.661353 | +1.3737 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | scene | p10 | 0.616700 | 0.624415 | +0.7715 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | scene | macro_p10 | 0.571790 | 0.577353 | +0.5563 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | scene | probe_unscaled_accuracy | 0.755000 | 0.767040 | +1.2040 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | scene | probe_unscaled_balanced_accuracy | 0.637911 | 0.643897 | +0.5986 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | scene | probe_unscaled_macro_f1 | 0.665750 | 0.674159 | +0.8409 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | timeofday | p10 | 0.819300 | 0.827263 | +0.7963 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | timeofday | macro_p10 | 0.657749 | 0.663795 | +0.6046 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | timeofday | probe_unscaled_accuracy | 0.913000 | 0.922686 | +0.9686 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | timeofday | probe_unscaled_balanced_accuracy | 0.749446 | 0.759524 | +1.0078 | +0.0000 | +0.0000 | +0.0000 |
| c1_pristine | timeofday | probe_unscaled_macro_f1 | 0.766903 | 0.777141 | +1.0238 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | weather | p10 | 0.572200 | 0.578637 | +0.6437 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | weather | macro_p10 | 0.370424 | 0.371181 | +0.0757 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | weather | probe_unscaled_accuracy | 0.784000 | 0.794507 | +1.0507 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | weather | probe_unscaled_balanced_accuracy | 0.606946 | 0.624657 | +1.7711 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | weather | probe_unscaled_macro_f1 | 0.647616 | 0.661353 | +1.3737 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | scene | p10 | 0.616700 | 0.624415 | +0.7715 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | scene | macro_p10 | 0.571790 | 0.577353 | +0.5563 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | scene | probe_unscaled_accuracy | 0.755000 | 0.767040 | +1.2040 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | scene | probe_unscaled_balanced_accuracy | 0.637911 | 0.643897 | +0.5986 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | scene | probe_unscaled_macro_f1 | 0.665750 | 0.674159 | +0.8409 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | timeofday | p10 | 0.819300 | 0.827263 | +0.7963 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | timeofday | macro_p10 | 0.657749 | 0.663795 | +0.6046 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | timeofday | probe_unscaled_accuracy | 0.913000 | 0.922686 | +0.9686 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | timeofday | probe_unscaled_balanced_accuracy | 0.749446 | 0.759524 | +1.0078 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0 | timeofday | probe_unscaled_macro_f1 | 0.766903 | 0.777141 | +1.0238 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | weather | p10 | 0.572200 | 0.578637 | +0.6437 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | weather | macro_p10 | 0.370424 | 0.371181 | +0.0757 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | weather | probe_unscaled_accuracy | 0.782000 | 0.792472 | +1.0472 | -0.2000 | -0.2035 | -0.0035 |
| c1_scale_contraction_0.25 | weather | probe_unscaled_balanced_accuracy | 0.588814 | 0.606129 | +1.7315 | -1.8132 | -1.8528 | -0.0396 |
| c1_scale_contraction_0.25 | weather | probe_unscaled_macro_f1 | 0.626076 | 0.639411 | +1.3335 | -2.1540 | -2.1942 | -0.0402 |
| c1_scale_contraction_0.25 | scene | p10 | 0.616700 | 0.624415 | +0.7715 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | scene | macro_p10 | 0.571790 | 0.577353 | +0.5563 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | scene | probe_unscaled_accuracy | 0.767000 | 0.779247 | +1.2247 | +1.2000 | +1.2208 | +0.0208 |
| c1_scale_contraction_0.25 | scene | probe_unscaled_balanced_accuracy | 0.684375 | 0.692050 | +0.7676 | +4.6464 | +4.8153 | +0.1690 |
| c1_scale_contraction_0.25 | scene | probe_unscaled_macro_f1 | 0.710510 | 0.719365 | +0.8855 | +4.4761 | +4.5206 | +0.0445 |
| c1_scale_contraction_0.25 | timeofday | p10 | 0.819300 | 0.827263 | +0.7963 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | timeofday | macro_p10 | 0.657749 | 0.663795 | +0.6046 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.25 | timeofday | probe_unscaled_accuracy | 0.915000 | 0.924720 | +0.9720 | +0.2000 | +0.2035 | +0.0035 |
| c1_scale_contraction_0.25 | timeofday | probe_unscaled_balanced_accuracy | 0.747218 | 0.757155 | +0.9937 | -0.2229 | -0.2370 | -0.0141 |
| c1_scale_contraction_0.25 | timeofday | probe_unscaled_macro_f1 | 0.766589 | 0.776949 | +1.0360 | -0.0314 | -0.0192 | +0.0123 |
| c1_scale_contraction_0.5 | weather | p10 | 0.572200 | 0.578637 | +0.6437 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | weather | macro_p10 | 0.370424 | 0.371181 | +0.0757 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | weather | probe_unscaled_accuracy | 0.777000 | 0.787386 | +1.0386 | -0.7000 | -0.7121 | -0.0121 |
| c1_scale_contraction_0.5 | weather | probe_unscaled_balanced_accuracy | 0.616783 | 0.634862 | +1.8080 | +0.9837 | +1.0205 | +0.0368 |
| c1_scale_contraction_0.5 | weather | probe_unscaled_macro_f1 | 0.652448 | 0.666009 | +1.3560 | +0.4832 | +0.4656 | -0.0177 |
| c1_scale_contraction_0.5 | scene | p10 | 0.616700 | 0.624415 | +0.7715 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | scene | macro_p10 | 0.571790 | 0.577353 | +0.5563 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | scene | probe_unscaled_accuracy | 0.761000 | 0.773143 | +1.2143 | +0.6000 | +0.6104 | +0.0104 |
| c1_scale_contraction_0.5 | scene | probe_unscaled_balanced_accuracy | 0.662200 | 0.669032 | +0.6833 | +2.4289 | +2.5136 | +0.0847 |
| c1_scale_contraction_0.5 | scene | probe_unscaled_macro_f1 | 0.688883 | 0.697533 | +0.8650 | +2.3134 | +2.3374 | +0.0240 |
| c1_scale_contraction_0.5 | timeofday | p10 | 0.819300 | 0.827263 | +0.7963 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | timeofday | macro_p10 | 0.657749 | 0.663795 | +0.6046 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.5 | timeofday | probe_unscaled_accuracy | 0.922000 | 0.931841 | +0.9841 | +0.9000 | +0.9156 | +0.0156 |
| c1_scale_contraction_0.5 | timeofday | probe_unscaled_balanced_accuracy | 0.755669 | 0.765839 | +1.0170 | +0.6223 | +0.6315 | +0.0092 |
| c1_scale_contraction_0.5 | timeofday | probe_unscaled_macro_f1 | 0.781539 | 0.792647 | +1.1108 | +1.4636 | +1.5507 | +0.0870 |
| c1_scale_contraction_0.75 | weather | p10 | 0.572200 | 0.578637 | +0.6437 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | weather | macro_p10 | 0.370424 | 0.371181 | +0.0757 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | weather | probe_unscaled_accuracy | 0.782000 | 0.792472 | +1.0472 | -0.2000 | -0.2035 | -0.0035 |
| c1_scale_contraction_0.75 | weather | probe_unscaled_balanced_accuracy | 0.590943 | 0.608257 | +1.7313 | -1.6003 | -1.6401 | -0.0398 |
| c1_scale_contraction_0.75 | weather | probe_unscaled_macro_f1 | 0.629327 | 0.642664 | +1.3337 | -1.8289 | -1.8689 | -0.0400 |
| c1_scale_contraction_0.75 | scene | p10 | 0.616700 | 0.624415 | +0.7715 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | scene | macro_p10 | 0.571790 | 0.577353 | +0.5563 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | scene | probe_unscaled_accuracy | 0.746000 | 0.757884 | +1.1884 | -0.9000 | -0.9156 | -0.0156 |
| c1_scale_contraction_0.75 | scene | probe_unscaled_balanced_accuracy | 0.613358 | 0.618524 | +0.5165 | -2.4553 | -2.5373 | -0.0820 |
| c1_scale_contraction_0.75 | scene | probe_unscaled_macro_f1 | 0.635914 | 0.643844 | +0.7930 | -2.9835 | -3.0315 | -0.0479 |
| c1_scale_contraction_0.75 | timeofday | p10 | 0.819300 | 0.827263 | +0.7963 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | timeofday | macro_p10 | 0.657749 | 0.663795 | +0.6046 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.75 | timeofday | probe_unscaled_accuracy | 0.914000 | 0.923703 | +0.9703 | +0.1000 | +0.1017 | +0.0017 |
| c1_scale_contraction_0.75 | timeofday | probe_unscaled_balanced_accuracy | 0.746536 | 0.756463 | +0.9927 | -0.2910 | -0.3061 | -0.0151 |
| c1_scale_contraction_0.75 | timeofday | probe_unscaled_macro_f1 | 0.765133 | 0.775411 | +1.0278 | -0.1770 | -0.1730 | +0.0040 |
| c1_scale_contraction_0.9 | weather | p10 | 0.572200 | 0.578637 | +0.6437 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | weather | macro_p10 | 0.370424 | 0.371181 | +0.0757 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | weather | probe_unscaled_accuracy | 0.784000 | 0.794507 | +1.0507 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | weather | probe_unscaled_balanced_accuracy | 0.606946 | 0.624657 | +1.7711 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | weather | probe_unscaled_macro_f1 | 0.647616 | 0.661353 | +1.3737 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | scene | p10 | 0.616700 | 0.624415 | +0.7715 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | scene | macro_p10 | 0.571790 | 0.577353 | +0.5563 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | scene | probe_unscaled_accuracy | 0.755000 | 0.767040 | +1.2040 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | scene | probe_unscaled_balanced_accuracy | 0.637911 | 0.643897 | +0.5986 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | scene | probe_unscaled_macro_f1 | 0.665750 | 0.674159 | +0.8409 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | timeofday | p10 | 0.819300 | 0.827263 | +0.7963 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | timeofday | macro_p10 | 0.657749 | 0.663795 | +0.6046 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | timeofday | probe_unscaled_accuracy | 0.913000 | 0.922686 | +0.9686 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | timeofday | probe_unscaled_balanced_accuracy | 0.749446 | 0.759524 | +1.0078 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.9 | timeofday | probe_unscaled_macro_f1 | 0.766903 | 0.777141 | +1.0238 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | weather | p10 | 0.572200 | 0.578637 | +0.6437 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | weather | macro_p10 | 0.370424 | 0.371181 | +0.0757 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | weather | probe_unscaled_accuracy | 0.784000 | 0.794507 | +1.0507 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | weather | probe_unscaled_balanced_accuracy | 0.606946 | 0.624657 | +1.7711 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | weather | probe_unscaled_macro_f1 | 0.647616 | 0.661353 | +1.3737 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | scene | p10 | 0.616700 | 0.624415 | +0.7715 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | scene | macro_p10 | 0.571790 | 0.577353 | +0.5563 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | scene | probe_unscaled_accuracy | 0.755000 | 0.767040 | +1.2040 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | scene | probe_unscaled_balanced_accuracy | 0.637911 | 0.643897 | +0.5986 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | scene | probe_unscaled_macro_f1 | 0.665750 | 0.674159 | +0.8409 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | timeofday | p10 | 0.819300 | 0.827263 | +0.7963 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | timeofday | macro_p10 | 0.657749 | 0.663795 | +0.6046 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | timeofday | probe_unscaled_accuracy | 0.913000 | 0.922686 | +0.9686 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | timeofday | probe_unscaled_balanced_accuracy | 0.749446 | 0.759524 | +1.0078 | +0.0000 | +0.0000 | +0.0000 |
| c1_scale_contraction_0.99 | timeofday | probe_unscaled_macro_f1 | 0.766903 | 0.777141 | +1.0238 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | weather | p10 | 0.572200 | 0.578637 | +0.6437 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | weather | macro_p10 | 0.370424 | 0.371181 | +0.0757 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | weather | probe_unscaled_accuracy | 0.784000 | 0.794507 | +1.0507 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | weather | probe_unscaled_balanced_accuracy | 0.606946 | 0.624657 | +1.7711 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | weather | probe_unscaled_macro_f1 | 0.647616 | 0.661353 | +1.3737 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | scene | p10 | 0.616700 | 0.624415 | +0.7715 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | scene | macro_p10 | 0.571790 | 0.577353 | +0.5563 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | scene | probe_unscaled_accuracy | 0.755000 | 0.767040 | +1.2040 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | scene | probe_unscaled_balanced_accuracy | 0.637911 | 0.643897 | +0.5986 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | scene | probe_unscaled_macro_f1 | 0.665750 | 0.674159 | +0.8409 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | timeofday | p10 | 0.819300 | 0.827263 | +0.7963 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | timeofday | macro_p10 | 0.657749 | 0.663795 | +0.6046 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | timeofday | probe_unscaled_accuracy | 0.913000 | 0.922686 | +0.9686 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | timeofday | probe_unscaled_balanced_accuracy | 0.749446 | 0.759524 | +1.0078 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0 | timeofday | probe_unscaled_macro_f1 | 0.766903 | 0.777141 | +1.0238 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | weather | p10 | 0.572200 | 0.578637 | +0.6437 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | weather | macro_p10 | 0.370424 | 0.371181 | +0.0757 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | weather | probe_unscaled_accuracy | 0.784000 | 0.794507 | +1.0507 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | weather | probe_unscaled_balanced_accuracy | 0.606946 | 0.624657 | +1.7711 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | weather | probe_unscaled_macro_f1 | 0.647616 | 0.661353 | +1.3737 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | scene | p10 | 0.616700 | 0.624415 | +0.7715 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | scene | macro_p10 | 0.571790 | 0.577353 | +0.5563 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | scene | probe_unscaled_accuracy | 0.755000 | 0.767040 | +1.2040 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | scene | probe_unscaled_balanced_accuracy | 0.637911 | 0.643897 | +0.5986 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | scene | probe_unscaled_macro_f1 | 0.665750 | 0.674159 | +0.8409 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | timeofday | p10 | 0.819300 | 0.827263 | +0.7963 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | timeofday | macro_p10 | 0.657749 | 0.663795 | +0.6046 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | timeofday | probe_unscaled_accuracy | 0.913000 | 0.922686 | +0.9686 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | timeofday | probe_unscaled_balanced_accuracy | 0.749446 | 0.759524 | +1.0078 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.25 | timeofday | probe_unscaled_macro_f1 | 0.766903 | 0.777141 | +1.0238 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.5 | weather | p10 | 0.571400 | 0.577620 | +0.6220 | -0.0800 | -0.1017 | -0.0217 |
| c1_rank_truncation_0.5 | weather | macro_p10 | 0.369892 | 0.370123 | +0.0232 | -0.0532 | -0.1057 | -0.0525 |
| c1_rank_truncation_0.5 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.5 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.5 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.5 | weather | probe_standardized_accuracy | 0.769000 | 0.779247 | +1.0247 | -0.6000 | -0.6104 | -0.0104 |
| c1_rank_truncation_0.5 | weather | probe_standardized_balanced_accuracy | 0.608147 | 0.626060 | +1.7913 | -0.4170 | -0.4179 | -0.0009 |
| c1_rank_truncation_0.5 | weather | probe_standardized_macro_f1 | 0.641594 | 0.654945 | +1.3352 | -0.5420 | -0.6052 | -0.0631 |
| c1_rank_truncation_0.5 | weather | probe_unscaled_accuracy | 0.782000 | 0.792472 | +1.0472 | -0.2000 | -0.2035 | -0.0035 |
| c1_rank_truncation_0.5 | weather | probe_unscaled_balanced_accuracy | 0.603278 | 0.620881 | +1.7602 | -0.3667 | -0.3777 | -0.0109 |
| c1_rank_truncation_0.5 | weather | probe_unscaled_macro_f1 | 0.642663 | 0.656645 | +1.3982 | -0.4953 | -0.4708 | +0.0245 |
| c1_rank_truncation_0.5 | scene | p10 | 0.616800 | 0.624517 | +0.7717 | +0.0100 | +0.0102 | +0.0002 |
| c1_rank_truncation_0.5 | scene | macro_p10 | 0.570701 | 0.576222 | +0.5521 | -0.1089 | -0.1131 | -0.0042 |
| c1_rank_truncation_0.5 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.5 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.5 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.5 | scene | probe_standardized_accuracy | 0.765000 | 0.776195 | +1.1195 | +0.6000 | +0.7121 | +0.1121 |
| c1_rank_truncation_0.5 | scene | probe_standardized_balanced_accuracy | 0.695759 | 0.701756 | +0.5997 | +0.4291 | +0.6559 | +0.2269 |
| c1_rank_truncation_0.5 | scene | probe_standardized_macro_f1 | 0.712124 | 0.722919 | +1.0795 | +0.6803 | +0.9134 | +0.2331 |
| c1_rank_truncation_0.5 | scene | probe_unscaled_accuracy | 0.753000 | 0.765005 | +1.2005 | -0.2000 | -0.2035 | -0.0035 |
| c1_rank_truncation_0.5 | scene | probe_unscaled_balanced_accuracy | 0.635196 | 0.641110 | +0.5914 | -0.2714 | -0.2786 | -0.0072 |
| c1_rank_truncation_0.5 | scene | probe_unscaled_macro_f1 | 0.662077 | 0.670402 | +0.8326 | -0.3673 | -0.3757 | -0.0084 |
| c1_rank_truncation_0.5 | timeofday | p10 | 0.821300 | 0.829400 | +0.8100 | +0.2000 | +0.2136 | +0.0136 |
| c1_rank_truncation_0.5 | timeofday | macro_p10 | 0.659095 | 0.665604 | +0.6509 | +0.1346 | +0.1809 | +0.0464 |
| c1_rank_truncation_0.5 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.5 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.5 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.5 | timeofday | probe_standardized_accuracy | 0.921000 | 0.930824 | +0.9824 | +0.1000 | +0.1017 | +0.0017 |
| c1_rank_truncation_0.5 | timeofday | probe_standardized_balanced_accuracy | 0.758491 | 0.768810 | +1.0319 | +0.4274 | +0.4444 | +0.0171 |
| c1_rank_truncation_0.5 | timeofday | probe_standardized_macro_f1 | 0.782239 | 0.793192 | +1.0953 | +0.4818 | +0.4936 | +0.0118 |
| c1_rank_truncation_0.5 | timeofday | probe_unscaled_accuracy | 0.909000 | 0.918616 | +0.9616 | -0.4000 | -0.4069 | -0.0069 |
| c1_rank_truncation_0.5 | timeofday | probe_unscaled_balanced_accuracy | 0.742951 | 0.752823 | +0.9872 | -0.6495 | -0.6701 | -0.0206 |
| c1_rank_truncation_0.5 | timeofday | probe_unscaled_macro_f1 | 0.759470 | 0.769492 | +1.0022 | -0.7433 | -0.7648 | -0.0215 |
| c1_rank_truncation_0.75 | weather | p10 | 0.568200 | 0.574262 | +0.6062 | -0.4000 | -0.4374 | -0.0374 |
| c1_rank_truncation_0.75 | weather | macro_p10 | 0.367953 | 0.368192 | +0.0239 | -0.2471 | -0.2989 | -0.0518 |
| c1_rank_truncation_0.75 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.75 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.75 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.75 | weather | probe_standardized_accuracy | 0.764000 | 0.774161 | +1.0161 | -1.1000 | -1.1190 | -0.0190 |
| c1_rank_truncation_0.75 | weather | probe_standardized_balanced_accuracy | 0.595045 | 0.611319 | +1.6275 | -1.7272 | -1.8919 | -0.1647 |
| c1_rank_truncation_0.75 | weather | probe_standardized_macro_f1 | 0.625931 | 0.638864 | +1.2933 | -2.1083 | -2.2133 | -0.1050 |
| c1_rank_truncation_0.75 | weather | probe_unscaled_accuracy | 0.781000 | 0.790437 | +0.9437 | -0.3000 | -0.4069 | -0.1069 |
| c1_rank_truncation_0.75 | weather | probe_unscaled_balanced_accuracy | 0.597814 | 0.613753 | +1.5939 | -0.9132 | -1.0904 | -0.1772 |
| c1_rank_truncation_0.75 | weather | probe_unscaled_macro_f1 | 0.636415 | 0.648916 | +1.2502 | -1.1201 | -1.2436 | -0.1235 |
| c1_rank_truncation_0.75 | scene | p10 | 0.619900 | 0.627670 | +0.7770 | +0.3200 | +0.3255 | +0.0055 |
| c1_rank_truncation_0.75 | scene | macro_p10 | 0.569481 | 0.574788 | +0.5307 | -0.2309 | -0.2565 | -0.0256 |
| c1_rank_truncation_0.75 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.75 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.75 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.75 | scene | probe_standardized_accuracy | 0.753000 | 0.765005 | +1.2005 | -0.6000 | -0.4069 | +0.1931 |
| c1_rank_truncation_0.75 | scene | probe_standardized_balanced_accuracy | 0.677268 | 0.684998 | +0.7730 | -1.4200 | -1.0198 | +0.4002 |
| c1_rank_truncation_0.75 | scene | probe_standardized_macro_f1 | 0.696982 | 0.705486 | +0.8504 | -0.8339 | -0.8299 | +0.0040 |
| c1_rank_truncation_0.75 | scene | probe_unscaled_accuracy | 0.755000 | 0.767040 | +1.2040 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.75 | scene | probe_unscaled_balanced_accuracy | 0.636262 | 0.642138 | +0.5876 | -0.1649 | -0.1759 | -0.0110 |
| c1_rank_truncation_0.75 | scene | probe_unscaled_macro_f1 | 0.662512 | 0.670838 | +0.8326 | -0.3238 | -0.3321 | -0.0083 |
| c1_rank_truncation_0.75 | timeofday | p10 | 0.828500 | 0.836623 | +0.8123 | +0.9200 | +0.9359 | +0.0159 |
| c1_rank_truncation_0.75 | timeofday | macro_p10 | 0.660314 | 0.667412 | +0.7098 | +0.2565 | +0.3617 | +0.1052 |
| c1_rank_truncation_0.75 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.75 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.75 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.75 | timeofday | probe_standardized_accuracy | 0.919000 | 0.928789 | +0.9789 | -0.1000 | -0.1017 | -0.0017 |
| c1_rank_truncation_0.75 | timeofday | probe_standardized_balanced_accuracy | 0.753536 | 0.763674 | +1.0138 | -0.0682 | -0.0692 | -0.0010 |
| c1_rank_truncation_0.75 | timeofday | probe_standardized_macro_f1 | 0.776695 | 0.787519 | +1.0824 | -0.0726 | -0.0738 | -0.0011 |
| c1_rank_truncation_0.75 | timeofday | probe_unscaled_accuracy | 0.923000 | 0.932859 | +0.9859 | +1.0000 | +1.0173 | +0.0173 |
| c1_rank_truncation_0.75 | timeofday | probe_unscaled_balanced_accuracy | 0.752847 | 0.762869 | +1.0022 | +0.3401 | +0.3345 | -0.0057 |
| c1_rank_truncation_0.75 | timeofday | probe_unscaled_macro_f1 | 0.779923 | 0.791098 | +1.1175 | +1.3020 | +1.3957 | +0.0938 |
| c1_rank_truncation_0.9 | weather | p10 | 0.564700 | 0.570295 | +0.5595 | -0.7500 | -0.8342 | -0.0842 |
| c1_rank_truncation_0.9 | weather | macro_p10 | 0.362558 | 0.361789 | -0.0769 | -0.7865 | -0.9392 | -0.1526 |
| c1_rank_truncation_0.9 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.9 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.9 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.9 | weather | probe_standardized_accuracy | 0.770000 | 0.780264 | +1.0264 | -0.5000 | -0.5086 | -0.0086 |
| c1_rank_truncation_0.9 | weather | probe_standardized_balanced_accuracy | 0.590110 | 0.606250 | +1.6141 | -2.2207 | -2.3988 | -0.1781 |
| c1_rank_truncation_0.9 | weather | probe_standardized_macro_f1 | 0.622055 | 0.635070 | +1.3015 | -2.4959 | -2.5927 | -0.0968 |
| c1_rank_truncation_0.9 | weather | probe_unscaled_accuracy | 0.764000 | 0.774161 | +1.0161 | -2.0000 | -2.0346 | -0.0346 |
| c1_rank_truncation_0.9 | weather | probe_unscaled_balanced_accuracy | 0.562752 | 0.579296 | +1.6544 | -4.4194 | -4.5361 | -0.1167 |
| c1_rank_truncation_0.9 | weather | probe_unscaled_macro_f1 | 0.591952 | 0.605090 | +1.3138 | -5.5664 | -5.6263 | -0.0599 |
| c1_rank_truncation_0.9 | scene | p10 | 0.627300 | 0.635504 | +0.8204 | +1.0600 | +1.1089 | +0.0489 |
| c1_rank_truncation_0.9 | scene | macro_p10 | 0.573389 | 0.579037 | +0.5649 | +0.1598 | +0.1684 | +0.0086 |
| c1_rank_truncation_0.9 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.9 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.9 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.9 | scene | probe_standardized_accuracy | 0.761000 | 0.773143 | +1.2143 | +0.2000 | +0.4069 | +0.2069 |
| c1_rank_truncation_0.9 | scene | probe_standardized_balanced_accuracy | 0.677549 | 0.685173 | +0.7623 | -1.3919 | -1.0024 | +0.3895 |
| c1_rank_truncation_0.9 | scene | probe_standardized_macro_f1 | 0.701730 | 0.710364 | +0.8634 | -0.3592 | -0.3421 | +0.0170 |
| c1_rank_truncation_0.9 | scene | probe_unscaled_accuracy | 0.753000 | 0.765005 | +1.2005 | -0.2000 | -0.2035 | -0.0035 |
| c1_rank_truncation_0.9 | scene | probe_unscaled_balanced_accuracy | 0.634462 | 0.640361 | +0.5899 | -0.3449 | -0.3536 | -0.0086 |
| c1_rank_truncation_0.9 | scene | probe_unscaled_macro_f1 | 0.662133 | 0.670513 | +0.8380 | -0.3616 | -0.3646 | -0.0029 |
| c1_rank_truncation_0.9 | timeofday | p10 | 0.837600 | 0.845778 | +0.8178 | +1.8300 | +1.8515 | +0.0215 |
| c1_rank_truncation_0.9 | timeofday | macro_p10 | 0.662857 | 0.669451 | +0.6594 | +0.5108 | +0.5656 | +0.0548 |
| c1_rank_truncation_0.9 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.9 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.9 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.9 | timeofday | probe_standardized_accuracy | 0.928000 | 0.936928 | +0.8928 | +0.8000 | +0.7121 | -0.0879 |
| c1_rank_truncation_0.9 | timeofday | probe_standardized_balanced_accuracy | 0.777718 | 0.788062 | +1.0343 | +2.3501 | +2.3696 | +0.0196 |
| c1_rank_truncation_0.9 | timeofday | probe_standardized_macro_f1 | 0.806781 | 0.816692 | +0.9910 | +2.9361 | +2.8435 | -0.0925 |
| c1_rank_truncation_0.9 | timeofday | probe_unscaled_accuracy | 0.928000 | 0.936928 | +0.8928 | +1.5000 | +1.4242 | -0.0758 |
| c1_rank_truncation_0.9 | timeofday | probe_unscaled_balanced_accuracy | 0.767031 | 0.776894 | +0.9863 | +1.7585 | +1.7369 | -0.0215 |
| c1_rank_truncation_0.9 | timeofday | probe_unscaled_macro_f1 | 0.798107 | 0.808004 | +0.9897 | +3.1204 | +3.0863 | -0.0341 |
| c1_rank_truncation_0.99 | weather | p10 | 0.521200 | 0.527263 | +0.6063 | -5.1000 | -5.1373 | -0.0373 |
| c1_rank_truncation_0.99 | weather | macro_p10 | 0.314003 | 0.312698 | -0.1305 | -5.6421 | -5.8483 | -0.2062 |
| c1_rank_truncation_0.99 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.99 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.99 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.99 | weather | probe_standardized_accuracy | 0.604000 | 0.610376 | +0.6376 | -17.1000 | -17.4975 | -0.3975 |
| c1_rank_truncation_0.99 | weather | probe_standardized_balanced_accuracy | 0.309968 | 0.307544 | -0.2423 | -30.2350 | -32.2694 | -2.0345 |
| c1_rank_truncation_0.99 | weather | probe_standardized_macro_f1 | 0.311118 | 0.307378 | -0.3740 | -33.5896 | -35.3619 | -1.7723 |
| c1_rank_truncation_0.99 | weather | probe_unscaled_accuracy | 0.604000 | 0.610376 | +0.6376 | -18.0000 | -18.4130 | -0.4130 |
| c1_rank_truncation_0.99 | weather | probe_unscaled_balanced_accuracy | 0.309968 | 0.307544 | -0.2423 | -29.6978 | -31.7113 | -2.0135 |
| c1_rank_truncation_0.99 | weather | probe_unscaled_macro_f1 | 0.311317 | 0.307579 | -0.3737 | -33.6299 | -35.3773 | -1.7474 |
| c1_rank_truncation_0.99 | scene | p10 | 0.652500 | 0.661750 | +0.9250 | +3.5800 | +3.7335 | +0.1535 |
| c1_rank_truncation_0.99 | scene | macro_p10 | 0.596027 | 0.601661 | +0.5633 | +2.4237 | +2.4308 | +0.0071 |
| c1_rank_truncation_0.99 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.99 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.99 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.99 | scene | probe_standardized_accuracy | 0.763000 | 0.775178 | +1.2178 | +0.4000 | +0.6104 | +0.2104 |
| c1_rank_truncation_0.99 | scene | probe_standardized_balanced_accuracy | 0.680505 | 0.688107 | +0.7602 | -1.0963 | -0.7090 | +0.3873 |
| c1_rank_truncation_0.99 | scene | probe_standardized_macro_f1 | 0.705625 | 0.714400 | +0.8775 | +0.0304 | +0.0614 | +0.0310 |
| c1_rank_truncation_0.99 | scene | probe_unscaled_accuracy | 0.763000 | 0.775178 | +1.2178 | +0.8000 | +0.8138 | +0.0138 |
| c1_rank_truncation_0.99 | scene | probe_unscaled_balanced_accuracy | 0.680505 | 0.688107 | +0.7602 | +4.2594 | +4.4210 | +0.1616 |
| c1_rank_truncation_0.99 | scene | probe_unscaled_macro_f1 | 0.705625 | 0.714400 | +0.8775 | +3.9875 | +4.0241 | +0.0365 |
| c1_rank_truncation_0.99 | timeofday | p10 | 0.845500 | 0.855036 | +0.9536 | +2.6200 | +2.7772 | +0.1572 |
| c1_rank_truncation_0.99 | timeofday | macro_p10 | 0.652994 | 0.657982 | +0.4989 | -0.4756 | -0.5813 | -0.1057 |
| c1_rank_truncation_0.99 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.99 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.99 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_rank_truncation_0.99 | timeofday | probe_standardized_accuracy | 0.915000 | 0.923703 | +0.8703 | -0.5000 | -0.6104 | -0.1104 |
| c1_rank_truncation_0.99 | timeofday | probe_standardized_balanced_accuracy | 0.697108 | 0.704105 | +0.6996 | -5.7110 | -6.0261 | -0.3151 |
| c1_rank_truncation_0.99 | timeofday | probe_standardized_macro_f1 | 0.706268 | 0.714054 | +0.7786 | -7.1153 | -7.4203 | -0.3049 |
| c1_rank_truncation_0.99 | timeofday | probe_unscaled_accuracy | 0.915000 | 0.923703 | +0.8703 | +0.2000 | +0.1017 | -0.0983 |
| c1_rank_truncation_0.99 | timeofday | probe_unscaled_balanced_accuracy | 0.697108 | 0.704105 | +0.6996 | -5.2338 | -5.5420 | -0.3082 |
| c1_rank_truncation_0.99 | timeofday | probe_unscaled_macro_f1 | 0.706268 | 0.714054 | +0.7786 | -6.0635 | -6.3087 | -0.2452 |
| c1_mean_injection_0 | weather | p10 | 0.572200 | 0.578637 | +0.6437 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | weather | macro_p10 | 0.370424 | 0.371181 | +0.0757 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | weather | probe_unscaled_accuracy | 0.784000 | 0.794507 | +1.0507 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | weather | probe_unscaled_balanced_accuracy | 0.606946 | 0.624657 | +1.7711 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | weather | probe_unscaled_macro_f1 | 0.647616 | 0.661353 | +1.3737 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | scene | p10 | 0.616700 | 0.624415 | +0.7715 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | scene | macro_p10 | 0.571790 | 0.577353 | +0.5563 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | scene | probe_unscaled_accuracy | 0.755000 | 0.767040 | +1.2040 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | scene | probe_unscaled_balanced_accuracy | 0.637911 | 0.643897 | +0.5986 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | scene | probe_unscaled_macro_f1 | 0.665750 | 0.674159 | +0.8409 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | timeofday | p10 | 0.819300 | 0.827263 | +0.7963 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | timeofday | macro_p10 | 0.657749 | 0.663795 | +0.6046 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | timeofday | probe_unscaled_accuracy | 0.913000 | 0.922686 | +0.9686 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | timeofday | probe_unscaled_balanced_accuracy | 0.749446 | 0.759524 | +1.0078 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0 | timeofday | probe_unscaled_macro_f1 | 0.766903 | 0.777141 | +1.0238 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | weather | p10 | 0.572900 | 0.579349 | +0.6449 | +0.0700 | +0.0712 | +0.0012 |
| c1_mean_injection_0.25 | weather | macro_p10 | 0.369244 | 0.369782 | +0.0538 | -0.1180 | -0.1399 | -0.0219 |
| c1_mean_injection_0.25 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | weather | probe_unscaled_accuracy | 0.784000 | 0.794507 | +1.0507 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | weather | probe_unscaled_balanced_accuracy | 0.606946 | 0.624657 | +1.7711 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | weather | probe_unscaled_macro_f1 | 0.647616 | 0.661353 | +1.3737 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | scene | p10 | 0.618000 | 0.625738 | +0.7738 | +0.1300 | +0.1322 | +0.0022 |
| c1_mean_injection_0.25 | scene | macro_p10 | 0.573026 | 0.578582 | +0.5555 | +0.1236 | +0.1229 | -0.0007 |
| c1_mean_injection_0.25 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | scene | probe_unscaled_accuracy | 0.755000 | 0.767040 | +1.2040 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | scene | probe_unscaled_balanced_accuracy | 0.637911 | 0.643897 | +0.5986 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | scene | probe_unscaled_macro_f1 | 0.665750 | 0.674159 | +0.8409 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | timeofday | p10 | 0.819600 | 0.827569 | +0.7969 | +0.0300 | +0.0305 | +0.0005 |
| c1_mean_injection_0.25 | timeofday | macro_p10 | 0.660100 | 0.666245 | +0.6145 | +0.2351 | +0.2450 | +0.0099 |
| c1_mean_injection_0.25 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | timeofday | probe_unscaled_accuracy | 0.913000 | 0.922686 | +0.9686 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | timeofday | probe_unscaled_balanced_accuracy | 0.749446 | 0.759524 | +1.0078 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.25 | timeofday | probe_unscaled_macro_f1 | 0.766903 | 0.777141 | +1.0238 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | weather | p10 | 0.571700 | 0.578128 | +0.6428 | -0.0500 | -0.0509 | -0.0009 |
| c1_mean_injection_0.5 | weather | macro_p10 | 0.366912 | 0.367263 | +0.0350 | -0.3512 | -0.3918 | -0.0407 |
| c1_mean_injection_0.5 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | weather | probe_unscaled_accuracy | 0.784000 | 0.794507 | +1.0507 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | weather | probe_unscaled_balanced_accuracy | 0.606946 | 0.624657 | +1.7711 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | weather | probe_unscaled_macro_f1 | 0.647616 | 0.661353 | +1.3737 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | scene | p10 | 0.619100 | 0.626857 | +0.7757 | +0.2400 | +0.2442 | +0.0042 |
| c1_mean_injection_0.5 | scene | macro_p10 | 0.573341 | 0.578880 | +0.5539 | +0.1551 | +0.1527 | -0.0024 |
| c1_mean_injection_0.5 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | scene | probe_unscaled_accuracy | 0.755000 | 0.767040 | +1.2040 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | scene | probe_unscaled_balanced_accuracy | 0.637911 | 0.643897 | +0.5986 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | scene | probe_unscaled_macro_f1 | 0.665750 | 0.674159 | +0.8409 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | timeofday | p10 | 0.818800 | 0.826755 | +0.7955 | -0.0500 | -0.0509 | -0.0009 |
| c1_mean_injection_0.5 | timeofday | macro_p10 | 0.659861 | 0.666013 | +0.6151 | +0.2112 | +0.2218 | +0.0106 |
| c1_mean_injection_0.5 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | timeofday | probe_unscaled_accuracy | 0.913000 | 0.922686 | +0.9686 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | timeofday | probe_unscaled_balanced_accuracy | 0.749446 | 0.759524 | +1.0078 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.5 | timeofday | probe_unscaled_macro_f1 | 0.766903 | 0.777141 | +1.0238 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | weather | p10 | 0.573100 | 0.579552 | +0.6452 | +0.0900 | +0.0916 | +0.0016 |
| c1_mean_injection_0.75 | weather | macro_p10 | 0.369406 | 0.369932 | +0.0526 | -0.1018 | -0.1249 | -0.0232 |
| c1_mean_injection_0.75 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | weather | probe_unscaled_accuracy | 0.784000 | 0.794507 | +1.0507 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | weather | probe_unscaled_balanced_accuracy | 0.606946 | 0.624657 | +1.7711 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | weather | probe_unscaled_macro_f1 | 0.647616 | 0.661353 | +1.3737 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | scene | p10 | 0.618600 | 0.626348 | +0.7748 | +0.1900 | +0.1933 | +0.0033 |
| c1_mean_injection_0.75 | scene | macro_p10 | 0.573300 | 0.578854 | +0.5555 | +0.1509 | +0.1501 | -0.0008 |
| c1_mean_injection_0.75 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | scene | probe_unscaled_accuracy | 0.755000 | 0.767040 | +1.2040 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | scene | probe_unscaled_balanced_accuracy | 0.637911 | 0.643897 | +0.5986 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | scene | probe_unscaled_macro_f1 | 0.665750 | 0.674159 | +0.8409 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | timeofday | p10 | 0.820100 | 0.828077 | +0.7977 | +0.0800 | +0.0814 | +0.0014 |
| c1_mean_injection_0.75 | timeofday | macro_p10 | 0.660064 | 0.666197 | +0.6133 | +0.2315 | +0.2402 | +0.0088 |
| c1_mean_injection_0.75 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | timeofday | probe_unscaled_accuracy | 0.913000 | 0.922686 | +0.9686 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | timeofday | probe_unscaled_balanced_accuracy | 0.749446 | 0.759524 | +1.0078 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.75 | timeofday | probe_unscaled_macro_f1 | 0.766903 | 0.777141 | +1.0238 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | weather | p10 | 0.571400 | 0.577823 | +0.6423 | -0.0800 | -0.0814 | -0.0014 |
| c1_mean_injection_0.9 | weather | macro_p10 | 0.369008 | 0.369646 | +0.0638 | -0.1416 | -0.1535 | -0.0119 |
| c1_mean_injection_0.9 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | weather | probe_unscaled_accuracy | 0.784000 | 0.794507 | +1.0507 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | weather | probe_unscaled_balanced_accuracy | 0.606946 | 0.624657 | +1.7711 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | weather | probe_unscaled_macro_f1 | 0.647616 | 0.661353 | +1.3737 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | scene | p10 | 0.618300 | 0.626043 | +0.7743 | +0.1600 | +0.1628 | +0.0028 |
| c1_mean_injection_0.9 | scene | macro_p10 | 0.572991 | 0.578553 | +0.5562 | +0.1200 | +0.1200 | -0.0001 |
| c1_mean_injection_0.9 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | scene | probe_unscaled_accuracy | 0.755000 | 0.767040 | +1.2040 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | scene | probe_unscaled_balanced_accuracy | 0.637911 | 0.643897 | +0.5986 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | scene | probe_unscaled_macro_f1 | 0.665750 | 0.674159 | +0.8409 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | timeofday | p10 | 0.819400 | 0.827365 | +0.7965 | +0.0100 | +0.0102 | +0.0002 |
| c1_mean_injection_0.9 | timeofday | macro_p10 | 0.658509 | 0.664587 | +0.6078 | +0.0760 | +0.0792 | +0.0032 |
| c1_mean_injection_0.9 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | timeofday | probe_unscaled_accuracy | 0.913000 | 0.922686 | +0.9686 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | timeofday | probe_unscaled_balanced_accuracy | 0.749446 | 0.759524 | +1.0078 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.9 | timeofday | probe_unscaled_macro_f1 | 0.766903 | 0.777141 | +1.0238 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | weather | p10 | 0.571500 | 0.577925 | +0.6425 | -0.0700 | -0.0712 | -0.0012 |
| c1_mean_injection_0.99 | weather | macro_p10 | 0.369577 | 0.370320 | +0.0744 | -0.0847 | -0.0860 | -0.0013 |
| c1_mean_injection_0.99 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | weather | probe_unscaled_accuracy | 0.785000 | 0.795524 | +1.0524 | +0.1000 | +0.1017 | +0.0017 |
| c1_mean_injection_0.99 | weather | probe_unscaled_balanced_accuracy | 0.608262 | 0.625991 | +1.7729 | +0.1316 | +0.1333 | +0.0018 |
| c1_mean_injection_0.99 | weather | probe_unscaled_macro_f1 | 0.648942 | 0.662683 | +1.3741 | +0.1326 | +0.1330 | +0.0004 |
| c1_mean_injection_0.99 | scene | p10 | 0.617700 | 0.625432 | +0.7732 | +0.1000 | +0.1017 | +0.0017 |
| c1_mean_injection_0.99 | scene | macro_p10 | 0.572441 | 0.578007 | +0.5565 | +0.0651 | +0.0654 | +0.0002 |
| c1_mean_injection_0.99 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | scene | probe_unscaled_accuracy | 0.755000 | 0.767040 | +1.2040 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | scene | probe_unscaled_balanced_accuracy | 0.637911 | 0.643897 | +0.5986 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | scene | probe_unscaled_macro_f1 | 0.665750 | 0.674159 | +0.8409 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | timeofday | p10 | 0.818600 | 0.826551 | +0.7951 | -0.0700 | -0.0712 | -0.0012 |
| c1_mean_injection_0.99 | timeofday | macro_p10 | 0.657955 | 0.664025 | +0.6070 | +0.0206 | +0.0230 | +0.0024 |
| c1_mean_injection_0.99 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | timeofday | probe_unscaled_accuracy | 0.913000 | 0.922686 | +0.9686 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | timeofday | probe_unscaled_balanced_accuracy | 0.749446 | 0.759524 | +1.0078 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_injection_0.99 | timeofday | probe_unscaled_macro_f1 | 0.766903 | 0.777141 | +1.0238 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | weather | p10 | 0.572200 | 0.578637 | +0.6437 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | weather | macro_p10 | 0.370424 | 0.371181 | +0.0757 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | weather | probe_unscaled_accuracy | 0.784000 | 0.794507 | +1.0507 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | weather | probe_unscaled_balanced_accuracy | 0.606946 | 0.624657 | +1.7711 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | weather | probe_unscaled_macro_f1 | 0.647616 | 0.661353 | +1.3737 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | scene | p10 | 0.616700 | 0.624415 | +0.7715 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | scene | macro_p10 | 0.571790 | 0.577353 | +0.5563 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | scene | probe_unscaled_accuracy | 0.755000 | 0.767040 | +1.2040 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | scene | probe_unscaled_balanced_accuracy | 0.637911 | 0.643897 | +0.5986 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | scene | probe_unscaled_macro_f1 | 0.665750 | 0.674159 | +0.8409 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | timeofday | p10 | 0.819300 | 0.827263 | +0.7963 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | timeofday | macro_p10 | 0.657749 | 0.663795 | +0.6046 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | timeofday | probe_unscaled_accuracy | 0.913000 | 0.922686 | +0.9686 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | timeofday | probe_unscaled_balanced_accuracy | 0.749446 | 0.759524 | +1.0078 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0 | timeofday | probe_unscaled_macro_f1 | 0.766903 | 0.777141 | +1.0238 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.25 | weather | p10 | 0.565900 | 0.572126 | +0.6226 | -0.6300 | -0.6511 | -0.0211 |
| c1_isotropic_noise_0.25 | weather | macro_p10 | 0.360672 | 0.361154 | +0.0482 | -0.9752 | -1.0027 | -0.0275 |
| c1_isotropic_noise_0.25 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.25 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.25 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.25 | weather | probe_standardized_accuracy | 0.759000 | 0.770092 | +1.1092 | -1.6000 | -1.5259 | +0.0741 |
| c1_isotropic_noise_0.25 | weather | probe_standardized_balanced_accuracy | 0.583645 | 0.599945 | +1.6300 | -2.8672 | -3.0294 | -0.1622 |
| c1_isotropic_noise_0.25 | weather | probe_standardized_macro_f1 | 0.616811 | 0.631905 | +1.5094 | -3.0203 | -2.9092 | +0.1111 |
| c1_isotropic_noise_0.25 | weather | probe_unscaled_accuracy | 0.762000 | 0.774161 | +1.2161 | -2.2000 | -2.0346 | +0.1654 |
| c1_isotropic_noise_0.25 | weather | probe_unscaled_balanced_accuracy | 0.567509 | 0.584630 | +1.7121 | -3.9436 | -4.0027 | -0.0591 |
| c1_isotropic_noise_0.25 | weather | probe_unscaled_macro_f1 | 0.601166 | 0.616578 | +1.5413 | -4.6450 | -4.4774 | +0.1676 |
| c1_isotropic_noise_0.25 | scene | p10 | 0.608200 | 0.615666 | +0.7466 | -0.8500 | -0.8749 | -0.0249 |
| c1_isotropic_noise_0.25 | scene | macro_p10 | 0.558788 | 0.563981 | +0.5193 | -1.3002 | -1.3372 | -0.0370 |
| c1_isotropic_noise_0.25 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.25 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.25 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.25 | scene | probe_standardized_accuracy | 0.742000 | 0.753815 | +1.1815 | -1.7000 | -1.5259 | +0.1741 |
| c1_isotropic_noise_0.25 | scene | probe_standardized_balanced_accuracy | 0.671491 | 0.679211 | +0.7720 | -1.9977 | -1.5986 | +0.3991 |
| c1_isotropic_noise_0.25 | scene | probe_standardized_macro_f1 | 0.685885 | 0.695013 | +0.9128 | -1.9436 | -1.8773 | +0.0663 |
| c1_isotropic_noise_0.25 | scene | probe_unscaled_accuracy | 0.752000 | 0.763988 | +1.1988 | -0.3000 | -0.3052 | -0.0052 |
| c1_isotropic_noise_0.25 | scene | probe_unscaled_balanced_accuracy | 0.629117 | 0.634766 | +0.5649 | -0.8794 | -0.9131 | -0.0337 |
| c1_isotropic_noise_0.25 | scene | probe_unscaled_macro_f1 | 0.653647 | 0.661816 | +0.8170 | -1.2103 | -1.2343 | -0.0240 |
| c1_isotropic_noise_0.25 | timeofday | p10 | 0.815200 | 0.823093 | +0.7893 | -0.4100 | -0.4171 | -0.0071 |
| c1_isotropic_noise_0.25 | timeofday | macro_p10 | 0.650539 | 0.655972 | +0.5433 | -0.7211 | -0.7823 | -0.0612 |
| c1_isotropic_noise_0.25 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.25 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.25 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.25 | timeofday | probe_standardized_accuracy | 0.920000 | 0.928789 | +0.8789 | +0.0000 | -0.1017 | -0.1017 |
| c1_isotropic_noise_0.25 | timeofday | probe_standardized_balanced_accuracy | 0.743530 | 0.752506 | +0.8976 | -1.0687 | -1.1859 | -0.1172 |
| c1_isotropic_noise_0.25 | timeofday | probe_standardized_macro_f1 | 0.767972 | 0.777130 | +0.9157 | -0.9448 | -1.1126 | -0.1678 |
| c1_isotropic_noise_0.25 | timeofday | probe_unscaled_accuracy | 0.924000 | 0.932859 | +0.8859 | +1.1000 | +1.0173 | -0.0827 |
| c1_isotropic_noise_0.25 | timeofday | probe_unscaled_balanced_accuracy | 0.749849 | 0.759025 | +0.9176 | +0.0403 | -0.0499 | -0.0902 |
| c1_isotropic_noise_0.25 | timeofday | probe_unscaled_macro_f1 | 0.777899 | 0.787406 | +0.9507 | +1.0996 | +1.0265 | -0.0731 |
| c1_isotropic_noise_0.5 | weather | p10 | 0.506100 | 0.510682 | +0.4582 | -6.6100 | -6.7955 | -0.1855 |
| c1_isotropic_noise_0.5 | weather | macro_p10 | 0.289874 | 0.287042 | -0.2832 | -8.0550 | -8.4139 | -0.3589 |
| c1_isotropic_noise_0.5 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.5 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.5 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.5 | weather | probe_standardized_accuracy | 0.641000 | 0.650051 | +0.9051 | -13.4000 | -13.5300 | -0.1300 |
| c1_isotropic_noise_0.5 | weather | probe_standardized_balanced_accuracy | 0.409164 | 0.419309 | +1.0145 | -20.3153 | -21.0930 | -0.7777 |
| c1_isotropic_noise_0.5 | weather | probe_standardized_macro_f1 | 0.432937 | 0.442934 | +0.9997 | -21.4077 | -21.8063 | -0.3986 |
| c1_isotropic_noise_0.5 | weather | probe_unscaled_accuracy | 0.664000 | 0.672431 | +0.8431 | -12.0000 | -12.2075 | -0.2075 |
| c1_isotropic_noise_0.5 | weather | probe_unscaled_balanced_accuracy | 0.374434 | 0.382261 | +0.7827 | -23.2511 | -24.2396 | -0.9885 |
| c1_isotropic_noise_0.5 | weather | probe_unscaled_macro_f1 | 0.400732 | 0.410074 | +0.9342 | -24.6884 | -25.1279 | -0.4394 |
| c1_isotropic_noise_0.5 | scene | p10 | 0.525500 | 0.531740 | +0.6240 | -9.1200 | -9.2675 | -0.1475 |
| c1_isotropic_noise_0.5 | scene | macro_p10 | 0.462169 | 0.465110 | +0.2941 | -10.9621 | -11.2243 | -0.2622 |
| c1_isotropic_noise_0.5 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.5 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.5 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.5 | scene | probe_standardized_accuracy | 0.699000 | 0.709054 | +1.0054 | -6.0000 | -6.0020 | -0.0020 |
| c1_isotropic_noise_0.5 | scene | probe_standardized_balanced_accuracy | 0.601421 | 0.606056 | +0.4635 | -9.0047 | -8.9141 | +0.0906 |
| c1_isotropic_noise_0.5 | scene | probe_standardized_macro_f1 | 0.615995 | 0.623721 | +0.7726 | -8.9326 | -9.0064 | -0.0738 |
| c1_isotropic_noise_0.5 | scene | probe_unscaled_accuracy | 0.720000 | 0.730417 | +1.0417 | -3.5000 | -3.6623 | -0.1623 |
| c1_isotropic_noise_0.5 | scene | probe_unscaled_balanced_accuracy | 0.584845 | 0.588271 | +0.3426 | -5.3066 | -5.5626 | -0.2560 |
| c1_isotropic_noise_0.5 | scene | probe_unscaled_macro_f1 | 0.602716 | 0.609541 | +0.6825 | -6.3034 | -6.4618 | -0.1584 |
| c1_isotropic_noise_0.5 | timeofday | p10 | 0.726600 | 0.733367 | +0.6767 | -9.2700 | -9.3896 | -0.1196 |
| c1_isotropic_noise_0.5 | timeofday | macro_p10 | 0.569350 | 0.573682 | +0.4332 | -8.8400 | -9.0113 | -0.1714 |
| c1_isotropic_noise_0.5 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.5 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.5 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.5 | timeofday | probe_standardized_accuracy | 0.906000 | 0.915565 | +0.9565 | -1.4000 | -1.4242 | -0.0242 |
| c1_isotropic_noise_0.5 | timeofday | probe_standardized_balanced_accuracy | 0.708403 | 0.716791 | +0.8388 | -4.5814 | -4.7574 | -0.1760 |
| c1_isotropic_noise_0.5 | timeofday | probe_standardized_macro_f1 | 0.723715 | 0.732605 | +0.8890 | -5.3706 | -5.5651 | -0.1945 |
| c1_isotropic_noise_0.5 | timeofday | probe_unscaled_accuracy | 0.897000 | 0.906409 | +0.9409 | -1.6000 | -1.6277 | -0.0277 |
| c1_isotropic_noise_0.5 | timeofday | probe_unscaled_balanced_accuracy | 0.648126 | 0.653910 | +0.5784 | -10.1320 | -10.5614 | -0.4295 |
| c1_isotropic_noise_0.5 | timeofday | probe_unscaled_macro_f1 | 0.623088 | 0.629004 | +0.5916 | -14.3815 | -14.8137 | -0.4321 |
| c1_isotropic_noise_0.75 | weather | p10 | 0.420000 | 0.425941 | +0.5941 | -15.2200 | -15.2696 | -0.0496 |
| c1_isotropic_noise_0.75 | weather | macro_p10 | 0.215012 | 0.214781 | -0.0231 | -15.5411 | -15.6400 | -0.0988 |
| c1_isotropic_noise_0.75 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.75 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.75 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.75 | weather | probe_standardized_accuracy | 0.515000 | 0.520855 | +0.5855 | -26.0000 | -26.4496 | -0.4496 |
| c1_isotropic_noise_0.75 | weather | probe_standardized_balanced_accuracy | 0.250305 | 0.252981 | +0.2676 | -36.2012 | -37.7258 | -1.5245 |
| c1_isotropic_noise_0.75 | weather | probe_standardized_macro_f1 | 0.249161 | 0.252788 | +0.3627 | -39.7853 | -40.8209 | -1.0356 |
| c1_isotropic_noise_0.75 | weather | probe_unscaled_accuracy | 0.588000 | 0.595117 | +0.7117 | -19.6000 | -19.9390 | -0.3390 |
| c1_isotropic_noise_0.75 | weather | probe_unscaled_balanced_accuracy | 0.200000 | 0.200000 | +0.0000 | -40.6946 | -42.4657 | -1.7711 |
| c1_isotropic_noise_0.75 | weather | probe_unscaled_macro_f1 | 0.148111 | 0.149235 | +0.1124 | -49.9505 | -51.2118 | -1.2613 |
| c1_isotropic_noise_0.75 | scene | p10 | 0.426400 | 0.435707 | +0.9307 | -19.0300 | -18.8708 | +0.1592 |
| c1_isotropic_noise_0.75 | scene | macro_p10 | 0.351457 | 0.353961 | +0.2504 | -22.0333 | -22.3392 | -0.3059 |
| c1_isotropic_noise_0.75 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.75 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.75 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.75 | scene | probe_standardized_accuracy | 0.581000 | 0.590031 | +0.9031 | -17.8000 | -17.9044 | -0.1044 |
| c1_isotropic_noise_0.75 | scene | probe_standardized_balanced_accuracy | 0.455806 | 0.458272 | +0.2467 | -23.5662 | -23.6924 | -0.1262 |
| c1_isotropic_noise_0.75 | scene | probe_standardized_macro_f1 | 0.454138 | 0.459399 | +0.5261 | -25.1183 | -25.4386 | -0.3203 |
| c1_isotropic_noise_0.75 | scene | probe_unscaled_accuracy | 0.615000 | 0.624619 | +0.9619 | -14.0000 | -14.2421 | -0.2421 |
| c1_isotropic_noise_0.75 | scene | probe_unscaled_balanced_accuracy | 0.421672 | 0.423153 | +0.1481 | -21.6238 | -22.0744 | -0.4505 |
| c1_isotropic_noise_0.75 | scene | probe_unscaled_macro_f1 | 0.398821 | 0.403311 | +0.4490 | -26.6929 | -27.0848 | -0.3920 |
| c1_isotropic_noise_0.75 | timeofday | p10 | 0.516000 | 0.519736 | +0.3736 | -30.3300 | -30.7528 | -0.4228 |
| c1_isotropic_noise_0.75 | timeofday | macro_p10 | 0.398315 | 0.399995 | +0.1680 | -25.9434 | -26.3800 | -0.4365 |
| c1_isotropic_noise_0.75 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.75 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.75 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.75 | timeofday | probe_standardized_accuracy | 0.763000 | 0.769074 | +0.6074 | -15.7000 | -16.0732 | -0.3732 |
| c1_isotropic_noise_0.75 | timeofday | probe_standardized_balanced_accuracy | 0.566038 | 0.570106 | +0.4068 | -18.8180 | -19.4259 | -0.6079 |
| c1_isotropic_noise_0.75 | timeofday | probe_standardized_macro_f1 | 0.559520 | 0.564135 | +0.4615 | -21.7901 | -22.4121 | -0.6220 |
| c1_isotropic_noise_0.75 | timeofday | probe_unscaled_accuracy | 0.784000 | 0.789420 | +0.5420 | -12.9000 | -13.3266 | -0.4266 |
| c1_isotropic_noise_0.75 | timeofday | probe_unscaled_balanced_accuracy | 0.564398 | 0.567198 | +0.2800 | -18.5048 | -19.2326 | -0.7278 |
| c1_isotropic_noise_0.75 | timeofday | probe_unscaled_macro_f1 | 0.543548 | 0.546759 | +0.3211 | -22.3355 | -23.0382 | -0.7027 |
| c1_isotropic_noise_0.9 | weather | p10 | 0.398400 | 0.403662 | +0.5262 | -17.3800 | -17.4975 | -0.1175 |
| c1_isotropic_noise_0.9 | weather | macro_p10 | 0.205084 | 0.204127 | -0.0957 | -16.5340 | -16.7054 | -0.1714 |
| c1_isotropic_noise_0.9 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.9 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.9 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.9 | weather | probe_standardized_accuracy | 0.469000 | 0.474059 | +0.5059 | -30.6000 | -31.1292 | -0.5292 |
| c1_isotropic_noise_0.9 | weather | probe_standardized_balanced_accuracy | 0.210021 | 0.211973 | +0.1952 | -40.2296 | -41.8265 | -1.5969 |
| c1_isotropic_noise_0.9 | weather | probe_standardized_macro_f1 | 0.206878 | 0.209706 | +0.2829 | -44.0136 | -45.1291 | -1.1155 |
| c1_isotropic_noise_0.9 | weather | probe_unscaled_accuracy | 0.580000 | 0.586979 | +0.6979 | -20.4000 | -20.7528 | -0.3528 |
| c1_isotropic_noise_0.9 | weather | probe_unscaled_balanced_accuracy | 0.199230 | 0.199248 | +0.0018 | -40.7716 | -42.5409 | -1.7694 |
| c1_isotropic_noise_0.9 | weather | probe_unscaled_macro_f1 | 0.151771 | 0.152949 | +0.1178 | -49.5845 | -50.8404 | -1.2558 |
| c1_isotropic_noise_0.9 | scene | p10 | 0.415700 | 0.426653 | +1.0953 | -20.1000 | -19.7762 | +0.3238 |
| c1_isotropic_noise_0.9 | scene | macro_p10 | 0.329722 | 0.333134 | +0.3413 | -24.2069 | -24.4219 | -0.2150 |
| c1_isotropic_noise_0.9 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.9 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.9 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.9 | scene | probe_standardized_accuracy | 0.507000 | 0.514751 | +0.7751 | -25.2000 | -25.4323 | -0.2323 |
| c1_isotropic_noise_0.9 | scene | probe_standardized_balanced_accuracy | 0.369729 | 0.370947 | +0.1218 | -32.1739 | -32.4250 | -0.2511 |
| c1_isotropic_noise_0.9 | scene | probe_standardized_macro_f1 | 0.361497 | 0.365017 | +0.3520 | -34.3824 | -34.8769 | -0.4944 |
| c1_isotropic_noise_0.9 | scene | probe_unscaled_accuracy | 0.571000 | 0.579858 | +0.8858 | -18.4000 | -18.7182 | -0.3182 |
| c1_isotropic_noise_0.9 | scene | probe_unscaled_balanced_accuracy | 0.351172 | 0.351591 | +0.0420 | -28.6739 | -29.2305 | -0.5566 |
| c1_isotropic_noise_0.9 | scene | probe_unscaled_macro_f1 | 0.299314 | 0.302342 | +0.3027 | -36.6435 | -37.1818 | -0.5382 |
| c1_isotropic_noise_0.9 | timeofday | p10 | 0.455800 | 0.455748 | -0.0052 | -36.3500 | -37.1516 | -0.8016 |
| c1_isotropic_noise_0.9 | timeofday | macro_p10 | 0.353257 | 0.351484 | -0.1772 | -30.4493 | -31.2311 | -0.7818 |
| c1_isotropic_noise_0.9 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.9 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.9 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.9 | timeofday | probe_standardized_accuracy | 0.530000 | 0.528993 | -0.1007 | -39.0000 | -40.0814 | -1.0814 |
| c1_isotropic_noise_0.9 | timeofday | probe_standardized_balanced_accuracy | 0.386913 | 0.385638 | -0.1275 | -36.7304 | -37.8727 | -1.1422 |
| c1_isotropic_noise_0.9 | timeofday | probe_standardized_macro_f1 | 0.377334 | 0.376542 | -0.0792 | -40.0087 | -41.1714 | -1.1627 |
| c1_isotropic_noise_0.9 | timeofday | probe_unscaled_accuracy | 0.577000 | 0.576806 | -0.0194 | -33.6000 | -34.5880 | -0.9880 |
| c1_isotropic_noise_0.9 | timeofday | probe_unscaled_balanced_accuracy | 0.415271 | 0.414298 | -0.0973 | -33.4175 | -34.5226 | -1.1052 |
| c1_isotropic_noise_0.9 | timeofday | probe_unscaled_macro_f1 | 0.398708 | 0.398126 | -0.0581 | -36.8195 | -37.9014 | -1.0819 |
| c1_isotropic_noise_0.99 | weather | p10 | 0.393600 | 0.400509 | +0.6909 | -17.8600 | -17.8128 | +0.0472 |
| c1_isotropic_noise_0.99 | weather | macro_p10 | 0.203716 | 0.204473 | +0.0756 | -16.6708 | -16.6708 | -0.0001 |
| c1_isotropic_noise_0.99 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.99 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.99 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.99 | weather | probe_standardized_accuracy | 0.449000 | 0.453713 | +0.4713 | -32.6000 | -33.1638 | -0.5638 |
| c1_isotropic_noise_0.99 | weather | probe_standardized_balanced_accuracy | 0.187268 | 0.188484 | +0.1216 | -42.5049 | -44.1754 | -1.6705 |
| c1_isotropic_noise_0.99 | weather | probe_standardized_macro_f1 | 0.179517 | 0.181614 | +0.2097 | -46.7497 | -47.9383 | -1.1886 |
| c1_isotropic_noise_0.99 | weather | probe_unscaled_accuracy | 0.331000 | 0.332655 | +0.1655 | -45.3000 | -46.1851 | -0.8851 |
| c1_isotropic_noise_0.99 | weather | probe_unscaled_balanced_accuracy | 0.196919 | 0.197116 | +0.0197 | -41.0027 | -42.7541 | -1.7514 |
| c1_isotropic_noise_0.99 | weather | probe_unscaled_macro_f1 | 0.193862 | 0.193669 | -0.0194 | -45.3754 | -46.7684 | -1.3930 |
| c1_isotropic_noise_0.99 | scene | p10 | 0.413200 | 0.425839 | +1.2639 | -20.3500 | -19.8576 | +0.4924 |
| c1_isotropic_noise_0.99 | scene | macro_p10 | 0.322642 | 0.326755 | +0.4113 | -24.9148 | -25.0598 | -0.1449 |
| c1_isotropic_noise_0.99 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.99 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.99 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.99 | scene | probe_standardized_accuracy | 0.490000 | 0.496439 | +0.6439 | -26.9000 | -27.2635 | -0.3635 |
| c1_isotropic_noise_0.99 | scene | probe_standardized_balanced_accuracy | 0.348619 | 0.348900 | +0.0281 | -34.2849 | -34.6297 | -0.3448 |
| c1_isotropic_noise_0.99 | scene | probe_standardized_macro_f1 | 0.337546 | 0.340078 | +0.2533 | -36.7776 | -37.3707 | -0.5931 |
| c1_isotropic_noise_0.99 | scene | probe_unscaled_accuracy | 0.452000 | 0.456765 | +0.4765 | -30.3000 | -31.0275 | -0.7275 |
| c1_isotropic_noise_0.99 | scene | probe_unscaled_balanced_accuracy | 0.367667 | 0.367409 | -0.0258 | -27.0244 | -27.6488 | -0.6244 |
| c1_isotropic_noise_0.99 | scene | probe_unscaled_macro_f1 | 0.365529 | 0.366470 | +0.0941 | -30.0221 | -30.7689 | -0.7468 |
| c1_isotropic_noise_0.99 | timeofday | p10 | 0.445100 | 0.444456 | -0.0644 | -37.4200 | -38.2808 | -0.8608 |
| c1_isotropic_noise_0.99 | timeofday | macro_p10 | 0.345275 | 0.342742 | -0.2533 | -31.2474 | -32.1053 | -0.8579 |
| c1_isotropic_noise_0.99 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.99 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.99 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_isotropic_noise_0.99 | timeofday | probe_standardized_accuracy | 0.447000 | 0.443540 | -0.3460 | -47.3000 | -48.6267 | -1.3267 |
| c1_isotropic_noise_0.99 | timeofday | probe_standardized_balanced_accuracy | 0.325927 | 0.322911 | -0.3016 | -42.8291 | -44.1455 | -1.3164 |
| c1_isotropic_noise_0.99 | timeofday | probe_standardized_macro_f1 | 0.318720 | 0.316105 | -0.2615 | -45.8701 | -47.2151 | -1.3450 |
| c1_isotropic_noise_0.99 | timeofday | probe_unscaled_accuracy | 0.427000 | 0.424212 | -0.2788 | -48.6000 | -49.8474 | -1.2474 |
| c1_isotropic_noise_0.99 | timeofday | probe_unscaled_balanced_accuracy | 0.329812 | 0.324510 | -0.5302 | -41.9634 | -43.5015 | -1.5381 |
| c1_isotropic_noise_0.99 | timeofday | probe_unscaled_macro_f1 | 0.329690 | 0.324304 | -0.5386 | -43.7213 | -45.2836 | -1.5624 |
| c1_mean_interpolation_0 | weather | p10 | 0.572200 | 0.578637 | +0.6437 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | weather | macro_p10 | 0.370424 | 0.371181 | +0.0757 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | weather | probe_unscaled_accuracy | 0.784000 | 0.794507 | +1.0507 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | weather | probe_unscaled_balanced_accuracy | 0.606946 | 0.624657 | +1.7711 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | weather | probe_unscaled_macro_f1 | 0.647616 | 0.661353 | +1.3737 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | scene | p10 | 0.616700 | 0.624415 | +0.7715 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | scene | macro_p10 | 0.571790 | 0.577353 | +0.5563 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | scene | probe_unscaled_accuracy | 0.755000 | 0.767040 | +1.2040 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | scene | probe_unscaled_balanced_accuracy | 0.637911 | 0.643897 | +0.5986 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | scene | probe_unscaled_macro_f1 | 0.665750 | 0.674159 | +0.8409 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | timeofday | p10 | 0.819300 | 0.827263 | +0.7963 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | timeofday | macro_p10 | 0.657749 | 0.663795 | +0.6046 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | timeofday | probe_unscaled_accuracy | 0.913000 | 0.922686 | +0.9686 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | timeofday | probe_unscaled_balanced_accuracy | 0.749446 | 0.759524 | +1.0078 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0 | timeofday | probe_unscaled_macro_f1 | 0.766903 | 0.777141 | +1.0238 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | weather | p10 | 0.573100 | 0.579552 | +0.6452 | +0.0900 | +0.0916 | +0.0016 |
| c1_mean_interpolation_0.25 | weather | macro_p10 | 0.368532 | 0.369216 | +0.0684 | -0.1892 | -0.1965 | -0.0073 |
| c1_mean_interpolation_0.25 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | weather | probe_unscaled_accuracy | 0.782000 | 0.792472 | +1.0472 | -0.2000 | -0.2035 | -0.0035 |
| c1_mean_interpolation_0.25 | weather | probe_unscaled_balanced_accuracy | 0.588814 | 0.606129 | +1.7315 | -1.8132 | -1.8528 | -0.0396 |
| c1_mean_interpolation_0.25 | weather | probe_unscaled_macro_f1 | 0.626076 | 0.639411 | +1.3335 | -2.1540 | -2.1942 | -0.0402 |
| c1_mean_interpolation_0.25 | scene | p10 | 0.616500 | 0.624212 | +0.7712 | -0.0200 | -0.0203 | -0.0003 |
| c1_mean_interpolation_0.25 | scene | macro_p10 | 0.571143 | 0.576687 | +0.5544 | -0.0648 | -0.0666 | -0.0019 |
| c1_mean_interpolation_0.25 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | scene | probe_unscaled_accuracy | 0.767000 | 0.779247 | +1.2247 | +1.2000 | +1.2208 | +0.0208 |
| c1_mean_interpolation_0.25 | scene | probe_unscaled_balanced_accuracy | 0.684375 | 0.692050 | +0.7676 | +4.6464 | +4.8153 | +0.1690 |
| c1_mean_interpolation_0.25 | scene | probe_unscaled_macro_f1 | 0.710510 | 0.719365 | +0.8855 | +4.4761 | +4.5206 | +0.0445 |
| c1_mean_interpolation_0.25 | timeofday | p10 | 0.819200 | 0.827162 | +0.7962 | -0.0100 | -0.0102 | -0.0002 |
| c1_mean_interpolation_0.25 | timeofday | macro_p10 | 0.659153 | 0.665263 | +0.6110 | +0.1404 | +0.1468 | +0.0065 |
| c1_mean_interpolation_0.25 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.25 | timeofday | probe_unscaled_accuracy | 0.915000 | 0.924720 | +0.9720 | +0.2000 | +0.2035 | +0.0035 |
| c1_mean_interpolation_0.25 | timeofday | probe_unscaled_balanced_accuracy | 0.747218 | 0.757155 | +0.9937 | -0.2229 | -0.2370 | -0.0141 |
| c1_mean_interpolation_0.25 | timeofday | probe_unscaled_macro_f1 | 0.766589 | 0.776949 | +1.0360 | -0.0314 | -0.0192 | +0.0123 |
| c1_mean_interpolation_0.5 | weather | p10 | 0.574000 | 0.580468 | +0.6468 | +0.1800 | +0.1831 | +0.0031 |
| c1_mean_interpolation_0.5 | weather | macro_p10 | 0.368065 | 0.368669 | +0.0605 | -0.2359 | -0.2511 | -0.0152 |
| c1_mean_interpolation_0.5 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.5 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.5 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.5 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.5 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.5 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.5 | weather | probe_unscaled_accuracy | 0.777000 | 0.787386 | +1.0386 | -0.7000 | -0.7121 | -0.0121 |
| c1_mean_interpolation_0.5 | weather | probe_unscaled_balanced_accuracy | 0.616783 | 0.634862 | +1.8080 | +0.9837 | +1.0205 | +0.0368 |
| c1_mean_interpolation_0.5 | weather | probe_unscaled_macro_f1 | 0.652448 | 0.666009 | +1.3560 | +0.4832 | +0.4656 | -0.0177 |
| c1_mean_interpolation_0.5 | scene | p10 | 0.617500 | 0.625229 | +0.7729 | +0.0800 | +0.0814 | +0.0014 |
| c1_mean_interpolation_0.5 | scene | macro_p10 | 0.571179 | 0.576702 | +0.5523 | -0.0611 | -0.0651 | -0.0040 |
| c1_mean_interpolation_0.5 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.5 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.5 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.5 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.5 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.5 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.5 | scene | probe_unscaled_accuracy | 0.761000 | 0.773143 | +1.2143 | +0.6000 | +0.6104 | +0.0104 |
| c1_mean_interpolation_0.5 | scene | probe_unscaled_balanced_accuracy | 0.662200 | 0.669032 | +0.6833 | +2.4289 | +2.5136 | +0.0847 |
| c1_mean_interpolation_0.5 | scene | probe_unscaled_macro_f1 | 0.688883 | 0.697533 | +0.8650 | +2.3134 | +2.3374 | +0.0240 |
| c1_mean_interpolation_0.5 | timeofday | p10 | 0.817700 | 0.825636 | +0.7936 | -0.1600 | -0.1628 | -0.0028 |
| c1_mean_interpolation_0.5 | timeofday | macro_p10 | 0.658516 | 0.664629 | +0.6112 | +0.0767 | +0.0834 | +0.0067 |
| c1_mean_interpolation_0.5 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.5 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.5 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.5 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.5 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.5 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.5 | timeofday | probe_unscaled_accuracy | 0.922000 | 0.931841 | +0.9841 | +0.9000 | +0.9156 | +0.0156 |
| c1_mean_interpolation_0.5 | timeofday | probe_unscaled_balanced_accuracy | 0.755669 | 0.765839 | +1.0170 | +0.6223 | +0.6315 | +0.0092 |
| c1_mean_interpolation_0.5 | timeofday | probe_unscaled_macro_f1 | 0.781539 | 0.792647 | +1.1108 | +1.4636 | +1.5507 | +0.0870 |
| c1_mean_interpolation_0.75 | weather | p10 | 0.573100 | 0.579552 | +0.6452 | +0.0900 | +0.0916 | +0.0016 |
| c1_mean_interpolation_0.75 | weather | macro_p10 | 0.366367 | 0.366943 | +0.0575 | -0.4056 | -0.4238 | -0.0182 |
| c1_mean_interpolation_0.75 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.75 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.75 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.75 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.75 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.75 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.75 | weather | probe_unscaled_accuracy | 0.782000 | 0.792472 | +1.0472 | -0.2000 | -0.2035 | -0.0035 |
| c1_mean_interpolation_0.75 | weather | probe_unscaled_balanced_accuracy | 0.590943 | 0.608257 | +1.7313 | -1.6003 | -1.6401 | -0.0398 |
| c1_mean_interpolation_0.75 | weather | probe_unscaled_macro_f1 | 0.629327 | 0.642664 | +1.3337 | -1.8289 | -1.8689 | -0.0400 |
| c1_mean_interpolation_0.75 | scene | p10 | 0.617400 | 0.625127 | +0.7727 | +0.0700 | +0.0712 | +0.0012 |
| c1_mean_interpolation_0.75 | scene | macro_p10 | 0.570809 | 0.576318 | +0.5509 | -0.0981 | -0.1035 | -0.0054 |
| c1_mean_interpolation_0.75 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.75 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.75 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.75 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.75 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.75 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.75 | scene | probe_unscaled_accuracy | 0.746000 | 0.757884 | +1.1884 | -0.9000 | -0.9156 | -0.0156 |
| c1_mean_interpolation_0.75 | scene | probe_unscaled_balanced_accuracy | 0.613358 | 0.618524 | +0.5165 | -2.4553 | -2.5373 | -0.0820 |
| c1_mean_interpolation_0.75 | scene | probe_unscaled_macro_f1 | 0.635914 | 0.643844 | +0.7930 | -2.9835 | -3.0315 | -0.0479 |
| c1_mean_interpolation_0.75 | timeofday | p10 | 0.817400 | 0.825331 | +0.7931 | -0.1900 | -0.1933 | -0.0033 |
| c1_mean_interpolation_0.75 | timeofday | macro_p10 | 0.658312 | 0.664421 | +0.6109 | +0.0562 | +0.0626 | +0.0064 |
| c1_mean_interpolation_0.75 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.75 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.75 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.75 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.75 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.75 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.75 | timeofday | probe_unscaled_accuracy | 0.914000 | 0.923703 | +0.9703 | +0.1000 | +0.1017 | +0.0017 |
| c1_mean_interpolation_0.75 | timeofday | probe_unscaled_balanced_accuracy | 0.746536 | 0.756463 | +0.9927 | -0.2910 | -0.3061 | -0.0151 |
| c1_mean_interpolation_0.75 | timeofday | probe_unscaled_macro_f1 | 0.765133 | 0.775411 | +1.0278 | -0.1770 | -0.1730 | +0.0040 |
| c1_mean_interpolation_0.9 | weather | p10 | 0.572700 | 0.579145 | +0.6445 | +0.0500 | +0.0509 | +0.0009 |
| c1_mean_interpolation_0.9 | weather | macro_p10 | 0.365907 | 0.366458 | +0.0551 | -0.4517 | -0.4723 | -0.0206 |
| c1_mean_interpolation_0.9 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | weather | probe_unscaled_accuracy | 0.784000 | 0.794507 | +1.0507 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | weather | probe_unscaled_balanced_accuracy | 0.606946 | 0.624657 | +1.7711 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | weather | probe_unscaled_macro_f1 | 0.647616 | 0.661353 | +1.3737 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | scene | p10 | 0.617100 | 0.624822 | +0.7722 | +0.0400 | +0.0407 | +0.0007 |
| c1_mean_interpolation_0.9 | scene | macro_p10 | 0.571039 | 0.576571 | +0.5532 | -0.0751 | -0.0782 | -0.0031 |
| c1_mean_interpolation_0.9 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | scene | probe_unscaled_accuracy | 0.755000 | 0.767040 | +1.2040 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | scene | probe_unscaled_balanced_accuracy | 0.637911 | 0.643897 | +0.5986 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | scene | probe_unscaled_macro_f1 | 0.665750 | 0.674159 | +0.8409 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | timeofday | p10 | 0.817800 | 0.825738 | +0.7938 | -0.1500 | -0.1526 | -0.0026 |
| c1_mean_interpolation_0.9 | timeofday | macro_p10 | 0.658584 | 0.664698 | +0.6113 | +0.0835 | +0.0903 | +0.0068 |
| c1_mean_interpolation_0.9 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | timeofday | probe_unscaled_accuracy | 0.913000 | 0.922686 | +0.9686 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | timeofday | probe_unscaled_balanced_accuracy | 0.749446 | 0.759524 | +1.0078 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.9 | timeofday | probe_unscaled_macro_f1 | 0.766903 | 0.777141 | +1.0238 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | weather | p10 | 0.573000 | 0.579451 | +0.6451 | +0.0800 | +0.0814 | +0.0014 |
| c1_mean_interpolation_0.99 | weather | macro_p10 | 0.366464 | 0.367003 | +0.0539 | -0.3960 | -0.4178 | -0.0219 |
| c1_mean_interpolation_0.99 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | weather | probe_unscaled_accuracy | 0.783000 | 0.793489 | +1.0489 | -0.1000 | -0.1017 | -0.0017 |
| c1_mean_interpolation_0.99 | weather | probe_unscaled_balanced_accuracy | 0.604620 | 0.622157 | +1.7537 | -0.2326 | -0.2500 | -0.0174 |
| c1_mean_interpolation_0.99 | weather | probe_unscaled_macro_f1 | 0.645683 | 0.659382 | +1.3699 | -0.1933 | -0.1970 | -0.0038 |
| c1_mean_interpolation_0.99 | scene | p10 | 0.616600 | 0.624313 | +0.7713 | -0.0100 | -0.0102 | -0.0002 |
| c1_mean_interpolation_0.99 | scene | macro_p10 | 0.570677 | 0.576207 | +0.5530 | -0.1113 | -0.1146 | -0.0033 |
| c1_mean_interpolation_0.99 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | scene | probe_unscaled_accuracy | 0.755000 | 0.767040 | +1.2040 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | scene | probe_unscaled_balanced_accuracy | 0.637911 | 0.643897 | +0.5986 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | scene | probe_unscaled_macro_f1 | 0.665750 | 0.674159 | +0.8409 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | timeofday | p10 | 0.818000 | 0.825941 | +0.7941 | -0.1300 | -0.1322 | -0.0022 |
| c1_mean_interpolation_0.99 | timeofday | macro_p10 | 0.659045 | 0.665175 | +0.6130 | +0.1295 | +0.1380 | +0.0085 |
| c1_mean_interpolation_0.99 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | timeofday | probe_unscaled_accuracy | 0.913000 | 0.922686 | +0.9686 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | timeofday | probe_unscaled_balanced_accuracy | 0.749446 | 0.759524 | +1.0078 | +0.0000 | +0.0000 | +0.0000 |
| c1_mean_interpolation_0.99 | timeofday | probe_unscaled_macro_f1 | 0.766903 | 0.777141 | +1.0238 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | weather | p10 | 0.572200 | 0.578637 | +0.6437 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | weather | macro_p10 | 0.370424 | 0.371181 | +0.0757 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | weather | probe_standardized_accuracy | 0.775000 | 0.785351 | +1.0351 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | weather | probe_standardized_balanced_accuracy | 0.612317 | 0.630239 | +1.7921 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | weather | probe_standardized_macro_f1 | 0.647014 | 0.660997 | +1.3983 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | scene | p10 | 0.616700 | 0.624415 | +0.7715 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | scene | macro_p10 | 0.571790 | 0.577353 | +0.5563 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | scene | probe_standardized_accuracy | 0.759000 | 0.769074 | +1.0074 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | scene | probe_standardized_balanced_accuracy | 0.691468 | 0.695197 | +0.3729 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | scene | probe_standardized_macro_f1 | 0.705321 | 0.713786 | +0.8464 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | timeofday | p10 | 0.819300 | 0.827263 | +0.7963 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | timeofday | macro_p10 | 0.657749 | 0.663795 | +0.6046 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | timeofday | probe_standardized_accuracy | 0.920000 | 0.929807 | +0.9807 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | timeofday | probe_standardized_balanced_accuracy | 0.754218 | 0.764365 | +1.0148 | +0.0000 | +0.0000 | +0.0000 |
| c2_native_768 | timeofday | probe_standardized_macro_f1 | 0.777421 | 0.788256 | +1.0835 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_512 | weather | p10 | 0.571300 | 0.577721 | +0.6421 | -0.0900 | -0.0916 | -0.0016 |
| c2_mrl_512 | weather | macro_p10 | 0.366539 | 0.367193 | +0.0654 | -0.3885 | -0.3988 | -0.0103 |
| c2_mrl_512 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_512 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_512 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_512 | weather | probe_standardized_accuracy | 0.779000 | 0.790437 | +1.1437 | +0.4000 | +0.5086 | +0.1086 |
| c2_mrl_512 | weather | probe_standardized_balanced_accuracy | 0.609771 | 0.628169 | +1.8398 | -0.2546 | -0.2069 | +0.0477 |
| c2_mrl_512 | weather | probe_standardized_macro_f1 | 0.643326 | 0.658392 | +1.5066 | -0.3688 | -0.2605 | +0.1083 |
| c2_mrl_512 | scene | p10 | 0.614700 | 0.622279 | +0.7579 | -0.2000 | -0.2136 | -0.0136 |
| c2_mrl_512 | scene | macro_p10 | 0.567439 | 0.572883 | +0.5444 | -0.4351 | -0.4470 | -0.0119 |
| c2_mrl_512 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_512 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_512 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_512 | scene | probe_standardized_accuracy | 0.768000 | 0.780264 | +1.2264 | +0.9000 | +1.1190 | +0.2190 |
| c2_mrl_512 | scene | probe_standardized_balanced_accuracy | 0.697402 | 0.705570 | +0.8167 | +0.5934 | +1.0373 | +0.4439 |
| c2_mrl_512 | scene | probe_standardized_macro_f1 | 0.716568 | 0.727273 | +1.0705 | +1.1246 | +1.3487 | +0.2241 |
| c2_mrl_512 | timeofday | p10 | 0.818500 | 0.826450 | +0.7950 | -0.0800 | -0.0814 | -0.0014 |
| c2_mrl_512 | timeofday | macro_p10 | 0.655873 | 0.661850 | +0.5977 | -0.1876 | -0.1945 | -0.0069 |
| c2_mrl_512 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_512 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_512 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_512 | timeofday | probe_standardized_accuracy | 0.925000 | 0.933876 | +0.8876 | +0.5000 | +0.4069 | -0.0931 |
| c2_mrl_512 | timeofday | probe_standardized_balanced_accuracy | 0.761394 | 0.767313 | +0.5919 | +0.7176 | +0.2948 | -0.4228 |
| c2_mrl_512 | timeofday | probe_standardized_macro_f1 | 0.789781 | 0.796201 | +0.6420 | +1.2360 | +0.7945 | -0.4415 |
| c2_pca_512 | weather | p10 | 0.551700 | 0.557782 | +0.6082 | -2.0500 | -2.0855 | -0.0355 |
| c2_pca_512 | weather | macro_p10 | 0.358174 | 0.359323 | +0.1149 | -1.2250 | -1.1858 | +0.0392 |
| c2_pca_512 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_512 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_512 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_512 | weather | probe_standardized_accuracy | 0.729000 | 0.741607 | +1.2607 | -4.6000 | -4.3744 | +0.2256 |
| c2_pca_512 | weather | probe_standardized_balanced_accuracy | 0.537878 | 0.553562 | +1.5684 | -7.4439 | -7.6677 | -0.2238 |
| c2_pca_512 | weather | probe_standardized_macro_f1 | 0.573849 | 0.591931 | +1.8082 | -7.3165 | -6.9066 | +0.4099 |
| c2_pca_512 | scene | p10 | 0.610400 | 0.617904 | +0.7504 | -0.6300 | -0.6511 | -0.0211 |
| c2_pca_512 | scene | macro_p10 | 0.574571 | 0.580334 | +0.5763 | +0.2780 | +0.2981 | +0.0201 |
| c2_pca_512 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_512 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_512 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_512 | scene | probe_standardized_accuracy | 0.741000 | 0.748728 | +0.7728 | -1.8000 | -2.0346 | -0.2346 |
| c2_pca_512 | scene | probe_standardized_balanced_accuracy | 0.661918 | 0.658922 | -0.2997 | -2.9550 | -3.6275 | -0.6725 |
| c2_pca_512 | scene | probe_standardized_macro_f1 | 0.675949 | 0.679246 | +0.3297 | -2.9373 | -3.4540 | -0.5167 |
| c2_pca_512 | timeofday | p10 | 0.822800 | 0.830722 | +0.7922 | +0.3500 | +0.3459 | -0.0041 |
| c2_pca_512 | timeofday | macro_p10 | 0.645252 | 0.650613 | +0.5361 | -1.2497 | -1.3182 | -0.0685 |
| c2_pca_512 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_512 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_512 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_512 | timeofday | probe_standardized_accuracy | 0.918000 | 0.926755 | +0.8755 | -0.2000 | -0.3052 | -0.1052 |
| c2_pca_512 | timeofday | probe_standardized_balanced_accuracy | 0.742079 | 0.751032 | +0.8953 | -1.2139 | -1.3333 | -0.1194 |
| c2_pca_512 | timeofday | probe_standardized_macro_f1 | 0.765724 | 0.774797 | +0.9074 | -1.1697 | -1.3459 | -0.1762 |
| c2_mrl_256 | weather | p10 | 0.567100 | 0.573347 | +0.6247 | -0.5100 | -0.5290 | -0.0190 |
| c2_mrl_256 | weather | macro_p10 | 0.364362 | 0.364631 | +0.0268 | -0.6061 | -0.6550 | -0.0489 |
| c2_mrl_256 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_256 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_256 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_256 | weather | probe_standardized_accuracy | 0.771000 | 0.781282 | +1.0282 | -0.4000 | -0.4069 | -0.0069 |
| c2_mrl_256 | weather | probe_standardized_balanced_accuracy | 0.593058 | 0.610656 | +1.7597 | -1.9259 | -1.9583 | -0.0324 |
| c2_mrl_256 | weather | probe_standardized_macro_f1 | 0.628872 | 0.642268 | +1.3396 | -1.8142 | -1.8729 | -0.0587 |
| c2_mrl_256 | scene | p10 | 0.614400 | 0.621974 | +0.7574 | -0.2300 | -0.2442 | -0.0142 |
| c2_mrl_256 | scene | macro_p10 | 0.567802 | 0.573240 | +0.5438 | -0.3988 | -0.4113 | -0.0125 |
| c2_mrl_256 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_256 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_256 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_256 | scene | probe_standardized_accuracy | 0.770000 | 0.782299 | +1.2299 | +1.1000 | +1.3225 | +0.2225 |
| c2_mrl_256 | scene | probe_standardized_balanced_accuracy | 0.690117 | 0.697964 | +0.7846 | -0.1351 | +0.2767 | +0.4118 |
| c2_mrl_256 | scene | probe_standardized_macro_f1 | 0.712959 | 0.722266 | +0.9307 | +0.7637 | +0.8480 | +0.0843 |
| c2_mrl_256 | timeofday | p10 | 0.816100 | 0.824110 | +0.8010 | -0.3200 | -0.3154 | +0.0046 |
| c2_mrl_256 | timeofday | macro_p10 | 0.658309 | 0.664892 | +0.6583 | +0.0560 | +0.1097 | +0.0537 |
| c2_mrl_256 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_256 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_256 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_256 | timeofday | probe_standardized_accuracy | 0.907000 | 0.916582 | +0.9582 | -1.3000 | -1.3225 | -0.0225 |
| c2_mrl_256 | timeofday | probe_standardized_balanced_accuracy | 0.759635 | 0.770296 | +1.0660 | +0.5418 | +0.5930 | +0.0513 |
| c2_mrl_256 | timeofday | probe_standardized_macro_f1 | 0.770831 | 0.780820 | +0.9989 | -0.6590 | -0.7436 | -0.0846 |
| c2_pca_256 | weather | p10 | 0.554500 | 0.560427 | +0.5927 | -1.7700 | -1.8210 | -0.0510 |
| c2_pca_256 | weather | macro_p10 | 0.362496 | 0.363341 | +0.0846 | -0.7928 | -0.7840 | +0.0088 |
| c2_pca_256 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_256 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_256 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_256 | weather | probe_standardized_accuracy | 0.761000 | 0.773143 | +1.2143 | -1.4000 | -1.2208 | +0.1792 |
| c2_pca_256 | weather | probe_standardized_balanced_accuracy | 0.572515 | 0.588547 | +1.6032 | -3.9802 | -4.1691 | -0.1889 |
| c2_pca_256 | weather | probe_standardized_macro_f1 | 0.615703 | 0.632712 | +1.7009 | -3.1311 | -2.8285 | +0.3026 |
| c2_pca_256 | scene | p10 | 0.612100 | 0.619736 | +0.7636 | -0.4600 | -0.4680 | -0.0080 |
| c2_pca_256 | scene | macro_p10 | 0.574192 | 0.580041 | +0.5848 | +0.2402 | +0.2688 | +0.0286 |
| c2_pca_256 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_256 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_256 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_256 | scene | probe_standardized_accuracy | 0.747000 | 0.756867 | +0.9867 | -1.2000 | -1.2208 | -0.0208 |
| c2_pca_256 | scene | probe_standardized_balanced_accuracy | 0.661442 | 0.662672 | +0.1230 | -3.0026 | -3.2525 | -0.2499 |
| c2_pca_256 | scene | probe_standardized_macro_f1 | 0.681258 | 0.687289 | +0.6032 | -2.4064 | -2.6496 | -0.2432 |
| c2_pca_256 | timeofday | p10 | 0.825400 | 0.833367 | +0.7967 | +0.6100 | +0.6104 | +0.0004 |
| c2_pca_256 | timeofday | macro_p10 | 0.646324 | 0.652054 | +0.5730 | -1.1426 | -1.1741 | -0.0316 |
| c2_pca_256 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_256 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_256 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_256 | timeofday | probe_standardized_accuracy | 0.922000 | 0.931841 | +0.9841 | +0.2000 | +0.2035 | +0.0035 |
| c2_pca_256 | timeofday | probe_standardized_balanced_accuracy | 0.744894 | 0.754581 | +0.9687 | -0.9324 | -0.9785 | -0.0461 |
| c2_pca_256 | timeofday | probe_standardized_macro_f1 | 0.770337 | 0.781249 | +1.0912 | -0.7084 | -0.7007 | +0.0077 |
| c2_mrl_128 | weather | p10 | 0.557200 | 0.563276 | +0.6076 | -1.5000 | -1.5361 | -0.0361 |
| c2_mrl_128 | weather | macro_p10 | 0.342614 | 0.342027 | -0.0587 | -2.7810 | -2.9154 | -0.1345 |
| c2_mrl_128 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_128 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_128 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_128 | weather | probe_standardized_accuracy | 0.744000 | 0.754832 | +1.0832 | -3.1000 | -3.0519 | +0.0481 |
| c2_mrl_128 | weather | probe_standardized_balanced_accuracy | 0.533743 | 0.549085 | +1.5342 | -7.8574 | -8.1153 | -0.2579 |
| c2_mrl_128 | weather | probe_standardized_macro_f1 | 0.568651 | 0.581858 | +1.3207 | -7.8363 | -7.9140 | -0.0776 |
| c2_mrl_128 | scene | p10 | 0.600400 | 0.607833 | +0.7433 | -1.6300 | -1.6582 | -0.0282 |
| c2_mrl_128 | scene | macro_p10 | 0.548652 | 0.553721 | +0.5069 | -2.3138 | -2.3632 | -0.0493 |
| c2_mrl_128 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_128 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_128 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_128 | scene | probe_standardized_accuracy | 0.754000 | 0.763988 | +0.9988 | -0.5000 | -0.5086 | -0.0086 |
| c2_mrl_128 | scene | probe_standardized_balanced_accuracy | 0.663192 | 0.666983 | +0.3791 | -2.8276 | -2.8214 | +0.0062 |
| c2_mrl_128 | scene | probe_standardized_macro_f1 | 0.685876 | 0.695540 | +0.9664 | -1.9445 | -1.8246 | +0.1199 |
| c2_mrl_128 | timeofday | p10 | 0.798800 | 0.806612 | +0.7812 | -2.0500 | -2.0651 | -0.0151 |
| c2_mrl_128 | timeofday | macro_p10 | 0.640540 | 0.646756 | +0.6216 | -1.7209 | -1.7038 | +0.0171 |
| c2_mrl_128 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_128 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_128 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_mrl_128 | timeofday | probe_standardized_accuracy | 0.919000 | 0.929807 | +1.0807 | -0.1000 | +0.0000 | +0.1000 |
| c2_mrl_128 | timeofday | probe_standardized_balanced_accuracy | 0.732073 | 0.741939 | +0.9866 | -2.2145 | -2.2426 | -0.0282 |
| c2_mrl_128 | timeofday | probe_standardized_macro_f1 | 0.755781 | 0.768037 | +1.2256 | -2.1640 | -2.0219 | +0.1421 |
| c2_pca_128 | weather | p10 | 0.556000 | 0.561851 | +0.5851 | -1.6200 | -1.6785 | -0.0585 |
| c2_pca_128 | weather | macro_p10 | 0.368152 | 0.368825 | +0.0673 | -0.2272 | -0.2356 | -0.0085 |
| c2_pca_128 | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_128 | weather | majority_floor | 0.588000 | 0.595117 | +0.7117 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_128 | weather | balanced_constant_floor | 0.200000 | 0.200000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_128 | weather | probe_standardized_accuracy | 0.772000 | 0.782299 | +1.0299 | -0.3000 | -0.3052 | -0.0052 |
| c2_pca_128 | weather | probe_standardized_balanced_accuracy | 0.606779 | 0.623574 | +1.6795 | -0.5538 | -0.6665 | -0.1127 |
| c2_pca_128 | weather | probe_standardized_macro_f1 | 0.637759 | 0.650788 | +1.3029 | -0.9255 | -1.0209 | -0.0954 |
| c2_pca_128 | scene | p10 | 0.614500 | 0.622482 | +0.7982 | -0.2200 | -0.1933 | +0.0267 |
| c2_pca_128 | scene | macro_p10 | 0.574583 | 0.580870 | +0.6287 | +0.2793 | +0.3517 | +0.0724 |
| c2_pca_128 | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_128 | scene | majority_floor | 0.577000 | 0.585961 | +0.8961 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_128 | scene | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_128 | scene | probe_standardized_accuracy | 0.766000 | 0.778230 | +1.2230 | +0.7000 | +0.9156 | +0.2156 |
| c2_pca_128 | scene | probe_standardized_balanced_accuracy | 0.676271 | 0.683718 | +0.7447 | -1.5197 | -1.1479 | +0.3718 |
| c2_pca_128 | scene | probe_standardized_macro_f1 | 0.706284 | 0.715114 | +0.8830 | +0.0963 | +0.1328 | +0.0365 |
| c2_pca_128 | timeofday | p10 | 0.831500 | 0.839980 | +0.8480 | +1.2200 | +1.2716 | +0.0516 |
| c2_pca_128 | timeofday | macro_p10 | 0.651332 | 0.657820 | +0.6488 | -0.6417 | -0.5975 | +0.0442 |
| c2_pca_128 | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_128 | timeofday | majority_floor | 0.489000 | 0.490336 | +0.1336 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_128 | timeofday | balanced_constant_floor | 0.333333 | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| c2_pca_128 | timeofday | probe_standardized_accuracy | 0.919000 | 0.927772 | +0.8772 | -0.1000 | -0.2035 | -0.1035 |
| c2_pca_128 | timeofday | probe_standardized_balanced_accuracy | 0.771495 | 0.781747 | +1.0251 | +1.7278 | +1.7381 | +0.0104 |
| c2_pca_128 | timeofday | probe_standardized_macro_f1 | 0.791486 | 0.800743 | +0.9257 | +1.4065 | +1.2487 | -0.1578 |
| c3_native_ref | weather | p10 | 0.572200 | 0.578637 | +0.6437 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_ref | weather | macro_p10 | 0.370424 | 0.371181 | +0.0757 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_ref | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_ref | weather | chance_self_excluded | 0.390661 | 0.398083 | +0.7422 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_ref | scene | p10 | 0.616700 | 0.624415 | +0.7715 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_ref | scene | macro_p10 | 0.571790 | 0.577353 | +0.5563 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_ref | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_ref | scene | chance_self_excluded | 0.421259 | 0.431594 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_ref | timeofday | p10 | 0.819300 | 0.827263 | +0.7963 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_ref | timeofday | macro_p10 | 0.657749 | 0.663795 | +0.6046 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_ref | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_ref | timeofday | chance_self_excluded | 0.432126 | 0.433481 | +0.1355 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp16_gallery | weather | p10 | 0.572000 | 0.578433 | +0.6433 | -0.0200 | -0.0203 | -0.0003 |
| c3_native_fp16_gallery | weather | macro_p10 | 0.370356 | 0.371113 | +0.0757 | -0.0068 | -0.0068 | -0.0000 |
| c3_native_fp16_gallery | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp16_gallery | weather | chance_self_excluded | 0.390661 | 0.398083 | +0.7422 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp16_gallery | scene | p10 | 0.616900 | 0.624619 | +0.7719 | +0.0200 | +0.0203 | +0.0003 |
| c3_native_fp16_gallery | scene | macro_p10 | 0.572053 | 0.577619 | +0.5566 | +0.0262 | +0.0266 | +0.0003 |
| c3_native_fp16_gallery | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp16_gallery | scene | chance_self_excluded | 0.421259 | 0.431594 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp16_gallery | timeofday | p10 | 0.819200 | 0.827162 | +0.7962 | -0.0100 | -0.0102 | -0.0002 |
| c3_native_fp16_gallery | timeofday | macro_p10 | 0.657672 | 0.663717 | +0.6044 | -0.0077 | -0.0078 | -0.0001 |
| c3_native_fp16_gallery | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp16_gallery | timeofday | chance_self_excluded | 0.432126 | 0.433481 | +0.1355 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp16_both | weather | p10 | 0.572000 | 0.578433 | +0.6433 | -0.0200 | -0.0203 | -0.0003 |
| c3_native_fp16_both | weather | macro_p10 | 0.370356 | 0.371113 | +0.0757 | -0.0068 | -0.0068 | -0.0000 |
| c3_native_fp16_both | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp16_both | weather | chance_self_excluded | 0.390661 | 0.398083 | +0.7422 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp16_both | scene | p10 | 0.616900 | 0.624619 | +0.7719 | +0.0200 | +0.0203 | +0.0003 |
| c3_native_fp16_both | scene | macro_p10 | 0.572053 | 0.577619 | +0.5566 | +0.0262 | +0.0266 | +0.0003 |
| c3_native_fp16_both | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp16_both | scene | chance_self_excluded | 0.421259 | 0.431594 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp16_both | timeofday | p10 | 0.819200 | 0.827162 | +0.7962 | -0.0100 | -0.0102 | -0.0002 |
| c3_native_fp16_both | timeofday | macro_p10 | 0.657672 | 0.663717 | +0.6044 | -0.0077 | -0.0078 | -0.0001 |
| c3_native_fp16_both | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp16_both | timeofday | chance_self_excluded | 0.432126 | 0.433481 | +0.1355 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_int8_gallery | weather | p10 | 0.571800 | 0.578230 | +0.6430 | -0.0400 | -0.0407 | -0.0007 |
| c3_native_int8_gallery | weather | macro_p10 | 0.370015 | 0.370734 | +0.0719 | -0.0409 | -0.0447 | -0.0038 |
| c3_native_int8_gallery | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_int8_gallery | weather | chance_self_excluded | 0.390661 | 0.398083 | +0.7422 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_int8_gallery | scene | p10 | 0.616600 | 0.624313 | +0.7713 | -0.0100 | -0.0102 | -0.0002 |
| c3_native_int8_gallery | scene | macro_p10 | 0.571806 | 0.577370 | +0.5564 | +0.0016 | +0.0017 | +0.0001 |
| c3_native_int8_gallery | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_int8_gallery | scene | chance_self_excluded | 0.421259 | 0.431594 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_int8_gallery | timeofday | p10 | 0.819300 | 0.827263 | +0.7963 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_int8_gallery | timeofday | macro_p10 | 0.658450 | 0.664527 | +0.6077 | +0.0701 | +0.0732 | +0.0032 |
| c3_native_int8_gallery | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_int8_gallery | timeofday | chance_self_excluded | 0.432126 | 0.433481 | +0.1355 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_int8_both | weather | p10 | 0.572000 | 0.578433 | +0.6433 | -0.0200 | -0.0203 | -0.0003 |
| c3_native_int8_both | weather | macro_p10 | 0.369997 | 0.370706 | +0.0709 | -0.0426 | -0.0475 | -0.0048 |
| c3_native_int8_both | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_int8_both | weather | chance_self_excluded | 0.390661 | 0.398083 | +0.7422 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_int8_both | scene | p10 | 0.616600 | 0.624313 | +0.7713 | -0.0100 | -0.0102 | -0.0002 |
| c3_native_int8_both | scene | macro_p10 | 0.572035 | 0.577608 | +0.5572 | +0.0245 | +0.0255 | +0.0010 |
| c3_native_int8_both | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_int8_both | scene | chance_self_excluded | 0.421259 | 0.431594 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_int8_both | timeofday | p10 | 0.819000 | 0.826958 | +0.7958 | -0.0300 | -0.0305 | -0.0005 |
| c3_native_int8_both | timeofday | macro_p10 | 0.658228 | 0.664302 | +0.6074 | +0.0479 | +0.0507 | +0.0028 |
| c3_native_int8_both | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_int8_both | timeofday | chance_self_excluded | 0.432126 | 0.433481 | +0.1355 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp32_arithmetic | weather | p10 | 0.572200 | 0.578637 | +0.6437 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp32_arithmetic | weather | macro_p10 | 0.370424 | 0.371181 | +0.0757 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp32_arithmetic | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp32_arithmetic | weather | chance_self_excluded | 0.390661 | 0.398083 | +0.7422 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp32_arithmetic | scene | p10 | 0.616700 | 0.624415 | +0.7715 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp32_arithmetic | scene | macro_p10 | 0.571790 | 0.577353 | +0.5563 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp32_arithmetic | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp32_arithmetic | scene | chance_self_excluded | 0.421259 | 0.431594 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp32_arithmetic | timeofday | p10 | 0.819300 | 0.827263 | +0.7963 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp32_arithmetic | timeofday | macro_p10 | 0.657749 | 0.663795 | +0.6046 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp32_arithmetic | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c3_native_fp32_arithmetic | timeofday | chance_self_excluded | 0.432126 | 0.433481 | +0.1355 | +0.0000 | +0.0000 | +0.0000 |
| c3_mean99_ref | weather | p10 | 0.571500 | 0.577925 | +0.6425 | +0.0000 | +0.0000 | +0.0000 |
| c3_mean99_ref | weather | macro_p10 | 0.369577 | 0.370320 | +0.0744 | +0.0000 | +0.0000 | +0.0000 |
| c3_mean99_ref | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c3_mean99_ref | weather | chance_self_excluded | 0.390661 | 0.398083 | +0.7422 | +0.0000 | +0.0000 | +0.0000 |
| c3_mean99_ref | scene | p10 | 0.617700 | 0.625432 | +0.7732 | +0.0000 | +0.0000 | +0.0000 |
| c3_mean99_ref | scene | macro_p10 | 0.572441 | 0.578007 | +0.5565 | +0.0000 | +0.0000 | +0.0000 |
| c3_mean99_ref | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c3_mean99_ref | scene | chance_self_excluded | 0.421259 | 0.431594 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c3_mean99_ref | timeofday | p10 | 0.818600 | 0.826551 | +0.7951 | +0.0000 | +0.0000 | +0.0000 |
| c3_mean99_ref | timeofday | macro_p10 | 0.657955 | 0.664025 | +0.6070 | +0.0000 | +0.0000 | +0.0000 |
| c3_mean99_ref | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c3_mean99_ref | timeofday | chance_self_excluded | 0.432126 | 0.433481 | +0.1355 | +0.0000 | +0.0000 | +0.0000 |
| c3_mean99_fp16_gallery | weather | p10 | 0.571900 | 0.578332 | +0.6432 | +0.0400 | +0.0407 | +0.0007 |
| c3_mean99_fp16_gallery | weather | macro_p10 | 0.370841 | 0.371638 | +0.0797 | +0.1264 | +0.1317 | +0.0053 |
| c3_mean99_fp16_gallery | weather | chance_sum_p_squared | 0.391270 | 0.398695 | +0.7425 | +0.0000 | +0.0000 | +0.0000 |
| c3_mean99_fp16_gallery | weather | chance_self_excluded | 0.390661 | 0.398083 | +0.7422 | +0.0000 | +0.0000 | +0.0000 |
| c3_mean99_fp16_gallery | scene | p10 | 0.617900 | 0.625636 | +0.7736 | +0.0200 | +0.0203 | +0.0003 |
| c3_mean99_fp16_gallery | scene | macro_p10 | 0.572410 | 0.577972 | +0.5562 | -0.0031 | -0.0034 | -0.0003 |
| c3_mean99_fp16_gallery | scene | chance_sum_p_squared | 0.421838 | 0.432172 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c3_mean99_fp16_gallery | scene | chance_self_excluded | 0.421259 | 0.431594 | +1.0334 | +0.0000 | +0.0000 | +0.0000 |
| c3_mean99_fp16_gallery | timeofday | p10 | 0.820400 | 0.828383 | +0.7983 | +0.1800 | +0.1831 | +0.0031 |
| c3_mean99_fp16_gallery | timeofday | macro_p10 | 0.659577 | 0.665682 | +0.6105 | +0.1621 | +0.1656 | +0.0035 |
| c3_mean99_fp16_gallery | timeofday | chance_sum_p_squared | 0.432694 | 0.434058 | +0.1364 | +0.0000 | +0.0000 | +0.0000 |
| c3_mean99_fp16_gallery | timeofday | chance_self_excluded | 0.432126 | 0.433481 | +0.1355 | +0.0000 | +0.0000 | +0.0000 |

## Validation geometry

[ours] Raw units; every metric also in JSON.

| Condition | Metric | Historical | Reduced | Absolute change | Effect change |
| --- | --- | --- | --- | --- | --- |
| c1_pristine | total_variance | 0.260057 | 0.257033 | -0.003024 | 0.000000 |
| c1_pristine | mean_pairwise_cosine | 0.739943 | 0.742967 | 0.003024 | 0.000000 |
| c1_pristine | rankme | 271.900193 | 270.809584 | -1.090609 | 0.000000 |
| c1_pristine | participation_ratio | 39.361108 | 38.264940 | -1.096169 | 0.000000 |
| c1_scale_contraction_0 | total_variance | 0.260057 | 0.257033 | -0.003024 | 0.000000 |
| c1_scale_contraction_0 | mean_pairwise_cosine | 0.739943 | 0.742967 | 0.003024 | 0.000000 |
| c1_scale_contraction_0 | rankme | 271.900193 | 270.809584 | -1.090609 | 0.000000 |
| c1_scale_contraction_0 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | 0.000000 |
| c1_scale_contraction_0.25 | total_variance | 0.146282 | 0.144581 | -0.001701 | 0.001323 |
| c1_scale_contraction_0.25 | mean_pairwise_cosine | 0.739943 | 0.742967 | 0.003024 | 0.000000 |
| c1_scale_contraction_0.25 | rankme | 271.900193 | 270.809584 | -1.090609 | 0.000000 |
| c1_scale_contraction_0.25 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | -0.000000 |
| c1_scale_contraction_0.5 | total_variance | 0.065014 | 0.064258 | -0.000756 | 0.002268 |
| c1_scale_contraction_0.5 | mean_pairwise_cosine | 0.739943 | 0.742967 | 0.003024 | 0.000000 |
| c1_scale_contraction_0.5 | rankme | 271.900193 | 270.809584 | -1.090609 | 0.000000 |
| c1_scale_contraction_0.5 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | 0.000000 |
| c1_scale_contraction_0.75 | total_variance | 0.016254 | 0.016065 | -0.000189 | 0.002835 |
| c1_scale_contraction_0.75 | mean_pairwise_cosine | 0.739943 | 0.742967 | 0.003024 | 0.000000 |
| c1_scale_contraction_0.75 | rankme | 271.900193 | 270.809584 | -1.090609 | 0.000000 |
| c1_scale_contraction_0.75 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | 0.000000 |
| c1_scale_contraction_0.9 | total_variance | 0.002601 | 0.002570 | -0.000030 | 0.002994 |
| c1_scale_contraction_0.9 | mean_pairwise_cosine | 0.739943 | 0.742967 | 0.003024 | 0.000000 |
| c1_scale_contraction_0.9 | rankme | 271.900193 | 270.809584 | -1.090609 | -0.000000 |
| c1_scale_contraction_0.9 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | -0.000000 |
| c1_scale_contraction_0.99 | total_variance | 0.000026 | 0.000026 | -0.000000 | 0.003024 |
| c1_scale_contraction_0.99 | mean_pairwise_cosine | 0.739943 | 0.742967 | 0.003024 | 0.000000 |
| c1_scale_contraction_0.99 | rankme | 271.900193 | 270.809584 | -1.090609 | -0.000000 |
| c1_scale_contraction_0.99 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | -0.000000 |
| c1_rank_truncation_0 | total_variance | 0.260057 | 0.257033 | -0.003024 | 0.000000 |
| c1_rank_truncation_0 | mean_pairwise_cosine | 0.739943 | 0.742967 | 0.003024 | 0.000000 |
| c1_rank_truncation_0 | rankme | 271.900193 | 270.809584 | -1.090609 | 0.000000 |
| c1_rank_truncation_0 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | 0.000000 |
| c1_rank_truncation_0.25 | total_variance | 0.260057 | 0.257033 | -0.003024 | 0.000000 |
| c1_rank_truncation_0.25 | mean_pairwise_cosine | 0.739943 | 0.742967 | 0.003024 | 0.000000 |
| c1_rank_truncation_0.25 | rankme | 271.900098 | 270.809490 | -1.090608 | 0.000001 |
| c1_rank_truncation_0.25 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | -0.000000 |
| c1_rank_truncation_0.5 | total_variance | 0.252172 | 0.250371 | -0.001801 | 0.001223 |
| c1_rank_truncation_0.5 | mean_pairwise_cosine | 0.745616 | 0.747803 | 0.002187 | -0.000837 |
| c1_rank_truncation_0.5 | rankme | 220.984835 | 220.377279 | -0.607556 | 0.483052 |
| c1_rank_truncation_0.5 | participation_ratio | 37.410380 | 36.437584 | -0.972796 | 0.123373 |
| c1_rank_truncation_0.75 | total_variance | 0.222345 | 0.222446 | 0.000101 | 0.003125 |
| c1_rank_truncation_0.75 | mean_pairwise_cosine | 0.768455 | 0.769019 | 0.000564 | -0.002460 |
| c1_rank_truncation_0.75 | rankme | 118.178612 | 118.047239 | -0.131373 | 0.959236 |
| c1_rank_truncation_0.75 | participation_ratio | 29.808328 | 29.213502 | -0.594826 | 0.501342 |
| c1_rank_truncation_0.9 | total_variance | 0.173205 | 0.174181 | 0.000976 | 0.004000 |
| c1_rank_truncation_0.9 | mean_pairwise_cosine | 0.810081 | 0.809494 | -0.000586 | -0.003610 |
| c1_rank_truncation_0.9 | rankme | 45.861606 | 45.845794 | -0.015812 | 1.074797 |
| c1_rank_truncation_0.9 | participation_ratio | 18.775783 | 18.505399 | -0.270384 | 0.825785 |
| c1_rank_truncation_0.99 | total_variance | 0.078697 | 0.079585 | 0.000887 | 0.003912 |
| c1_rank_truncation_0.99 | mean_pairwise_cosine | 0.906139 | 0.905020 | -0.001119 | -0.004143 |
| c1_rank_truncation_0.99 | rankme | 4.607914 | 4.618317 | 0.010403 | 1.101012 |
| c1_rank_truncation_0.99 | participation_ratio | 4.427410 | 4.393465 | -0.033944 | 1.062224 |
| c1_mean_injection_0 | total_variance | 0.260057 | 0.257033 | -0.003024 | 0.000000 |
| c1_mean_injection_0 | mean_pairwise_cosine | 0.739943 | 0.742967 | 0.003024 | 0.000000 |
| c1_mean_injection_0 | rankme | 271.900193 | 270.809584 | -1.090609 | 0.000000 |
| c1_mean_injection_0 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | 0.000000 |
| c1_mean_injection_0.25 | total_variance | 0.260057 | 0.257033 | -0.003024 | 0.000000 |
| c1_mean_injection_0.25 | mean_pairwise_cosine | 0.759795 | 0.762652 | 0.002856 | -0.000168 |
| c1_mean_injection_0.25 | rankme | 267.286948 | 266.180828 | -1.106120 | -0.015511 |
| c1_mean_injection_0.25 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | -0.000000 |
| c1_mean_injection_0.5 | total_variance | 0.260057 | 0.257033 | -0.003024 | 0.000000 |
| c1_mean_injection_0.5 | mean_pairwise_cosine | 0.864222 | 0.865862 | 0.001640 | -0.001384 |
| c1_mean_injection_0.5 | rankme | 231.755392 | 230.631561 | -1.123831 | -0.033222 |
| c1_mean_injection_0.5 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | -0.000000 |
| c1_mean_injection_0.75 | total_variance | 0.260057 | 0.257033 | -0.003024 | 0.000000 |
| c1_mean_injection_0.75 | mean_pairwise_cosine | 0.973342 | 0.973659 | 0.000317 | -0.002707 |
| c1_mean_injection_0.75 | rankme | 123.283446 | 122.349288 | -0.934158 | 0.156451 |
| c1_mean_injection_0.75 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | -0.000000 |
| c1_mean_injection_0.9 | total_variance | 0.260057 | 0.257033 | -0.003024 | -0.000000 |
| c1_mean_injection_0.9 | mean_pairwise_cosine | 0.996803 | 0.996841 | 0.000038 | -0.002987 |
| c1_mean_injection_0.9 | rankme | 31.159275 | 30.839454 | -0.319821 | 0.770787 |
| c1_mean_injection_0.9 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | -0.000000 |
| c1_mean_injection_0.99 | total_variance | 0.260057 | 0.257033 | -0.003024 | 0.000000 |
| c1_mean_injection_0.99 | mean_pairwise_cosine | 0.999973 | 0.999974 | 0.000000 | -0.003024 |
| c1_mean_injection_0.99 | rankme | 2.015841 | 2.008168 | -0.007674 | 1.082935 |
| c1_mean_injection_0.99 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | -0.000000 |
| c1_isotropic_noise_0 | total_variance | 0.260057 | 0.257033 | -0.003024 | 0.000000 |
| c1_isotropic_noise_0 | mean_pairwise_cosine | 0.739943 | 0.742967 | 0.003024 | 0.000000 |
| c1_isotropic_noise_0 | rankme | 271.900193 | 270.809584 | -1.090609 | 0.000000 |
| c1_isotropic_noise_0 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | 0.000000 |
| c1_isotropic_noise_0.25 | total_variance | 0.371008 | 0.368069 | -0.002939 | 0.000085 |
| c1_isotropic_noise_0.25 | mean_pairwise_cosine | 0.666358 | 0.669037 | 0.002679 | -0.000345 |
| c1_isotropic_noise_0.25 | rankme | 490.466650 | 488.894616 | -1.572035 | -0.481426 |
| c1_isotropic_noise_0.25 | participation_ratio | 72.915571 | 71.443557 | -1.472014 | -0.375846 |
| c1_isotropic_noise_0.5 | total_variance | 1.259418 | 1.256656 | -0.002762 | 0.000262 |
| c1_isotropic_noise_0.5 | mean_pairwise_cosine | 0.371009 | 0.372492 | 0.001482 | -0.001542 |
| c1_isotropic_noise_0.5 | rankme | 616.196922 | 614.014116 | -2.182806 | -1.092198 |
| c1_isotropic_noise_0.5 | participation_ratio | 302.469114 | 299.941773 | -2.527342 | -1.431173 |
| c1_isotropic_noise_0.75 | total_variance | 9.256713 | 9.254537 | -0.002176 | 0.000849 |
| c1_isotropic_noise_0.75 | mean_pairwise_cosine | 0.074800 | 0.075112 | 0.000312 | -0.002712 |
| c1_isotropic_noise_0.75 | rankme | 660.006574 | 657.646954 | -2.359620 | -1.269011 |
| c1_isotropic_noise_0.75 | participation_ratio | 430.391117 | 427.359007 | -3.032110 | -1.935941 |
| c1_isotropic_noise_0.9 | total_variance | 81.237177 | 81.237263 | 0.000086 | 0.003111 |
| c1_isotropic_noise_0.9 | mean_pairwise_cosine | 0.009373 | 0.009417 | 0.000044 | -0.002980 |
| c1_isotropic_noise_0.9 | rankme | 668.560927 | 666.198029 | -2.362898 | -1.272290 |
| c1_isotropic_noise_0.9 | participation_ratio | 433.987976 | 430.917868 | -3.070109 | -1.973940 |
| c1_isotropic_noise_0.99 | total_variance | 9798.888506 | 9799.012919 | 0.124414 | 0.127438 |
| c1_isotropic_noise_0.99 | mean_pairwise_cosine | 0.000153 | 0.000156 | 0.000003 | -0.003022 |
| c1_isotropic_noise_0.99 | rankme | 669.671920 | 667.312649 | -2.359272 | -1.268663 |
| c1_isotropic_noise_0.99 | participation_ratio | 434.065202 | 430.993507 | -3.071695 | -1.975527 |
| c1_mean_interpolation_0 | total_variance | 0.260057 | 0.257033 | -0.003024 | 0.000000 |
| c1_mean_interpolation_0 | mean_pairwise_cosine | 0.739943 | 0.742967 | 0.003024 | 0.000000 |
| c1_mean_interpolation_0 | rankme | 271.900193 | 270.809584 | -1.090609 | 0.000000 |
| c1_mean_interpolation_0 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | 0.000000 |
| c1_mean_interpolation_0.25 | total_variance | 0.146282 | 0.144581 | -0.001701 | 0.001323 |
| c1_mean_interpolation_0.25 | mean_pairwise_cosine | 0.834468 | 0.836521 | 0.002053 | -0.000971 |
| c1_mean_interpolation_0.25 | rankme | 244.165302 | 243.017035 | -1.148267 | -0.057658 |
| c1_mean_interpolation_0.25 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | -0.000000 |
| c1_mean_interpolation_0.5 | total_variance | 0.065014 | 0.064258 | -0.000756 | 0.002268 |
| c1_mean_interpolation_0.5 | mean_pairwise_cosine | 0.918998 | 0.920027 | 0.001029 | -0.001995 |
| c1_mean_interpolation_0.5 | rankme | 197.065013 | 195.902648 | -1.162365 | -0.071756 |
| c1_mean_interpolation_0.5 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | -0.000000 |
| c1_mean_interpolation_0.75 | total_variance | 0.016254 | 0.016065 | -0.000189 | 0.002835 |
| c1_mean_interpolation_0.75 | mean_pairwise_cosine | 0.978533 | 0.978797 | 0.000264 | -0.002760 |
| c1_mean_interpolation_0.75 | rankme | 109.980500 | 109.066720 | -0.913780 | 0.176829 |
| c1_mean_interpolation_0.75 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | -0.000000 |
| c1_mean_interpolation_0.9 | total_variance | 0.002601 | 0.002570 | -0.000030 | 0.002994 |
| c1_mean_interpolation_0.9 | mean_pairwise_cosine | 0.996518 | 0.996559 | 0.000041 | -0.002983 |
| c1_mean_interpolation_0.9 | rankme | 33.277501 | 32.931321 | -0.346180 | 0.744428 |
| c1_mean_interpolation_0.9 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | -0.000000 |
| c1_mean_interpolation_0.99 | total_variance | 0.000026 | 0.000026 | -0.000000 | 0.003024 |
| c1_mean_interpolation_0.99 | mean_pairwise_cosine | 0.999965 | 0.999966 | 0.000000 | -0.003024 |
| c1_mean_interpolation_0.99 | rankme | 2.193568 | 2.184410 | -0.009158 | 1.081451 |
| c1_mean_interpolation_0.99 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | -0.000000 |
| c2_native_768 | total_variance | 0.260057 | 0.257033 | -0.003024 | 0.000000 |
| c2_native_768 | mean_pairwise_cosine | 0.739943 | 0.742967 | 0.003024 | 0.000000 |
| c2_native_768 | rankme | 271.900193 | 270.809584 | -1.090609 | 0.000000 |
| c2_native_768 | participation_ratio | 39.361108 | 38.264940 | -1.096169 | 0.000000 |
| c2_mrl_512 | total_variance | 0.255528 | 0.252496 | -0.003032 | -0.000008 |
| c2_mrl_512 | mean_pairwise_cosine | 0.744472 | 0.747504 | 0.003032 | 0.000008 |
| c2_mrl_512 | rankme | 252.011931 | 251.027998 | -0.983933 | 0.106675 |
| c2_mrl_512 | participation_ratio | 39.007475 | 37.957988 | -1.049487 | 0.046682 |
| c2_pca_512 | total_variance | 0.995252 | 0.995259 | 0.000006 | 0.003030 |
| c2_pca_512 | mean_pairwise_cosine | 0.004748 | 0.004741 | -0.000006 | -0.003030 |
| c2_pca_512 | rankme | 349.316750 | 348.632632 | -0.684118 | 0.406491 |
| c2_pca_512 | participation_ratio | 34.566550 | 33.818035 | -0.748515 | 0.347654 |
| c2_mrl_256 | total_variance | 0.234129 | 0.230969 | -0.003160 | -0.000136 |
| c2_mrl_256 | mean_pairwise_cosine | 0.765871 | 0.769031 | 0.003160 | 0.000136 |
| c2_mrl_256 | rankme | 150.567143 | 150.013509 | -0.553634 | 0.536975 |
| c2_mrl_256 | participation_ratio | 35.145667 | 34.139096 | -1.006571 | 0.089598 |
| c2_pca_256 | total_variance | 0.995384 | 0.995381 | -0.000003 | 0.003021 |
| c2_pca_256 | mean_pairwise_cosine | 0.004616 | 0.004619 | 0.000003 | -0.003021 |
| c2_pca_256 | rankme | 204.962847 | 204.882479 | -0.080368 | 1.010241 |
| c2_pca_256 | participation_ratio | 30.016292 | 29.405886 | -0.610405 | 0.485764 |
| c2_mrl_128 | total_variance | 0.156928 | 0.154881 | -0.002048 | 0.000976 |
| c2_mrl_128 | mean_pairwise_cosine | 0.843072 | 0.845119 | 0.002048 | -0.000976 |
| c2_mrl_128 | rankme | 72.183829 | 71.884861 | -0.298969 | 0.791640 |
| c2_mrl_128 | participation_ratio | 31.316158 | 30.515581 | -0.800577 | 0.295591 |
| c2_pca_128 | total_variance | 0.995726 | 0.995700 | -0.000026 | 0.002999 |
| c2_pca_128 | mean_pairwise_cosine | 0.004274 | 0.004300 | 0.000026 | -0.002999 |
| c2_pca_128 | rankme | 109.659276 | 109.668990 | 0.009714 | 1.100322 |
| c2_pca_128 | participation_ratio | 23.395414 | 22.942231 | -0.453183 | 0.642986 |

## C3 numerical summary changes

[ours] Counts retain their respective 1000/983 denominators; fixed-query identity
comparisons are in gallery_diagnostic.md. Bounds remain empirical checks.

| Condition | Metric | Historical | Reduced | Absolute change |
| --- | --- | --- | --- | --- |
| c3_native_ref | identity_changed_queries | 0.000000 | 0.000000 | 0.000000 |
| c3_native_ref | max_arithmetic_residual | 0.000000 | 0.000000 | 0.000000 |
| c3_native_ref | max_score_error | 0.000000 | 0.000000 | 0.000000 |
| c3_native_ref | mean_max_score_error | 0.000000 | 0.000000 | 0.000000 |
| c3_native_ref | mean_overlap | 1.000000 | 1.000000 | 0.000000 |
| c3_native_ref | prospective_empirical_violations | 0.000000 | 0.000000 | 0.000000 |
| c3_native_ref | prospective_sufficient_queries | 1000.000000 | 983.000000 | -17.000000 |
| c3_native_ref | retrospective_sufficient_queries | 1000.000000 | 983.000000 | -17.000000 |
| c3_native_fp16_gallery | identity_changed_queries | 5.000000 | 5.000000 | 0.000000 |
| c3_native_fp16_gallery | max_arithmetic_residual | 0.000000 | 0.000000 | 0.000000 |
| c3_native_fp16_gallery | max_score_error | 0.000060 | 0.000058 | -0.000002 |
| c3_native_fp16_gallery | mean_max_score_error | 0.000043 | 0.000043 | -0.000000 |
| c3_native_fp16_gallery | mean_overlap | 0.999500 | 0.999491 | -0.000009 |
| c3_native_fp16_gallery | prospective_empirical_violations | 0.000000 | 0.000000 | 0.000000 |
| c3_native_fp16_gallery | prospective_sufficient_queries | 592.000000 | 576.000000 | -16.000000 |
| c3_native_fp16_gallery | retrospective_sufficient_queries | 938.000000 | 921.000000 | -17.000000 |
| c3_native_fp16_both | identity_changed_queries | 5.000000 | 5.000000 | 0.000000 |
| c3_native_fp16_both | max_arithmetic_residual | 0.000000 | 0.000000 | 0.000000 |
| c3_native_fp16_both | max_score_error | 0.000102 | 0.000082 | -0.000020 |
| c3_native_fp16_both | mean_max_score_error | 0.000053 | 0.000051 | -0.000002 |
| c3_native_fp16_both | mean_overlap | 0.999500 | 0.999491 | -0.000009 |
| c3_native_fp16_both | prospective_empirical_violations | 0.000000 | 0.000000 | 0.000000 |
| c3_native_fp16_both | prospective_sufficient_queries | 412.000000 | 398.000000 | -14.000000 |
| c3_native_fp16_both | retrospective_sufficient_queries | 916.000000 | 903.000000 | -13.000000 |
| c3_native_int8_gallery | identity_changed_queries | 60.000000 | 60.000000 | 0.000000 |
| c3_native_int8_gallery | max_arithmetic_residual | 0.000000 | 0.000000 | 0.000000 |
| c3_native_int8_gallery | max_score_error | 0.003952 | 0.003952 | 0.000000 |
| c3_native_int8_gallery | mean_max_score_error | 0.002356 | 0.002363 | 0.000007 |
| c3_native_int8_gallery | mean_overlap | 0.994000 | 0.993896 | -0.000104 |
| c3_native_int8_gallery | prospective_empirical_violations | 0.000000 | 0.000000 | 0.000000 |
| c3_native_int8_gallery | prospective_sufficient_queries | 0.000000 | 0.000000 | 0.000000 |
| c3_native_int8_gallery | retrospective_sufficient_queries | 32.000000 | 20.000000 | -12.000000 |
| c3_native_int8_both | identity_changed_queries | 75.000000 | 75.000000 | 0.000000 |
| c3_native_int8_both | max_arithmetic_residual | 0.000000 | 0.000000 | 0.000000 |
| c3_native_int8_both | max_score_error | 0.004817 | 0.004817 | 0.000000 |
| c3_native_int8_both | mean_max_score_error | 0.002429 | 0.002434 | 0.000005 |
| c3_native_int8_both | mean_overlap | 0.992500 | 0.992370 | -0.000130 |
| c3_native_int8_both | prospective_empirical_violations | 0.000000 | 0.000000 | 0.000000 |
| c3_native_int8_both | prospective_sufficient_queries | 0.000000 | 0.000000 | 0.000000 |
| c3_native_int8_both | retrospective_sufficient_queries | 34.000000 | 23.000000 | -11.000000 |
| c3_native_fp32_arithmetic | identity_changed_queries | 0.000000 | 0.000000 | 0.000000 |
| c3_native_fp32_arithmetic | max_arithmetic_residual | 0.000001 | 0.000001 | 0.000000 |
| c3_native_fp32_arithmetic | max_score_error | 0.000001 | 0.000001 | 0.000000 |
| c3_native_fp32_arithmetic | mean_max_score_error | 0.000001 | 0.000001 | -0.000000 |
| c3_native_fp32_arithmetic | mean_overlap | 1.000000 | 1.000000 | 0.000000 |
| c3_native_fp32_arithmetic | prospective_empirical_violations | unscorable | unscorable | unscorable |
| c3_native_fp32_arithmetic | prospective_sufficient_queries | unscorable | unscorable | unscorable |
| c3_native_fp32_arithmetic | retrospective_sufficient_queries | 999.000000 | 982.000000 | -17.000000 |
| c3_mean99_ref | identity_changed_queries | 0.000000 | 0.000000 | 0.000000 |
| c3_mean99_ref | max_arithmetic_residual | 0.000000 | 0.000000 | 0.000000 |
| c3_mean99_ref | max_score_error | 0.000000 | 0.000000 | 0.000000 |
| c3_mean99_ref | mean_max_score_error | 0.000000 | 0.000000 | 0.000000 |
| c3_mean99_ref | mean_overlap | 1.000000 | 1.000000 | 0.000000 |
| c3_mean99_ref | prospective_empirical_violations | 0.000000 | 0.000000 | 0.000000 |
| c3_mean99_ref | prospective_sufficient_queries | 1000.000000 | 983.000000 | -17.000000 |
| c3_mean99_ref | retrospective_sufficient_queries | 1000.000000 | 983.000000 | -17.000000 |
| c3_mean99_fp16_gallery | identity_changed_queries | 195.000000 | 195.000000 | 0.000000 |
| c3_mean99_fp16_gallery | max_arithmetic_residual | 0.000000 | 0.000000 | 0.000000 |
| c3_mean99_fp16_gallery | max_score_error | 0.000000 | 0.000000 | 0.000000 |
| c3_mean99_fp16_gallery | mean_max_score_error | 0.000000 | 0.000000 | -0.000000 |
| c3_mean99_fp16_gallery | mean_overlap | 0.980300 | 0.979959 | -0.000341 |
| c3_mean99_fp16_gallery | prospective_empirical_violations | 0.000000 | 0.000000 | 0.000000 |
| c3_mean99_fp16_gallery | prospective_sufficient_queries | 0.000000 | 0.000000 | 0.000000 |
| c3_mean99_fp16_gallery | retrospective_sufficient_queries | 45.000000 | 34.000000 | -11.000000 |

## Matched-D MRL minus PCA

| D | Attribute | Endpoint | Old difference pp | New difference pp | Difference change pp |
| --- | --- | --- | --- | --- | --- |
| 512 | weather | p10 | +1.9600 | +1.9939 | +0.0339 |
| 512 | weather | macro_p10 | +0.8365 | +0.7869 | -0.0495 |
| 512 | weather | chance_sum_p_squared | +0.0000 | +0.0000 | +0.0000 |
| 512 | weather | majority_floor | +0.0000 | +0.0000 | +0.0000 |
| 512 | weather | balanced_constant_floor | +0.0000 | +0.0000 | +0.0000 |
| 512 | weather | probe_standardized_accuracy | +5.0000 | +4.8830 | -0.1170 |
| 512 | weather | probe_standardized_balanced_accuracy | +7.1893 | +7.4608 | +0.2715 |
| 512 | weather | probe_standardized_macro_f1 | +6.9477 | +6.6461 | -0.3016 |
| 512 | scene | p10 | +0.4300 | +0.4374 | +0.0074 |
| 512 | scene | macro_p10 | -0.7131 | -0.7451 | -0.0319 |
| 512 | scene | chance_sum_p_squared | +0.0000 | +0.0000 | +0.0000 |
| 512 | scene | majority_floor | +0.0000 | +0.0000 | +0.0000 |
| 512 | scene | balanced_constant_floor | +0.0000 | +0.0000 | +0.0000 |
| 512 | scene | probe_standardized_accuracy | +2.7000 | +3.1536 | +0.4536 |
| 512 | scene | probe_standardized_balanced_accuracy | +3.5484 | +4.6648 | +1.1164 |
| 512 | scene | probe_standardized_macro_f1 | +4.0619 | +4.8027 | +0.7408 |
| 512 | timeofday | p10 | -0.4300 | -0.4273 | +0.0027 |
| 512 | timeofday | macro_p10 | +1.0621 | +1.1237 | +0.0616 |
| 512 | timeofday | chance_sum_p_squared | +0.0000 | +0.0000 | +0.0000 |
| 512 | timeofday | majority_floor | +0.0000 | +0.0000 | +0.0000 |
| 512 | timeofday | balanced_constant_floor | +0.0000 | +0.0000 | +0.0000 |
| 512 | timeofday | probe_standardized_accuracy | +0.7000 | +0.7121 | +0.0121 |
| 512 | timeofday | probe_standardized_balanced_accuracy | +1.9315 | +1.6281 | -0.3034 |
| 512 | timeofday | probe_standardized_macro_f1 | +2.4057 | +2.1404 | -0.2653 |
| 256 | weather | p10 | +1.2600 | +1.2920 | +0.0320 |
| 256 | weather | macro_p10 | +0.1867 | +0.1290 | -0.0577 |
| 256 | weather | chance_sum_p_squared | +0.0000 | +0.0000 | +0.0000 |
| 256 | weather | majority_floor | +0.0000 | +0.0000 | +0.0000 |
| 256 | weather | balanced_constant_floor | +0.0000 | +0.0000 | +0.0000 |
| 256 | weather | probe_standardized_accuracy | +1.0000 | +0.8138 | -0.1862 |
| 256 | weather | probe_standardized_balanced_accuracy | +2.0543 | +2.2108 | +0.1565 |
| 256 | weather | probe_standardized_macro_f1 | +1.3169 | +0.9556 | -0.3613 |
| 256 | scene | p10 | +0.2300 | +0.2238 | -0.0062 |
| 256 | scene | macro_p10 | -0.6390 | -0.6801 | -0.0411 |
| 256 | scene | chance_sum_p_squared | +0.0000 | +0.0000 | +0.0000 |
| 256 | scene | majority_floor | +0.0000 | +0.0000 | +0.0000 |
| 256 | scene | balanced_constant_floor | +0.0000 | +0.0000 | +0.0000 |
| 256 | scene | probe_standardized_accuracy | +2.3000 | +2.5432 | +0.2432 |
| 256 | scene | probe_standardized_balanced_accuracy | +2.8675 | +3.5292 | +0.6617 |
| 256 | scene | probe_standardized_macro_f1 | +3.1701 | +3.4976 | +0.3275 |
| 256 | timeofday | p10 | -0.9300 | -0.9257 | +0.0043 |
| 256 | timeofday | macro_p10 | +1.1986 | +1.2839 | +0.0853 |
| 256 | timeofday | chance_sum_p_squared | +0.0000 | +0.0000 | +0.0000 |
| 256 | timeofday | majority_floor | +0.0000 | +0.0000 | +0.0000 |
| 256 | timeofday | balanced_constant_floor | +0.0000 | +0.0000 | +0.0000 |
| 256 | timeofday | probe_standardized_accuracy | -1.5000 | -1.5259 | -0.0259 |
| 256 | timeofday | probe_standardized_balanced_accuracy | +1.4742 | +1.5715 | +0.0973 |
| 256 | timeofday | probe_standardized_macro_f1 | +0.0494 | -0.0429 | -0.0923 |
| 128 | weather | p10 | +0.1200 | +0.1424 | +0.0224 |
| 128 | weather | macro_p10 | -2.5538 | -2.6798 | -0.1260 |
| 128 | weather | chance_sum_p_squared | +0.0000 | +0.0000 | +0.0000 |
| 128 | weather | majority_floor | +0.0000 | +0.0000 | +0.0000 |
| 128 | weather | balanced_constant_floor | +0.0000 | +0.0000 | +0.0000 |
| 128 | weather | probe_standardized_accuracy | -2.8000 | -2.7467 | +0.0533 |
| 128 | weather | probe_standardized_balanced_accuracy | -7.3036 | -7.4489 | -0.1453 |
| 128 | weather | probe_standardized_macro_f1 | -6.9108 | -6.8930 | +0.0178 |
| 128 | scene | p10 | -1.4100 | -1.4649 | -0.0549 |
| 128 | scene | macro_p10 | -2.5931 | -2.7148 | -0.1218 |
| 128 | scene | chance_sum_p_squared | +0.0000 | +0.0000 | +0.0000 |
| 128 | scene | majority_floor | +0.0000 | +0.0000 | +0.0000 |
| 128 | scene | balanced_constant_floor | +0.0000 | +0.0000 | +0.0000 |
| 128 | scene | probe_standardized_accuracy | -1.2000 | -1.4242 | -0.2242 |
| 128 | scene | probe_standardized_balanced_accuracy | -1.3079 | -1.6735 | -0.3656 |
| 128 | scene | probe_standardized_macro_f1 | -2.0408 | -1.9574 | +0.0834 |
| 128 | timeofday | p10 | -3.2700 | -3.3367 | -0.0667 |
| 128 | timeofday | macro_p10 | -1.0792 | -1.1064 | -0.0272 |
| 128 | timeofday | chance_sum_p_squared | +0.0000 | +0.0000 | +0.0000 |
| 128 | timeofday | majority_floor | +0.0000 | +0.0000 | +0.0000 |
| 128 | timeofday | balanced_constant_floor | +0.0000 | +0.0000 | +0.0000 |
| 128 | timeofday | probe_standardized_accuracy | +0.0000 | +0.2035 | +0.2035 |
| 128 | timeofday | probe_standardized_balanced_accuracy | -3.9422 | -3.9808 | -0.0386 |
| 128 | timeofday | probe_standardized_macro_f1 | -3.5705 | -3.2706 | +0.2999 |
