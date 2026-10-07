# C1 protocol hardening — cached pilot vectors only

[measurement] `investigation.json` records every fitted model, including inner
selection candidates. No neural inference was performed. SHA256 checks in
`pilot_preservation.json` verify that the historical pilot files are unchanged.

[measurement] At positive scale 0.01, objective-equivalent unscaled C values
restore pristine validation accuracy: weather C=1e6, accuracy 0.7578125;
scene C=1e5, accuracy 0.8046875; timeofday C=1e6, accuracy 0.8828125.
Values through 1e9 were tested. Weather fits above 1e6 did not converge; the
scene fit at 1e7 did not converge. All fitted values and scores are retained.

[interpretation] Scaling X by a changes the equivalent penalized logistic
objective to C/a². Main's secondary unscaled grid therefore adds one decade,
ending at 1e6. It does not adopt the arbitrarily large investigated grid or
silently introduce scale-aware regularization. Boundary fits remain qualified.
The standardized primary protocol is unchanged.

[measurement] Severe mean injection reaches 5000 iterations for raw weather
and scene; timeofday converges at 4989. Standardized fits converge in 70, 161
and 58 iterations respectively. Centering only at the same C gives 250, 108
and 139 iterations, with unchanged raw validation accuracies. The augmented
nonzero-singular-value design condition squared increases from 1.09e6 to
6.08e9 under injection and drops to 6.21e5 after centering.

[interpretation] The common offset creates numerical conditioning problems in
the secondary raw probe, while the centered/standardized controls preserve
separability. max_iter stays 5000; warnings remain outputs rather than evidence
of semantic failure. Centering-only is an investigative control, not a changed
main secondary probe.

[measurement] Four-direction weather fits converge in equivalent orthonormal
coordinates. Weak regularization (C=1e4 and 1e6) gives training accuracy
0.6171875 and balanced accuracy 0.187831, versus full-vector training accuracy
0.997396 at C=100. Their validation accuracy is 0.5546875.

[interpretation] The weak four-direction fit cannot be explained solely by the
original C ceiling or convergence failure. Its historical underfit flag does
not identify a unique cause or prove that all weather information disappeared.
Main's existing severity grid retains eight directions at its most severe
rank setting; the pilot-specific four-direction result is not a main hypothesis.

[measurement] Verification: 401 tests passed, one optional accelerator test
skipped; Ruff passed. No main embedding extraction has begun.

[open] Main fit boundaries/warnings must remain qualified. The repository's
paired training-seed intervals are unavailable for one fixed pretrained encoder;
the main protocol reports paired descriptive effects without manufactured seeds.
