# Public project page

The current case study is hosted at
[richardcheam.github.io/embedding-diagnostics/](https://richardcheam.github.io/embedding-diagnostics/).

```sh
python3 viz/build_site.py
python3 -m http.server 8080 --directory viz/dist
```

Open `http://localhost:8080/`. Building uses only Python's standard library. The static page
uses relative URLs, local SVG/JavaScript/CSS, and no external fonts or animation services.
Generated output is ignored by Git.

## Report structure

The page introduces the problem before the measurements: motivation, controlled approach,
datasets and evaluation definitions, corrected results, the measurement audit, and limits
with next steps. Dataset names, JEPA, the interventions, linear probing, and retrieval P@10
are explained before their results. Architecture details and rank formulas are optional
expandable material. The scroll audit follows the results rather than acting as the introduction.

## Evidence and publication safeguards

- Geometry and retrieval: final step-4,000 records from `experiments/phaseB_s*/**/metrics.jsonl`.
- Corrected probes: `metrics_reprobed.jsonl` alongside each run. No fallback to original probes.
- All seven conditions and all five seeds are required. Identity, split, and variance
  consistency are checked before publication. The variance check accounts for the original
  sample-covariance versus reprobe population-variance convention.
- Paired intervals use the contrast definitions and Student-t quantiles in
  `src/embedding_diagnostics/stats.py`. They cover seed variation on a fixed evaluation
  split, without multiplicity correction or verdict language.
- Convergence, iteration counts, selected regularization, and underfit flags are retained
  in the downloadable `data.json`. Corrected fitting is post-hoc; iteration caps remain disclosed.
- Phase-A geometry is published from final logs, separately from the corrected Phase-B evidence. Phase-A probe endpoints are uncorrected and excluded. The page does not claim a
  reproduction of the full LeJEPA system.

The scroll scene changes emphasis between measured endpoints; it does not interpolate
accuracy or simulate training. Mobile and reduced-motion views retain the full storyboard.
The static table and audit remain readable without JavaScript.

## GitHub Pages

In repository **Settings → Pages**, choose **GitHub Actions** as the source. The
`Publish project page` workflow builds and uploads `viz/dist`, then deploys on `main` or
manual dispatch. Pull requests build the artifact without publishing. Deployment uses
the `github-pages` environment and narrowly scoped `pages: write` / `id-token: write`
permissions. The training environment and saved encoder files are not needed for hosting.

## Historical demo

`build_report.py` and `embedding_diagnostics.report_html` retain the earlier narrative.
They are not used or published by this workflow. The public page uses `build_site.py` and
`site/` exclusively.
