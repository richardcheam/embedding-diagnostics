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
BDD100K and evaluation definitions, final results, limits with next steps, bibliography, and an appendix. BDD100K, JEPA, the interventions, linear probing, and retrieval P@10
are explained before their results. The architecture and condition matrix are directly visible in the Approach section;
The appendix follows the bibliography and keeps both technical sections expandable.
The optional dimensionality section defines the matrix, derives the soft counts,
and works a clearly labelled illustrative example. The portfolio page reports the final method; the linked research status preserves the evaluation history.

The contents list remains beside the report on desktop, with a scroll-tracked current
section. Smaller screens use a sticky contents control; links work without JavaScript.
Shared architecture, BDD100K results, paired uncertainty, and dimensionality
explanations have direct entries. Contents links open the optional sections,
including when loaded from a direct URL.

The interactive figures show all seven conditions for each driving attribute, with
five seed markers and a mean per row. The table explanation is generated from the
same endpoints. The case study presents BDD100K only; historical experiment records remain in the repository.

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
- The public build requires only the 35 BDD100K runs and publishes no other dataset.
  The page does not claim a reproduction of the full LeJEPA system.

The architecture figure explores all seven conditions through Forward, Loss, Backward,
and Update. It traces computation in one optimizer step, not measured training dynamics.
Motion runs only while the figure is visible on wider screens; explicit pause persists
across scrolling. Reduced-motion and phone views retain manual phase selection. Without
JavaScript, the controls are hidden and the labelled schematic remains readable.

The static evidence table remain readable without JavaScript. Interactive charts allow
readers to compare the three BDD100K scenario attributes.

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

Paired uncertainty is presented as static forest plots inside an expandable section,
with five seed differences, mean diamonds, 95% t intervals, and a zero reference.
Each row names both contrast arms. Classification and retrieval have separate axes,
kept fixed across attributes; the attribute selectors stay synchronized. The plots
remain available for time of day without JavaScript. Narrative warnings use muted
iron-oxide red; findings use malachite green. Interval signs carry no colour verdict.

Controls share a warm-paper fill, pigment borders, ochre state marks and chevrons.
Select elements retain native keyboard and phone pickers; disclosures remain
native details/summary elements. Controls have visible focus and at least 44px
hit targets, with two-by-two architecture step controls on narrow phones.
