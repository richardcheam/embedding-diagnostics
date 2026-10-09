# Wrap-up verification and review

Pre-publication verification recorded for wrap-up `994937b`, based on `e902aa0`.
Experimental scope closed; no endpoints recomputed.

| Check | Observed result |
|---|---|
| `uv run pytest` | 539 passed, 1 optional skip |
| `uv run ruff check .` | Passed |
| `uv lock --check` | Passed; root dependency files unchanged |
| `python3 viz/build_site.py --report-out report/sections/final_tables.tex` | Passed; all four generated report tables match their checked-in versions |
| Accepted JSON reader | Source SHA-256 and individual record hashes match |
| Preserved caches/history | 11,003 COCO binding files, 750 BDD chunks, 491 historical artifact hashes match |
| Protected tree | No `experiments/`, `src/`, `pyproject.toml` or `uv.lock` diff from baseline |
| Links | All packaged files/fragments and updated Markdown links resolve; six unique external page links returned HTTP 200 |
| Firefox 157.0.1 | 1440/768/500px outer viewports and 390px same-origin phone frame; no page-width overflow; four historical charts load |
| Keyboard | Skip anchor, mobile menu expand/Tab/activate/close, attribute select, horizontal table scrolling |
| Fresh review | Numerical quotations match accepted records; report conclusion/future publication wording corrected |
| LaTeX PDF | Not compiled: no `latexmk`, `pdflatex` or `tectonic`; inputs and generated tables checked |

The phone frame exercises actual 390px CSS layout in Firefox; its host window is
larger because headless Firefox imposes a 500px outer-window minimum. It is not a
physical-device test. Screenshots were visually inspected. This is a bounded
browser and keyboard check, not comprehensive screen-reader certification.

Preview: [http://127.0.0.1:8080/](http://127.0.0.1:8080/), served locally from ignored
`viz/dist/`. Rebuild/serve instructions are in [viz/README](../viz/README.md).
Browser measurements, external-link statuses and screenshots remain there for
review. The page keeps its existing typography, paper/pigment palette, responsive
contents, shared-architecture illustration and training uncertainty appendix.

At the pre-publication verification stage, the new page was not live and no push,
publication or deployment had been performed. The subsequent user-authorized release
uses GitHub Pages; this historical verification record remains preserved.
Historical scientific records, failed attempts and canonical arrays are unchanged.
Experimental expansion is closed; maintenance and transparent corrections remain
possible. The audio–visual project is separate.
