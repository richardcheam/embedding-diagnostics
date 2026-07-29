# jepa-lens — Design

Date: 2026-07-29
Status: approved, pre-implementation

## Summary

`jepa-lens` tests whether stop-gradient still matters under SIGReg once training runs
long enough, and whether cheap collapse diagnostics detect the problem earlier than an
expensive linear probe does. It ships three artifacts: the experiment code, a LaTeX
technical report, and a self-contained interactive visualization of the results.

## Provenance statement

This project is mostly assembly of existing work. Stating that up front so no part of
the repository implies otherwise.

This table is the source of truth for the report's contributions section
(`report/sections/contributions.tex`). There is no standalone `PROVENANCE.md`; the paper
is where these claims live.

| Component | Status | Source |
| --- | --- | --- |
| JEPA architecture (context/target encoder + predictor) | Established | I-JEPA, [arXiv 2301.08243](https://arxiv.org/pdf/2301.08243) |
| SIGReg loss | Established, not ours | LeJEPA, [arXiv 2511.08544](https://huggingface.co/papers/2511.08544); reference impl [rbalestr-lab/lejepa](https://github.com/rbalestr-lab/lejepa) |
| EMA target encoder + stop-gradient | Established | BYOL / SimSiam lineage, used by I-JEPA and V-JEPA |
| Collapse diagnostics (variance, effective rank, pairwise cosine) | Established practice | Standard in SSL literature; predates this project |
| Linear-probe evaluation protocol | Established practice | Standard SSL evaluation |
| 4-condition ablation grid | Standard methodology | Not novel; conventional ablation design |
| Scale-dependence of stop-gradient under SIGReg | **Replication attempt** | Encountered secondhand via search summary (reported ~-0.4pp at small scale vs ~-13.1pp at 20k steps). Primary source not read end-to-end at time of writing. Treat as unverified. |
| Cheap diagnostics as an early-warning signal for probe degradation | **Potentially new; conditional** | Only a contribution if the timing gap is observed. May not exist. |

Rules that apply for the life of the project:

1. Every claim in the README, report, and visualization carries one of four labels:
   `[established]`, `[replication]`, `[ours]`, `[interpretation]`. In LaTeX these are
   macros (`\established{}`, `\replication{}`, `\ours{}`, `\interp{}`) defined in
   `report/macros.tex`, so labelling is mechanical rather than remembered.
2. Code that reimplements a published method cites the paper in its module docstring.
3. A negative result is reported as prominently as a positive one. If the replication
   fails, the report says the replication failed.
4. No figure ships without its label attached, so it cannot circulate stripped of caveat.
   Figure captions carry the label macro.

## Research question

Split into two parts, tested in sequence. Part 2 is only meaningful if Part 1 finds a gap.

**Part 1 (replication).** LeJEPA reports that SIGReg removes the need for heuristics such
as stop-gradient and EMA teachers. The secondhand result above suggests that claim may
hold only for short runs, with a large gap appearing by ~20k steps. Does the gap between
`sigreg_stopgrad` and `sigreg_nostopgrad` widen with training duration?

"Scale" here means **training duration in optimizer steps**. Model size and dataset size
are held fixed so duration is the only varying axis.

**Part 2 (potentially new).** If the gap appears, do the cheap diagnostics (embedding
variance, effective rank, pairwise cosine) start moving *before* linear-probe accuracy
drops? A lead time would make them usable as an early-warning signal, letting a doomed run
be killed without waiting for probe evaluation.

Both outcomes are publishable within the project. Curves that stay glued together through
the full budget is a clean negative result for Part 1 and is reported as such.

## Experimental design

### Conditions

Four conditions. Identical architecture, data, seed handling, and optimizer. Only the
collapse-prevention mechanism differs.

| Condition | EMA target | Target detached | Regularizer | Role |
| --- | --- | --- | --- | --- |
| `ema_stopgrad` | yes | yes | none | Classic JEPA baseline |
| `sigreg_stopgrad` | no | yes | SIGReg | Both mechanisms |
| `sigreg_nostopgrad` | no | no | SIGReg | LeJEPA's actual recipe; the condition under test |
| `none_nostopgrad` | no | no | none | Sanity control; expected to collapse |

`none_nostopgrad` removes every collapse-prevention mechanism at once — one shared encoder,
no detachment, no regularizer — so nothing prevents the constant solution. It exists to
prove the harness can detect collapse when collapse is present. If it does not collapse,
the measurement apparatus is suspect and results from the other three conditions cannot be
trusted.

**Revised during planning.** This control was originally specified as `ema_nostopgrad` (EMA
target, no stop-gradient). That combination is inert: an EMA target encoder's parameters sit
outside autograd, so the target branch reaches no trainable parameter and disabling
stop-gradient changes nothing — the condition would have silently duplicated `ema_stopgrad`
while appearing to be a control. Removing collapse prevention requires removing the
teacher–student split too.

### Model and data

- **Dataset:** CIFAR-10. Chosen for iteration speed on a single GPU and because it is
  conventional for small-scale SSL ablation. The SSL pretraining split is disjoint from
  the probe's labelled split.
- **Encoder:** small ViT, patch size 4, ~6 blocks, target ~5-10M parameters. Same family
  as I-JEPA, scaled to fit a single GPU comfortably.
- **Predictor:** small MLP.
- **Masking:** I-JEPA-style block masking for context/target construction.
- **Budget:** ~32k steps per condition, single run each.

### Measurement

One long run per condition with diagnostics checkpointed at intervals, rather than
separate short and long runs. This yields the full step-vs-metric curve at lower cost.

At each checkpoint, logged per condition:

- Embedding variance (per-dimension standard deviation, summarized)
- Effective rank of the embedding covariance
- Pairwise cosine similarity distribution
- Linear-probe accuracy — frozen encoder, logistic regression on the held-out labelled
  split
- A 2D PCA projection of a fixed sample of embeddings, for the animated visualization

Training loss is logged but is explicitly **not** treated as a collapse metric.
Regularized objectives plateau at non-zero values whether or not the encoder collapsed.

### Reading the result

- **Part 1 positive:** `sigreg_nostopgrad` probe accuracy separates from
  `sigreg_stopgrad` as steps increase, with the gap growing.
- **Part 1 negative:** the two track each other through 32k steps. Reported as a
  non-replication at this scale, with the caveat that 32k may be below the threshold
  where the effect appears.
- **Part 2:** measured only if Part 1 is positive. Compare the step at which each cheap
  diagnostic departs from its baseline trajectory against the step at which probe
  accuracy departs. A consistent lead time is the finding.

Single-seed runs mean an observed gap could be seed noise. The report must state this
limitation. Multi-seed replication is out of scope for the first pass and listed as
future work.

## Architecture

```
jepa-lens/
  README.md                      # lean: what it is, how to run, headline result, links
  AGENTS.md                      # conventions for agentic coding sessions
  pyproject.toml
  configs/
    base.yaml                    # shared: model, data, optimizer, schedule
    conditions/
      ema_stopgrad.yaml          # each overrides only the strategy block
      sigreg_stopgrad.yaml
      sigreg_nostopgrad.yaml
      none_nostopgrad.yaml
  src/jepa_lens/
    data.py                      # CIFAR-10 loading, context/target view construction
    models/
      vit.py                     # small ViT encoder
      predictor.py               # MLP predictor
    training/
      strategy.py                # CollapsePreventionStrategy ABC + 4 implementations
      trainer.py                 # single shared training loop
    diagnostics/
      metrics.py                 # variance, effective rank, pairwise cosine
      probe.py                   # linear probe evaluation
      projection.py              # 2D PCA of embeddings for the animation
    logging_utils.py             # JSONL run logger
  scripts/
    train.py                     # run one condition
    run_all_conditions.py        # run the full grid
    make_figures.py              # JSONL -> report/figures/*.pdf (matplotlib)
  viz/
    build_report.py              # JSONL -> self-contained interactive HTML
  report/
    main.tex                     # preprint, article class
    macros.tex                   # \established \replication \ours \interp
    refs.bib
    sections/
      introduction.tex
      contributions.tex          # the provenance table, as the contributions section
      related_work.tex
      method.tex
      experiments.tex
      results.tex
      limitations.tex
      conclusion.tex
    figures/                     # generated; gitignored except a .gitkeep
    Makefile                     # latexmk build
  docs/                          # design docs, notes; not the report
  tests/                         # CPU-only, synthetic tensors, fast
  experiments/                   # metrics.jsonl + resolved config per run (tracked)
  .github/workflows/ci.yml       # lint + CPU tests
```

### Strategy pattern

All four conditions run through one `Trainer`. The swappable piece is a
`CollapsePreventionStrategy` with three responsibilities:

- `compute_loss(prediction, target_latent)` — prediction loss plus regularizer if any
- `post_step_update(context_encoder, target_encoder)` — EMA update, or no-op
- `detaches_target` — whether the target branch is detached

Rationale: if each condition had its own training loop, a difference in results could come
from an incidental difference in the loop rather than from the mechanism under test. One
loop plus a narrow interface makes the comparison structurally honest. This is a
correctness requirement, not a style preference.

### Local/GPU workflow

Development happens on a machine without a GPU; training happens on a separate GPU machine.
Sync is over git.

- `experiments/**/metrics.jsonl` and resolved configs are **tracked in git**. They are
  small.
- `*.pt` checkpoints are **gitignored**. They are not needed to build the report.

So: push code, pull on the GPU box, train, push results, pull locally, rebuild the report.
No separate sync tooling.

### Tests

CPU-only, synthetic tensors, fast enough to run on every change. Their job is catching
wiring bugs before GPU time is spent, not validating research claims. Coverage targets:

- Each strategy produces finite loss and correct gradient-flow behaviour — in particular
  that `detaches_target` actually detaches, verified by gradient presence on target params
- Diagnostics return known values on constructed inputs (a constant matrix has effective
  rank near 1; an orthogonal matrix does not)
- Config loading resolves condition overrides onto the base correctly
- The JSONL logger round-trips

## Deliverables

### README (lean)

What the project is, a two-line honesty note (replication attempt + conditional
contribution) pointing at the report's contributions section, setup and run instructions,
the headline result with its label, and links to the compiled PDF and the visualization.
Nothing else. Detail belongs in the report.

### Technical report (`report/`, LaTeX)

A preprint-style paper built with `latexmk`, article class. Not a conference template —
kept dependency-free and swappable if it is ever submitted somewhere with a required style.

Section plan:

- **Introduction** — what JEPA collapse is, why the heuristics question matters now
- **Contributions** — the provenance table rendered as prose plus a table: what is
  established, what is a replication attempt, what is conditionally new. This section
  replaces the standalone provenance file and is written before results exist.
- **Related work** — I-JEPA, V-JEPA, BYOL/SimSiam lineage, LeJEPA/SIGReg
- **Method** — architecture, the four conditions, the strategy abstraction, why one shared
  training loop is a correctness requirement
- **Experiments** — dataset, hyperparameters, budget, measurement protocol, the
  `none_nostopgrad` sanity control and what its failure would invalidate
- **Results** — Part 1 replication outcome, then Part 2 only if Part 1 is positive
- **Limitations** — single seed, single dataset, single model scale, 32k-step ceiling,
  and the fact that a null result does not disprove the effect at larger scale
- **Conclusion**

Figures are generated from `experiments/**/metrics.jsonl` by `scripts/make_figures.py`
into `report/figures/` as PDFs. The report therefore rebuilds from experiment data rather
than containing hand-placed images, and figures cannot silently drift from the runs that
produced them. Figure files are gitignored; the JSONL that generates them is tracked.

`report/figures/` and the interactive HTML consume the same JSONL through two renderers.
Neither is the source of truth; the run logs are.

Build: `make -C report`, wrapping `latexmk -pdf`. Verified available on the development
machine (`latexmk` and `pdflatex` both present). The PDF build stays out of CI — a TeX Live
install would dominate CI time for little benefit, and the report is built locally where it
is written. CI covers lint and CPU tests only.

### Interactive visualization (`viz/build_report.py` output)

A single self-contained HTML file with run data embedded inline. No server, works offline,
shareable as a link. Contents:

- Step scrubber; all four conditions update together
- Animated 2D embedding projection per checkpoint — the point cloud visibly contracting
  is the clearest expression of collapse
- Per-condition toggles
- Divergence markers: where cheap diagnostics start moving vs. where probe accuracy starts
  dropping, which is Part 2 made visual
- Claim labels rendered on the page itself

Chosen over a Streamlit app so that the artifact developed against and the artifact shared
are the same file, with no drift between them. Trade-off accepted: no live monitoring
during training; run logs cover that need.

## Out of scope

- Multi-seed runs (listed as future work; single-seed limitation stated in the report)
- Datasets beyond CIFAR-10
- Model scales beyond the single small ViT
- Beating any benchmark. This is a diagnostic study.
- Reusing `jepa-learning` code. That repository's constraints (synthetic data only, no GPU
  training) are incompatible with this project. Concepts carry over; code does not.
