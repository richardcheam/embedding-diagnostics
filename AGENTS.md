# Working in this repository

## What this is

A four-condition ablation harness testing JEPA collapse-prevention mechanisms. It is a
diagnostic study, not a benchmark attempt. Nothing here is trying to reach state of the art.

## Non-negotiable constraints

**One training loop.** All four conditions run through `Trainer` in
`src/jepa_lens/training/trainer.py`. The only thing that varies is the
`CollapsePreventionStrategy`. Never add a condition-specific branch to the loop — if a
condition needs different behaviour, it goes behind the strategy interface. A difference in
results must be attributable to the mechanism, not to the loop. This is a correctness
requirement for the experiment, not a style preference: if the plumbing itself could produce
a difference between conditions, the comparison is unsound and any conclusion drawn from it
is worthless.

**Claim labels.** Every result-bearing statement carries one of `[established]`,
`[replication]`, `[ours]`, `[interpretation]`. In LaTeX use the macros in
`report/macros.tex`. Figures carry their label in the caption.

**Provenance in docstrings.** Any module reimplementing a published method cites the paper.
If the implementation is unvalidated, the docstring says so — see
`src/jepa_lens/training/sigreg.py` for the pattern.

**Negative results are reported.** If the replication fails, the report says the replication
failed, in the abstract and the conclusion. Do not bury it.

## Conventions

- Tests are CPU-only, use synthetic tensors, and finish in seconds. They catch wiring bugs
  before GPU time is spent; they do not validate research claims.
- `experiments/**/metrics.jsonl` is tracked in git. `*.pt` and `report/figures/*.pdf` are not.
- Effective rank means the participation-ratio variant, `(sum lambda)^2 / sum(lambda^2)`.
  Say which variant in any docstring that touches it.
- Figures are generated from run logs, never hand-placed, so they cannot drift from the runs
  behind them.
- Run `uv run pytest && uv run ruff check .` before committing.

## Open items

`docs/open-questions.md` tracks unresolved issues. The SIGReg validation against the
reference implementation is the highest-priority one and blocks calling any result a
reproduction of LeJEPA.

## Reading the diagnostics (learned the hard way)

**Effective rank is not a collapse detector on its own.** It is a ratio of eigenvalue
sums, so it is scale-invariant. When a representation collapses into numerical noise, that
noise is isotropic and effective rank *rises*. In this project's pilot it went 2.71 to
30.29 while the representation died. Always read it beside `total_variance`, which is the
quantity that actually falls.

**Probe accuracy is logged twice and both are needed.** `probe_accuracy` standardizes
features so the regularization strength means the same thing at every checkpoint;
`probe_accuracy_unscaled` does not. Standardizing rescales a collapsed encoder's numerical
residue back to unit variance, and that residue still carries input information, so the
standardized probe can read a fully collapsed encoder at well above chance. On synthetic
embeddings with cosine similarity 1.0000 it scored 1.000 against a 0.100 floor. A widening
gap between the two variants is itself a collapse signal.

**The metrics that did work** on total collapse were `mean_feature_std` and
`mean_pairwise_cosine`. Both are scale- or direction-sensitive and both registered it
within 100 steps.

**Validate the instrument before spending a budget.** Run the pilot, confirm
`none_nostopgrad` reads as collapsed on *every* axis, and only then run full length. It has
no collapse prevention at all; if it looks healthy, the apparatus is lying and no other
condition's number means anything.

**Adding a diagnostic later is expected.** Run logs are kept in git for the life of the
project, so anything consuming them must tolerate records written before a metric existed.
`_plot_metric` skips records missing the key rather than raising.

## Do not add DDP without reading this first

SIGReg and every collapse diagnostic are **batch-level statistics** -- they measure a
property of the distribution of embeddings in the batch, not a mean of per-sample terms.
Splitting a batch across ranks changes what they measure, so naive DDP would silently
alter the object under study. If DDP is ever genuinely needed, the embeddings require a
gradient-aware all-gather before the loss, and the equivalence must be verified against a
single-GPU baseline before any result is trusted.

Use `scripts/run_all_conditions.py --parallel` instead: one condition per GPU, four
independent processes, training code unchanged. Full reasoning in
`docs/development-log.md`.
