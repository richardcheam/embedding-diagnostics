# Working in this repository

## What this is

A seven-condition harness that manufactures known degeneration modes in JEPA training and
tests which label-free diagnostics detect them. It is a diagnostic study, not a benchmark
attempt. Nothing here is trying to reach state of the art.

## Non-negotiable constraints

**One training loop.** All seven conditions run through `Trainer` in
`src/embedding_diagnostics/training/trainer.py`. The only thing that varies is the
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
`src/embedding_diagnostics/training/sigreg.py` for the pattern.

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

## The probe is noisier than the loss

Training is bit-deterministic, but probe accuracy varies by up to ~0.002 between identical
runs of the same commit, because non-deterministic CPU kernels perturb the embeddings'
last bits and a few test samples flip across the decision boundary. Do not treat
between-condition probe differences below ~0.005 as real. See `docs/development-log.md`.

## Phase-2 rules (post-audit)

The central question is now diagnostic reliability, with the conditions as manufactured
degeneration modes — see `docs/STATUS.md` section 0 and the Phase-2 spec. Non-negotiables
added by the audit:

- **Endpoints are pre-registered.** The primary number is the FINAL-checkpoint unscaled
  probe (or retrieval P@10 on BDD). Never select best-over-checkpoints as a result; the
  expected max of 21 noise checkpoints is already +0.005.
- **No single-seed claims.** A difference is real only if it clears 2x the across-seed SD
  **WITHDRAWN 2026-08-14 (96184e2), before any Phase-A run.** It was not a paired
  analysis and acted as an arbitrary threshold. The protocol in force is paired
  Student-t 95% intervals over the five contrasts in `embedding_diagnostics.stats.CONTRASTS`, with
  no verdict language: never label a result significant, real, learned, or collapsed.
  `aggregate_seeds.py` prints the intervals.
- **The projector is part of the design.** SIGReg results without a projector describe the
  no-projector configuration, not SIGReg. Keep `reg_embedding` distinct from
  `context_embedding` — the rename exists so this cannot be missed silently.
- **Naming discipline for rank measures.** "Participation ratio" and "RankMe" are
  different quantities and both are logged; never write "effective rank" in new prose.
