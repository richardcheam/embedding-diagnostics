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
