# jepa-lens

Does stop-gradient still matter under SIGReg once training runs long enough? And do cheap
collapse diagnostics notice before a linear probe does?

**Status:** experiment harness complete, runs not yet performed.

## Honesty note

Most of this project is assembly of existing work. The scale-dependence result under test is
a **replication attempt** of a finding encountered secondhand, not an established fact. The
SIGReg implementation was written from a summary-level reading of
[LeJEPA](https://arxiv.org/abs/2511.08544) and is **not yet validated** against the reference
implementation — see `docs/open-questions.md`. Full accounting in the report's contributions
section.

## Setup

```bash
uv sync
uv run pytest          # CPU-only, seconds
uv run ruff check .
```

## Running

```bash
# One condition
uv run python scripts/train.py --condition sigreg_nostopgrad --device cuda --tag main

# All four
uv run python scripts/run_all_conditions.py --device cuda --tag main
```

Conditions: `ema_stopgrad`, `sigreg_stopgrad`, `sigreg_nostopgrad`, `none_nostopgrad`.

## Building outputs

```bash
uv run python scripts/make_figures.py --tag main   # -> report/figures/*.pdf
make -C report                                     # -> report/main.pdf
uv run python viz/build_report.py --tag main       # -> viz/dist/report.html
```

## Workflow across machines

Development happens without a GPU; training happens elsewhere. Run logs are small and
tracked in git, so results move back by `git pull` — no separate sync tooling. Checkpoints
(`*.pt`) are gitignored and are not needed to rebuild any output.

## Results

Not yet run.
