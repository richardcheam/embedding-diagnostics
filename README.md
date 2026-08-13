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

## GPU machines

Check the environment before training. It exits non-zero and explains itself if the
requested device cannot actually be used:

```bash
uv run python scripts/check_environment.py --device cuda
```

`train.py` runs the same check at startup, so a mismatched build costs seconds instead
of surfacing partway through a run.

Torch is pinned to PyTorch's CUDA 12.8 index on Linux (see `[tool.uv.sources]` in
`pyproject.toml`); macOS stays on ordinary PyPI wheels. The default PyPI wheel tracks
the newest CUDA major version, and CUDA is only minor-version compatible — a `cu130`
build cannot run on a 12.x driver at all, whatever the GPU. If your driver supports
CUDA 13, change `cu128` there.

Do not fix a broken environment with `uv pip install`. `uv run` re-syncs the venv to
`uv.lock` on every invocation and will undo it, potentially leaving a mix of CUDA
versions whose symptom is an `undefined symbol` ImportError from `libtorch_cuda.so`.
Change the pin in `pyproject.toml` and re-lock instead. To recover from a mixed venv:

```bash
rm -rf .venv && uv sync
```

The model is small — around 5M parameters on 32x32 inputs — so one GPU is plenty.

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
