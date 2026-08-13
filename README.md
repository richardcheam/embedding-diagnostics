# jepa-lens

Does stop-gradient still matter under SIGReg once training runs long enough? And do cheap
collapse diagnostics notice before a linear probe does?

**Start here: [`docs/STATUS.md`](docs/STATUS.md)** — the plain-language story of what we set
out to test, what actually happened, and what is still open.

**Status:** harness complete, pilots and a lambda sweep done, one 32k baseline run done.
Part 1 answered (stop-gradient still matters); Part 2 answered negatively so far.

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

# All four, one after another
uv run python scripts/run_all_conditions.py --device cuda --tag main

# All four at once, one per GPU (logs land in experiments/<tag>/<condition>/train.log)
uv run python scripts/run_all_conditions.py --device cuda --tag main --parallel
```

`--parallel` gives each condition its own GPU via `CUDA_VISIBLE_DEVICES`; add
`--gpus 0,1,2,3` to choose specific ones. There is deliberately no data-parallel
sharding of a single condition — SIGReg and the collapse diagnostics are batch-level
statistics, so splitting a batch across ranks would change what they measure. See
`docs/development-log.md`.

Conditions: `ema_stopgrad`, `sigreg_stopgrad`, `sigreg_nostopgrad`, `none_nostopgrad`.

## Sweeping the SIGReg weight

```bash
uv run python scripts/sweep_sigreg_weight.py --device cuda \
  --lambdas 0.01,0.02,0.05,0.1 --total-steps 2000 --checkpoint-every 100 --tag sweep
uv run python scripts/summarize_sweep.py --tag sweep
```

Lambdas use the reference LeJEPA parametrisation (`sigreg*lambda + other*(1-lambda)`);
this project's additive `sigreg_weight` is `lambda/(1-lambda)`. Both SIGReg conditions run
at every lambda, since the question is not only whether SIGReg helps but whether
stop-gradient still matters at that setting.

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

Summarised in [`docs/STATUS.md`](docs/STATUS.md); written up in `report/`.

- **Part 1 — does stop-gradient still matter under SIGReg?** Yes. Across every SIGReg
  strength in LeJEPA's own swept range, runs without stop-gradient collapsed and runs with it
  did not.
- **Part 2 — do cheap diagnostics warn before the probe?** No, on the evidence so far. On the
  one run that both learned and degraded, the probe turned 2,500 steps *earlier* than the
  cheap diagnostic.
- **Caveat that governs both:** in this setup SIGReg never produced a useful representation
  at any strength. We pair it with masked latent prediction; LeJEPA pairs it with multi-view
  invariance. These results are about that combination, not about LeJEPA's claim.
