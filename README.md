# embedding-diagnostics

Can you trust a self-supervised scenario embedding? This project manufactures known modes
of representation degeneration in masked JEPA training — via stop-gradient, SIGReg, and a
disposable projector as controlled interventions — and tests which label-free diagnostics
actually detect each mode, first on a CIFAR-10 calibration bench and then on BDD100K
driving scenarios, where the semantic endpoint is scenario-attribute retrieval.

**[Project page](https://richardcheam.github.io/embedding-diagnostics/)** — a visual case
study of the experiment and probe correction, rebuilt from tracked Phase-B records and
published by GitHub Actions. Includes per-seed figures and downloadable paired intervals.

**Metric definitions** — formula, worked example and blind spot for every measure used
here (RankMe, participation ratio, P@10, balanced accuracy, chance floors, paired
intervals): [`docs/metrics.md`](docs/metrics.md).

**New to this? Start with [`docs/primer.md`](docs/primer.md)** — builds the whole chain from
scratch (what an embedding is, why models collapse, how representations are evaluated) and
ends on the finding. Read it before the report.

**Already have the background: [`docs/STATUS.md`](docs/STATUS.md)** — the plain-language
story of what we set out to test, what actually happened, and what is still open.

**Status:** **Phases A and B complete** — 7 conditions x 5 paired seeds each, on CIFAR-10
and on BDD100K driving scenarios, endpoints and claim rule fixed in advance. Re-measuring
Phase B from its saved encoders refuted the original headline claim; Phase A's probe
numbers are still uncorrected, because that campaign predates encoder saving.

## Honesty note

Most of this project is assembly of existing work. The SIGReg loss **has now been validated**
against the reference implementation and matches it bit-for-bit — validation that found a
real bug in our version (see `docs/open-questions.md`). But the *system* around it is not
LeJEPA: we pair SIGReg with masked latent prediction rather than multi-view invariance, and
our projector is a project-specific MLP, not a reproduction of theirs. No result here is
evidence about LeJEPA's claim. Full accounting in the report's contributions section.

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

Conditions (7): `ema_stopgrad`, `none_stopgrad`, `sigreg_stopgrad`, `sigreg_nostopgrad`,
`proj_sigreg_stopgrad`, `proj_sigreg_nostopgrad`, `none_nostopgrad`. The `proj_*` pair
places a 512-d MLP projector between the encoder and SIGReg, as LeJEPA does; the probe
always reads the encoder.

## Phase A (calibration bench, pre-registered)

```bash
# 7 conditions x 5 paired seeds = 35 runs
uv run python scripts/run_all_conditions.py --device cuda --parallel \
  --total-steps 4000 --checkpoint-every 200 --tag phaseA --seeds 0,1,2,3,4 \
  --jobs-per-gpu 3

# Across-seed means at the fixed final-step endpoint
uv run python scripts/aggregate_seeds.py --tag phaseA
```

## Phase B (BDD100K driving scenarios)

The data need not live inside the repository — `data.root` is just a path, and `data/` is
already gitignored if you do put it there. Two layouts are auto-detected:

    official:   <root>/images/100k/<split>/*.jpg  +  <root>/labels/bdd100k_labels_images_<split>.json
    per_image:  <root>/<split>/*.jpg              +  a sibling <name>.json per image

Verify before spending GPU time — this exits non-zero on a bad tree:

```bash
uv run python scripts/inspect_dataset.py --root ../100k
```

Then pass the path with `--data-root`. Dataset location is a property of the machine, not
of the experiment, so it stays out of the committed config:

```bash
# Pilot first — full budget is decided from the pilot curves, never assumed
uv run python scripts/run_all_conditions.py --device cuda --parallel \
  --base-config bdd.yaml --data-root ../100k \
  --total-steps 4000 --checkpoint-every 200 \
  --tag bddpilot --seeds 0,1,2
uv run python scripts/aggregate_seeds.py --tag bddpilot

# Then the full matrix, once the pilot curves justify the budget
uv run python scripts/run_all_conditions.py --device cuda --parallel \
  --base-config bdd.yaml --data-root ../100k \
  --total-steps 4000 --checkpoint-every 200 \
  --tag phaseB --seeds 0,1,2,3,4 --jobs-per-gpu 2
```

**Use `--jobs-per-gpu 2` on BDD, not the 3 that works on CIFAR-10.** Measured from the first
pilot: a CIFAR condition uses about 1 GB, a BDD condition uses **3.6–7.9 GB** — the 128x128
inputs and batch 256 dominate. Five jobs on one 24 GB card exhausted it and killed three
runs with `torch.OutOfMemoryError`. Decode is also a bottleneck (1280x720 JPEGs), so raising
the count buys less than it does on CIFAR anyway.

If a crash takes out part of a matrix, recover with `--resume` instead of rerunning
everything:

```bash
uv run python scripts/run_all_conditions.py --device cuda --parallel \
  --base-config bdd.yaml --data-root ../100k \
  --total-steps 4000 --checkpoint-every 200 \
  --tag bddpilot --seeds 0,1,2 --jobs-per-gpu 2 --resume
```

`--resume` keeps every run whose log reaches `--total-steps` and reruns only the rest,
discarding their partial logs. It needs `--total-steps` explicitly, since that is how it
decides what "finished" means.

### Restricting to specific GPUs

`--gpus` names the devices to use; each job is pinned with `CUDA_VISIBLE_DEVICES`, so the
child sees its card as device 0 and `--device cuda` means that one.

```bash
# Everything on GPU 2, two jobs at a time, the rest queued behind them
uv run python scripts/run_all_conditions.py --device cuda --parallel --gpus 2 \
  --jobs-per-gpu 2 --base-config bdd.yaml --data-root ../100k \
  --total-steps 4000 --checkpoint-every 200 --tag bddpilot --seeds 0,1,2 --resume

# One job at a time on GPU 2, leaving the card as free as possible
uv run python scripts/run_all_conditions.py --device cuda --gpus 2 ...
```

`--gpus` works with or without `--parallel`. Without it, jobs run one at a time on the named
card; naming more than one device without `--parallel` is refused rather than silently using
the first.

Each BDD run logs per-attribute probes and retrieval (`probe_accuracy_weather`,
`retrieval_p10_scene`, ...) plus across-attribute means under the canonical keys.

`--jobs-per-gpu` packs several runs onto each card: a CIFAR condition uses about 1 GB of a
24 GB device, so one-job-per-GPU leaves it nearly idle. On 4 GPUs a 35-job matrix takes 9
waves at 1/GPU and 3 waves at 3/GPU. **The CPU saturates before the GPU does** — every job
runs dataloader workers *and* fits two sklearn linear probes at each checkpoint — so raise
it until steps/sec stops improving, then stop. Check `nproc` before going high, and lower
`num_workers` in the config if the workers start starving each other.

## Phase C (external pretrained reference)

C0 adds local BDD Lance image-byte extraction through pinned EmbeddingGemma 2,
an FP32 resumable canonical cache, and the existing geometry diagnostics.
EmbeddingGemma is an external reference encoder; the seven training conditions
are unchanged. C1-pilot is complete and exploratory; C1-main is complete, and
C2 compares learned MRL with train-fitted PCA using the same cached vectors.
C3/C4 remain deferred. See the [C2 tables](experiments/phaseC_c2/results.md),
[qualified interpretation](experiments/phaseC_c2/interpretation.md), and
[`docs/phase-c.md`](docs/phase-c.md) for the offline command, provenance, and
integration record, and the [pilot report](experiments/phaseC_c1_pilot/README.md).
Phase-C assessments are restricted to the predeclared hypotheses, frozen
representation sample, and documented instrument limitations.

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
uv run python scripts/make_figures.py --tag phaseA_s0   # -> report/figures/*.pdf
make -C report                                          # -> report/main.pdf

# Current public case study: standard-library build, corrected Phase-B endpoints
python3 viz/build_site.py                   # -> viz/dist/index.html + data.json
python3 -m http.server 8080 --directory viz/dist
```

The current project page is deployed by `.github/workflows/pages.yml` on changes to its
source or Phase-B records. It presents the corrected measurement audit, individual seed
scores, and paired Student-t intervals without verdicts. See [`viz/README.md`](viz/README.md)
for hosting and provenance details.

**Historical demo:** `viz/build_report.py` and the following description predate the probe
correction. That demo retains withdrawn claims and is not published by the Pages workflow.
Use `build_site.py` for the current public case study.

The historical demo is built around the finding rather than being a generic dashboard. It opens with
the two evaluations almost everyone runs — standardized linear probe and cosine retrieval —
asks which of the seven encoders is dead, and only then reveals total variance and the
unscaled probe. That ordering is deliberate: those first two charts rate the collapsed
control *above* most of the conditions that are training normally, so a plain dashboard
built from them would look broken rather than making a point.

`--tag` supplies the curves (one seed) and `--scorecard-tag` the scorecard (across seeds).
They are separate arguments on purpose — deriving the scorecard from a single seed would
put a number on the page whose stated basis is wrong.

The scorecard verdicts each diagnostic on whether it ranks the collapsed control below
`none_stopgrad`, the weakest condition that still genuinely trains, with the margin
required to clear twice the across-seed SD. **Both the comparator and that threshold were
chosen after seeing results — this scorecard is exploratory, not pre-registered**, and the
threshold is a withdrawn heuristic retained here only for the panel's descriptive purpose.
On Phase A it gives 1 correct (unscaled probe), 4 unresolved, and 2 inverted (standardized
probe, participation ratio) — but the unscaled-probe row is itself under recomputation, see
below.

## One environment for every machine

`uv sync` installs the same CUDA 12.8 torch build everywhere, and that is intentional — do
not repin per machine. CUDA drivers are **backward** compatible, so cu128 runs on a 12.8
driver and on a 13.0 driver alike, while a cu130 build cannot run on a 12.8 driver at all.
Pinning cu130 for a newer box would break the older one. cu128 also carries sm_75 through
sm_120 kernels and publishes both `x86_64` and `aarch64` wheels, so a Turing workstation and
a Grace-Hopper (ARM) node install from the same lockfile. Comparable results across machines
need the same build on all of them.

Confirm on any new box before spending GPU time — it names the exact problem if there is one:

```bash
uv run python scripts/check_environment.py --device cuda
```

### Shared memory, if you are in a container

DataLoader workers pass batches through `/dev/shm`. Containers default to 64 MB, and one
BDD batch (256 x 3 x 128 x 128 float32) is **50 MB** — so the default cannot hold even one,
and workers die with a bare `unable to allocate shared memory(shm)` that names neither the
setting nor the batch size. CIFAR at 32px is 16x smaller per batch, which is why this only
appears on BDD.

The run now detects this before training and falls back to torch's `file_system` sharing
strategy so it proceeds regardless. That is a workaround, not a fix — it is slower. Prefer,
in order:

```bash
docker run --shm-size=8g ...        # or --ipc=host; best, needs container restart
--num-workers N                     # fits within the shm you have
--num-workers 0                     # last resort: no prefetching, slow
```

On a large-memory node, raise `--jobs-per-gpu` well above the 2 that suits a 24 GB card: at
3.6–7.9 GB per BDD job a 144 GB device fits well over a dozen. The CPU saturates before the
GPU does — each job runs dataloader workers *and* fits two sklearn probes per checkpoint —
so raise it until steps/sec stops improving, then stop.

## Workflow across machines

Development happens without a GPU; training happens elsewhere. Run logs are small and
tracked in git, so results move back by `git pull` — no separate sync tooling. Checkpoints
(`*.pt`) are gitignored and are not needed to rebuild any output.

## Results

Summarised in [`docs/STATUS.md`](docs/STATUS.md); written up in `report/`.

**Supporting evidence, from controlled degradations (no training required):** applying
*known* transformations to embedding matrices maps out which diagnostic is invariant to
which change. Contracting 10,000x, or driving mean pairwise cosine to 1.000, leaves
retrieval P@10 at exactly 1.000 — but note these transformations are **invertible by
construction**, so they are designed to preserve information while changing geometry. They
establish each metric's *invariances*; they do not by themselves show that geometry can
never track semantics. The training results above are what show the dissociation arising on
its own.

**Phase B (BDD100K driving scenarios, 7 conditions × 5 paired seeds).** The control has no
collapse prevention. Every geometric measure says it is destroyed:

| geometric measure | healthy | **contracted control** | weakest run that trains |
| --- | --- | --- | --- |
| total variance | 107.71 | **0.0001** | 39.61 |
| mean pairwise cosine | 0.256 | **1.0000** | 0.676 |
| RankMe | 78.36 | **1.06** | 7.45 |

Measured with a working probe, it is nearly as decodable as a condition that genuinely
trained:

| attribute | healthy | **contracted** | weakest real | majority floor |
| --- | --- | --- | --- | --- |
| weather | 0.7271 | **0.6680** | 0.6778 | 0.6041 |
| scene | 0.6560 | **0.6381** | 0.6333 | 0.6034 |
| timeofday | 0.9271 | **0.9137** | 0.9167 | 0.4833 |

Guessing scores 0.483 on timeofday. A healthy encoder scores 0.927. An encoder with a
millionth of normal variance, every output pointing the same direction, scores **0.914**.

> **Geometric degeneration does not imply loss of decodable information.**

**Second finding: the two semantic evaluations disagree.** Linear decodability survives;
neighbourhood structure does not.

| | healthy | contracted | weakest real | chance |
| --- | --- | --- | --- | --- |
| linear probe (timeofday) | 0.9271 | 0.9137 | 0.9167 | 0.483 |
| **retrieval P@10** | 0.6750 | **0.5664** | 0.6329 | 0.429 |

The probe says fine; retrieval says clearly worse, by more than the seed noise. A linear
classifier only needs *some* separating direction; retrieval needs the *ordering of
neighbours*, and squashing the cloud scrambles that. **For scenario mining this is the half
that matters** — the application is retrieval, and it is the endpoint that degrades.

**Third finding: two "effective rank" measures invert.** The participation ratio reads
**25.45** on the contracted encoder against **10.99** on the healthy one — paired 95%
interval [−32.70, −15.36], in the wrong direction. It subtracts the mean before measuring,
which deletes exactly the failure; RankMe does not, and gets it right.

### What we withdrew

An earlier version of this README claimed the standard evaluation was blind to collapse
while the unstandardized probe detected it. **That was wrong.** The unstandardized probe's
optimiser stops when the objective's slope goes shallow, the slope scales with feature
magnitude, and at ~0.001 it stopped before fitting — predicting the majority class and
returning chance, with no warning. Re-measuring Phase B from its saved encoders moved the
contracted encoder's timeofday score from 0.4832 to 0.9137 while every other condition moved
by ≤0.01. There was no information loss for the standard protocol to be blind to.

⚠️ **Phase A (CIFAR-10) probe numbers are not corrected** — that campaign saved no encoders
and needs a re-run. Do not quote them. Its geometric numbers are unaffected.

Full account: [`docs/STATUS.md`](docs/STATUS.md); provenance and commit references in
`report/sections/provenance.tex`.

**The prescription.** No single label-free diagnostic covers every collapse mode — that is
an exhaustive search over subsets, not a preference. Two do: **total variance + RankMe**.
The argument is that the *scale-invariant* metrics (cosine, RankMe, participation ratio)
cannot see pure scale contraction, while the *centered* metrics (total variance,
participation ratio) cannot see the cloud shifting off the origin, so a sufficient panel
needs one from each family. Exactly four pairs qualify and **the participation ratio is in
none of them**. Reproduce with:

```bash
uv run python scripts/validate_panel.py
```

Two things there were surprises. Drift-based alarms — "flag when a metric moves far from
its init value" — give 5 false alarms out of 6 on CIFAR-10 and 4 out of 6 on BDD100K,
including on the only condition that actually learns; healthy SSL training legitimately
reshapes geometry. Absolute limits near each metric's degenerate floor instead give **zero
misclassifications across all 14 condition-dataset pairs**, with margins of 9x (RankMe) to
41,716x (total variance).

A tripped check means scale-invariant metrics are untrustworthy on that embedding — not
that it carries no information. Three of five controlled degradations leave retrieval P@10
at exactly 1.000 while wrecking the geometry.

Two further results:

- **Geometry and semantics fail to resolve in opposite places.** Across the SIGReg
  contrasts every geometric comparison clears the claim rule and is enormous (169 units of
  variance, 0.84 of cosine) while the probe moves at most 0.035. In the one contrast whose
  semantic difference is unambiguous (+0.210 probe accuracy), *neither* geometric endpoint
  clears the rule. Absolute detection of total collapse still works; what fails is using
  these diagnostics to rank or compare configurations.
- **RankMe and the participation ratio are not interchangeable**, despite both being called
  effective-rank measures. Each is blind to the collapse mode the other detects — PR uses
  the centered covariance spectrum and cannot see collapse to a non-zero constant; RankMe
  uses raw singular values and can, but reads high when a shrinking residual stays isotropic.

Only `ema_stopgrad` beat its own random initialisation (+0.161); every SIGReg arm finished
below it. That is a statement about SIGReg paired with masked latent prediction, **not**
about LeJEPA, which pairs it with multi-view invariance and no predictor.
