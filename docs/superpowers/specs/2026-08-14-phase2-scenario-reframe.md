# Phase 2 — Scenario-embedding reframe

Date: 2026-08-14
Status: approved (direction confirmed by owner after external advisor audit)

## The reframed central question

> **Can you trust a self-supervised scenario embedding?** Which label-free diagnostics
> reliably detect each mode of representation degeneration in negative-free masked JEPA
> training — and do they still work when the embedding is used for driving-scenario
> retrieval?

SIGReg, stop-gradient, and the projector stop being the contribution and become the
**instrument**: controlled interventions that manufacture known degeneration modes
(complete collapse, angular concentration, healthy-but-contracting) so diagnostic
reliability can be tested against ground truth.

Why this framing, given the owner's goals (interview preparation for an ADAS-validation
CIFRE, portfolio credibility, learning): scenario intelligence embeds unlabeled driving
logs and asks whether the embedding covers the ODD. That question is unanswerable with
labels at scale, so it rests entirely on label-free diagnostics — and this project has
already shown two standard ones move the wrong way under collapse.

## What the advisor audit established (accepted findings)

1. **B1**: SIGReg was applied to the probed representation with no projector; LeJEPA
   applies it to a disposable projector output (`MINIMAL.md:171-175`,
   `projector_dim=512`). Known ~20-point effect class. All "SIGReg didn't learn"
   conclusions are confounded until a projector arm exists.
2. **B2**: `none_stopgrad` missing — the 2×2 (stop-gradient × SIGReg) has a hole; the
   stop-gradient effect cannot be isolated from SIGReg.
3. **B4**: every result in the repo is n=1 seed.
4. **I5/I6**: max-over-checkpoints endpoints and post-hoc thresholds
   (`cosine > 0.5` flags the project's *best* run as collapsed).
5. **I9**: the logged "effective_rank" is the covariance participation ratio, not RankMe;
   the names must not be conflated.

## Phase A — calibration bench (CIFAR-10, extends existing harness)

### Condition matrix (7)

| condition | target | stop-grad | SIGReg | projector | degeneration it manufactures |
| --- | --- | --- | --- | --- | --- |
| `ema_stopgrad` | EMA | yes | – | – | healthy → slow angular re-concentration (observed) |
| `none_stopgrad` **new** | shared | yes | – | – | does stop-grad alone hold? (SimSiam-adjacent cell) |
| `none_nostopgrad` | shared | no | – | – | complete scale+angular collapse (verified) |
| `sigreg_stopgrad` | shared | yes | λ=0.05 | – | geometry held open, no semantic gain (observed) |
| `sigreg_nostopgrad` | shared | no | λ=0.05 | – | slow angular collapse under active regularizer |
| `proj_sigreg_stopgrad` **new** | shared | yes | λ=0.05 | 512-d MLP | does the buffer unlock semantic learning? |
| `proj_sigreg_nostopgrad` **new** | shared | no | λ=0.05 | 512-d MLP | can SIGReg-behind-a-projector replace stop-grad? |

Projector: MLP `embed_dim → 512 → 512` (matches LeJEPA launcher `projector_dim=512`,
`projector_arch="MLP"`). SIGReg sees the projector output; the probe and all diagnostics
see the encoder output; the projector is discarded at evaluation. **Deliberate deviation
from LeJEPA, documented:** our prediction loss stays in encoder latent space (LeJEPA has
no latent-prediction task; its invariance loss also acts on projections). The projector
here is purely the regularizer's buffer — which is exactly the variable under test.

### Design parameters (pre-registered)

- **Seeds: 3** (0, 1, 2), paired — same seed ⇒ identical init, masks, batch order across
  conditions (already guaranteed by the seed-threading work; verified bit-identical).
- **Budget: 4,000 steps**, checkpoint every 200. Separation was unambiguous by 2,000 in
  every prior run; the 32k run peaked at 11k and degraded after. 21 runs ≈ 6 waves.
- **Primary endpoint:** unscaled probe accuracy **at the final checkpoint** (fixed a
  priori — no max-over-checkpoints). Max is still reported, labelled selection-biased.
- **Secondary endpoints:** total variance, mean pairwise cosine, participation ratio,
  RankMe, per-component loss trajectories, standardized-vs-unscaled probe gap.
- **Noise accounting:** a difference is claimed only if it clears 2× the seed-level SD
  *and* the probe's binomial SE (≈0.007 at n=5,000 test samples — `probe_test_samples`
  raised from 2,000 for this reason).

### Falsification criteria (pre-registered)

- *Projector hypothesis*: falsified if `proj_sigreg_nostopgrad` collapses as badly as
  `sigreg_nostopgrad` across all 3 seeds (cosine within seed-SD of each other).
- *Stop-grad-alone hypothesis*: `none_stopgrad` collapsing would show stop-gradient alone
  is insufficient here (informative either way).
- *Diagnostic-reliability claims*: any diagnostic that orders two degeneration modes
  inconsistently across seeds is reported as unreliable — that is a result, not a failure.

## Phase B — driving scenarios (BDD100K)

- **Data:** BDD100K 100k still images (70k train / 10k val) + per-image attributes:
  weather (6), scene (6), timeofday (3). Owner downloads (registration required) to
  `data/bdd100k/`. Square resize to 128×128, patch 8 → 16×16 grid; masking fractions
  unchanged. Aspect distortion accepted and documented (alternative — non-square encoder
  — rejected as an architectural change with no bearing on the question).
- **Training:** identical harness, identical 7×3 matrix, same budget rules
  (pilot first, then decide full budget from the curves).
- **Endpoints:** per-attribute linear probes (standardized + unscaled, all six
  probe numbers logged per checkpoint) **plus retrieval P@10** per attribute on
  L2-normalized encoder embeddings — retrieval is the operation scenario mining actually
  performs, so it is the primary semantic endpoint in Phase B; probes are secondary.
- **The Phase B question:** do the diagnostics that predicted probe/retrieval quality on
  the calibration bench still predict it on scenario data — and do any flip?

Out of scope for Phase B (recorded as future work in the report): trajectory datasets
(highD/inD), multi-camera, video tubelets, any ADAS-system claims.

## Report restructure

- New title: *"Can you trust a self-supervised scenario embedding? Diagnosing
  representation degeneration in masked JEPA training."*
- Part 1 (stop-gradient × SIGReg) and Part 2 (probe-vs-diagnostics timing) become
  supporting sections; the diagnostic framework becomes the spine.
- The four instrument bugs stay in the report as the motivating narrative — they are the
  evidence that diagnostic failure is the norm, not the exception.
- Pre-registration subsection added to Experiments **before** Phase A runs execute.

## Implementation milestones

- **M1 (validity fixes, this commit series):** `none_stopgrad`; projector module +
  two `proj_*` conditions; `--seed` end-to-end with seed-suffixed tags; `--seeds` on the
  runner; `participation_ratio` (canonical name) + `rankme` logged alongside the legacy
  `effective_rank` key; summarizer reworked to fixed-step endpoints with per-phenomenon
  columns (no binary collapse verdicts); `aggregate_seeds.py` for mean±SD across seeds.
- **M2:** Phase A runs (21) + seed-aggregated analysis + report Phase-A results.
- **M3:** BDD100K loader (index from official JSON, synthetic-fixture tests, no download
  in CI), multi-attribute probe + retrieval evaluation, Phase B pilot, then full runs.
