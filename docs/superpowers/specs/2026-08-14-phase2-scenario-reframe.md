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

- **Seeds: 5** (0-4) for claim-grade Phase A, paired — same seed ⇒ identical init, masks,
  batch order AND the same evaluation images (`eval_split_seed` is fixed independently).
  **Three seeds may be reported only as an exploratory pilot**, never as claim-grade.
- **Budget: 4,000 steps**, checkpoint every 200. Separation was unambiguous by 2,000 in
  every prior run; the 32k run peaked at 11k and degraded after. 21 runs ≈ 6 waves.
- **Primary endpoint:** unscaled probe accuracy **at the final checkpoint** (fixed a
  priori — no max-over-checkpoints). Max is still reported, labelled selection-biased.
- **Secondary endpoints:** total variance, mean pairwise cosine, participation ratio,
  RankMe, per-component loss trajectories, standardized-vs-unscaled probe gap.
- **Analysis:** paired Student-t. For each pre-declared contrast, per-seed differences
  d_i, mean, sample SD (ddof=1), SE = s_d/√n, and a two-sided 95% interval
  d̄ ± t(0.975, n−1)·SE. Five contrasts, listed in `jepa_lens.stats.CONTRASTS`; EMA is a
  reference baseline only, never a contrast arm, since it differs in two factors at once.
  **No verdicts** — no result is labelled real, significant, learned or collapsed. The
  earlier "2× seed SD and binomial SE" rule is withdrawn: it was not a paired analysis and
  functioned as an arbitrary decision threshold.
- **What the interval covers:** training-seed variability conditional on the fixed
  evaluation split. Not split choice, dataset, or architecture.
- **Power:** with n=5, t(0.975,4)=2.776, so the half-width is ≈1.24 seed SDs; effects below
  roughly 1.3 seed SDs are unresolvable. A wide interval is not a null result.
- **Multiplicity:** five contrasts × several endpoints, reported without correction, as
  estimates rather than tests.

### Falsification criteria (pre-registered)

- *Projector hypothesis*: falsified if `proj_sigreg_nostopgrad` collapses as badly as
  `sigreg_nostopgrad` across all 3 seeds (cosine within seed-SD of each other).
- *Stop-grad-alone hypothesis*: `none_stopgrad` collapsing would show stop-gradient alone
  is insufficient here (informative either way).
- *Diagnostic-reliability claims*: any diagnostic that orders two degeneration modes
  inconsistently across seeds is reported as unreliable — that is a result, not a failure.

## Phase B — driving scenarios (BDD100K)

- **Data:** BDD100K 100k still images + per-image attributes: weather (6), scene (6),
  timeofday (3). **Confirmed on the GPU box 2026-08-14:** 70,000 train / 10,000 val, in
  the per-image-JSON layout (`<root>/<split>/*.jpg` with sibling `.json`); 61,591 train and
  8,801 val carry all three attributes. Loader auto-detects this and the official layout;
  `data.root` may sit outside the repo.

  **Class skew is severe and governs the endpoints.** Val: weather is 61% `clear` with
  `foggy`=13; scene is 61% `city street` with `gas stations`=7 and `tunnel`=27. Therefore
  majority-class accuracy is 0.53-0.61 and retrieval chance (class self-match probability)
  is 0.41-0.46 — not the naive 1/K of 0.17-0.33. Every logged number carries its floor
  (`probe_majority`, `retrieval_chance`) and its headroom; balanced accuracy (macro recall
  over classes with >=10 test samples) is the primary reading, and classes too rare to
  score are counted in `dropped_classes` rather than averaged away. The rare classes are
  the safety-relevant ones, so their unmeasurability at this sample size is reported as a
  finding, not hidden. Square resize to 128×128, patch 8 → 16×16 grid; masking fractions
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
