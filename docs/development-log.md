# Development log

This records defects found while building the harness, and process caveats a reader
should know. It exists because several of these were caught only by review, and the
same classes of mistake are easy to reintroduce.

## Defects found and fixed during development

Every one of these was a bug in the implementation plan, not in an implementer's
transcription of it. All were caught by review or by a failing test, not by inspection.

| Area | Defect | Why it mattered |
| --- | --- | --- |
| `diagnostics/metrics.py` | `effective_rank` returned `0.0` on a fully-collapsed (all-zero) spectrum, below its own documented floor of 1 | The `none_nostopgrad` control is built to collapse hard, so the headline figure would show a discontinuous jump to zero exactly where collapse is most complete |
| `diagnostics/projection.py` | True zero-variance input was never tested; sklearn emitted a `RuntimeWarning` on it | The collapsed condition hits this path at every checkpoint of a long run |
| `data.py` | Mask truncation kept a *prefix* of patch indices rather than a random subset | Measured 88% context retention for the top-left patch vs 2.8% for the bottom-right — a fixed spatial confound in every run |
| `training/sigreg.py` | Per-slice standardization erased the per-direction variance signal that isotropy is defined by | A heavily anisotropic batch scored *better* than an isotropic one, silently inverting the objective in two of four conditions |
| `training/strategy.py` | `components["sigreg"]` reported the raw penalty while `total` used the weighted one | Logged loss components did not reconcile with the total they claimed to break down, at any weight other than 1 |
| `training/trainer.py` | Logged learning rate was computed after the step increment | Every `lr` value in every run log was one schedule step ahead of the loss beside it |
| `training/strategy.py` + `data.py` | `sigreg_loss` drew slice directions from the *global* torch RNG, and the DataLoader reseeded its shuffle and worker augmentation from that same global RNG each epoch | After epoch 1, SIGReg and non-SIGReg conditions saw different data orderings and augmentations — a plumbing-induced difference between conditions, exactly what the shared training loop exists to rule out |
| `training/sigreg.py` | The fix above gave the strategy a CPU generator, but `torch.randn` rejects a generator whose device differs from the target — so a CPU generator plus CUDA embeddings raised at the first step | Introduced by the previous row's fix and invisible to the entire CPU test suite. Both SIGReg conditions would have crashed on the first GPU run |

Two notes on the SIGReg one, which was the most serious:

- It was caught because an implementer hit a failing behavioural test and **reported the
  maths rather than tuning constants until the test went green**. Tuning would have
  produced a harness that ran, trained, and passed its tests while measuring the opposite
  of what it claimed.
- It is recorded in `open-questions.md` as raising, not lowering, the priority of
  validating against the reference implementation — a summary-level reading encoded a
  plausible-but-wrong instinct, and only a test that happened to exercise that specific
  property revealed it.

## Process caveats

- Four commits did not pass through the normal implement-then-independently-review path,
  because repeated API outages blocked subagent dispatch: `23904b7` (HTML escaping),
  `2034790` (the global-RNG fix), `dcc2ada` and `8ba67c9` (test hardening). All four were
  subsequently covered by the final whole-branch review, which found no issues in them.
- The README and agent conventions were written before the figure and HTML tooling
  existed, so for a period the README documented files that did not yet exist. Resolved
  once those tasks landed; every documented command was then verified to run.

## Still outstanding

- **SIGReg is not validated against the reference implementation** at
  https://github.com/rbalestr-lab/lejepa. See `open-questions.md`. Nothing produced with
  it should be described as reproducing LeJEPA until that check is done.
- **No experiments have been run.** Only 4-step CPU smoke tests, which exist to prove the
  pipeline executes end to end, not to say anything scientific.

## Pilot findings (2026-08, 2000 steps, first real GPU run)

The pilot was run only to check the apparatus, not to answer anything. It found a defect
in the measurement itself, which is the reason the pilot exists.

`none_nostopgrad`, the control with no collapse prevention, collapsed completely by step
100 and stayed there: mean pairwise cosine 1.000, per-feature std 0.623 -> 0.0019,
prediction loss 1e-5. Every input mapped to essentially the same vector.

Two of the four diagnostics reported the opposite:

| diagnostic | step 0 | step 2000 | verdict |
| --- | --- | --- | --- |
| mean feature std | 0.623 | 0.0019 | correctly detected collapse |
| mean pairwise cosine | 0.333 | 1.000 | correctly detected collapse |
| effective rank | 2.71 | 30.29 | **rose as the representation died** |
| probe accuracy (standardized) | 0.375 | 0.409 | **rose as the representation died** |

Both failures are the same bug in different clothing: **the metric is scale-invariant and
the collapse is a loss of scale.** Effective rank is a ratio of eigenvalue sums, so once
signal drops below the noise floor it measures the shape of isotropic float noise.
StandardScaler divides features by their std, which multiplies a collapsed encoder's
residue back up to unit variance -- and that residue is still a deterministic function of
the input, so a linear model reads it. Verified synthetically: at cosine similarity
1.0000, the standardized probe scored 1.000 and the unstandardized probe 0.592, against a
0.100 chance floor.

The standardization was added deliberately, so that probe accuracy would stay comparable
across checkpoints as embedding scale drifts. That exact reasoning is what created the
blind spot.

This also threatened the research question, not just the control: `sigreg_nostopgrad` was
visibly degrading (std 0.623 -> 0.295, cosine 0.333 -> 0.879) yet its standardized probe
accuracy read *higher* than the healthier-looking `sigreg_stopgrad`. And Part 2 asks
whether cheap diagnostics move before the probe degrades -- unanswerable if the probe does
not degrade under collapse.

### Decisions taken

1. Log `total_variance` (trace of the covariance) as an explicit scale metric.
2. Log `probe_accuracy_unscaled` alongside `probe_accuracy`, and plot them together.
3. Document in `effective_rank`'s docstring, in AGENTS.md, and on every plot that
   effective rank is a shape metric, not a magnitude metric.
4. Do not start the 32k-step runs until the pilot is repeated and the control reads as
   collapsed on every axis.

Nothing about the between-condition orderings in this pilot should be treated as a result.
They were measured with the instrument described above.

## Why this study does not use data-parallel training (DDP)

Four GPUs were available and only one was being used, so distributed data-parallel was
the obvious next step. It is the wrong tool here, for a reason specific to what this
study measures.

**SIGReg is a batch-level statistic.** `sigreg_loss` estimates an empirical characteristic
function with `.mean(dim=0)` over the batch: it asks whether *the distribution of
embeddings in this batch* looks like an isotropic Gaussian. The collapse diagnostics
(`total_variance`, `effective_rank`, `mean_pairwise_cosine`, per-feature std) are batch
statistics for the same reason.

Under standard DDP with `batch_size: 256` and four ranks, each rank sees 64 samples,
computes SIGReg on its own shard, and DDP averages the resulting gradients. That is not
the same quantity as SIGReg computed on all 256:

- the empirical characteristic function is estimated from a quarter as many samples, so
  each rank's estimate is substantially noisier;
- averaging four gradients of four noisy estimates is not the gradient of one better
  estimate, because the loss is not linear in the samples.

The regularizer would therefore behave differently under DDP than under single-GPU
training -- and the regularizer is the object under study. Any comparison between a
DDP run and a single-GPU run would confound "does stop-gradient matter" with "how many
samples did the isotropy test see".

Making DDP correct here is possible but is real work: the embeddings would need a
gradient-aware all-gather before `sigreg_loss` (a plain `all_gather` detaches, so
gradients would not flow back to the other ranks' encoders), and the same treatment for
the diagnostics. That equivalence would then have to be verified against a single-GPU
baseline before trusting any result from it.

**It would also probably be slower.** The encoder is ~5M parameters on 32x32 inputs. At
that size the per-step NCCL communication plausibly costs more than the compute it saves.

**What is used instead:** one condition per GPU. There are exactly four conditions and
four cards, the four runs are completely independent, and `CUDA_VISIBLE_DEVICES` gives
each training process a single visible device so the training code is unchanged --
including the RNG determinism work, which DDP would have complicated further. See
`scripts/run_all_conditions.py --parallel`.

The general lesson: **data parallelism is only transparent when the loss decomposes over
samples.** Ordinary supervised losses are a mean over per-sample terms, so sharding is
exact. Any loss that measures a property *of the batch distribution* -- isotropy
regularizers, contrastive losses with in-batch negatives, batch-norm-dependent objectives
-- changes meaning when the batch is split, and needs explicit cross-rank gathering to
stay equivalent.

## The probe has a run-to-run noise floor (found 2026-08-14)

While checking whether an `ema_stopgrad` run predated the SIGReg fix, the same commit was
run twice with the same seed on the same machine:

| step | loss (run 1 vs run 2) | probe_std | probe_unscaled |
| --- | --- | --- | --- |
| 3 | 1.0093109608 vs 1.0093109608 | 0.3730 vs 0.3725 | 0.3725 vs 0.3705 |
| 6 | 0.9995970726 vs 0.9995970726 | 0.3740 vs 0.3735 | 0.3710 vs 0.3730 |

**Training is bit-deterministic** -- losses match to ten decimals. **The probe is not**,
varying by up to 0.002 between identical runs.

The probe itself is deterministic given identical features (verified: five repeats on a
fixed matrix return the identical accuracy). So the embeddings must differ in their last
bits, from non-deterministic multithreaded CPU kernels in torch. Scalar reductions like the
loss average those differences away; the probe does not, because a handful of test samples
sit near a decision boundary and flip.

**Consequence for interpreting results:** the probe has a noise floor of roughly +/-0.002
from non-determinism alone, before any seed-to-seed variance. Between-condition differences
smaller than about 0.005 should not be treated as real. For reference, the pilot-2 gap
between the two SIGReg conditions was 0.008 -- only about four times this floor.

This is an n=2 observation on CPU and needs proper quantification (repeat counts, and
whether GPU kernels are better or worse). It is not currently controlled for anywhere.

## Part 2's hypothesis is not supported on the 32k baseline (2026-08-14)

The first full-length run, `ema_stopgrad` at 32{,}000 steps, is the first condition that
both learns and then degrades, so it is the first data on which Part 2 can be asked at all.
(The SIGReg conditions never learned, so their flat probes had nothing to lag behind.)

Unscaled probe accuracy climbs 0.369 -> 0.603, peaking at **step 11,000**, then declines to
0.582. Mean pairwise cosine falls to a minimum of 0.294 at **step 13,500**, then rises to
0.530 -- the healthy baseline partially re-collapses in its second half.

**The probe turns 2,500 steps BEFORE the cheap diagnostic.** That is the opposite of the
hypothesis, which was that cheap diagnostics would give earlier warning.

What the diagnostic does give is a much *louder* signal: cosine moves 0.236 from its
trough, against a probe decline of 0.021 (four times the 0.005 noise floor). So the cheap
metric is a clearer indicator of how bad things have become, but a later one.

Caveats, and they are substantial: one condition, one seed, 500-step checkpoint resolution,
so 2,500 steps is five checkpoints. The probe decline is real but small. This is suggestive,
not settled, and multi-seed runs are what would settle it.

### Two detector defects found along the way

**`first_departure_step` assumes a flat baseline.** It takes the first quarter of the series
as the reference window. On a 32k run that window is the rapid-learning phase, where the
unscaled probe sweeps 0.369 to 0.582, giving a baseline std of 0.065 and a tolerance band of
0.396 to 0.654. The final value 0.582 sits inside the band, so the function reports no
departure at all despite a clear peak-and-decline. It is fine for series that start flat; it
is useless for series that learn first.

**A "decline from running best" detector needs a fixed good direction, and these metrics do
not have one.** Total variance falls monotonically from 87.2 to 12.6 across the entire run,
*including the phase where the probe is climbing*. Falling variance there is healthy
concentration, not degradation. A decline detector duly reported a "turn" at step 1,500,
which means nothing. Whether falling variance is good or bad depends on what the probe is
doing at the same time -- it is a joint signal, not a univariate one.

`monotone_fraction` now screens such series out (total variance 0.95, mean feature std 0.94,
both excluded), and `turning_point` is used for the timing comparison instead, since it only
asks where a series reversed and assumes nothing about direction.

This is the third instrument defect found by running the experiment rather than by
inspecting the code. The pattern is consistent: each assumption looked reasonable when
written and failed on contact with the shape real data actually has.

## External audit and the Phase-2 reframe (2026-08-14)

A senior-researcher audit (run by a second agent, read-only) confirmed all ten of its
checked observations against the JSONL logs and returned REFRAME_BEFORE_MORE_EXPERIMENTS,
with two blocking design findings the harness itself could never have surfaced:

1. **The projector confound.** LeJEPA applies SIGReg to a disposable projector output and
   probes the encoder beneath (`MINIMAL.md`: `emb, proj = net(vs); sigreg(proj);
   probe(emb.detach())`, launcher `projector_dim=512`). We applied SIGReg directly to the
   probed representation. Literature puts the projector at ~20 accuracy points in
   comparable settings and reports that removing it damages VICReg outright — so "SIGReg
   never learned anything here" was a statement about a known-bad configuration.
2. **The missing cell.** Without `none_stopgrad`, the stop-gradient effect could not be
   estimated independently of SIGReg, and there was no stop-grad-alone baseline in the
   SimSiam lineage.

Plus: every run was n=1 seed; sweep endpoints took max-over-checkpoints (expected max of
21 noise checkpoints is +0.005, so the sweep's one "learned" verdict at +0.008 was not
distinguishable from selection); the binary collapse thresholds misclassified the
project's best run on BOTH halves (32k EMA: cosine 0.53 and 6.9x variance shrinkage while
gaining +0.21 probe); and the logged "effective_rank" was the participation ratio, not
RankMe.

Decisions taken, with the owner's goals (ADAS-validation CIFRE preparation, portfolio
credibility) steering: reframe around diagnostic reliability with the conditions as
manufactured degeneration modes; add the projector arm and `none_stopgrad`; 3 paired
seeds; fixed-step endpoints and a pre-registered claim rule; per-phenomenon reporting
instead of binary verdicts; Phase B on BDD100K with scenario-attribute retrieval as the
primary semantic endpoint. Full pre-registration in the Phase-2 spec and the report.

The recurring lesson extends by one: the previous four bugs were caught by running the
experiment; these two could only be caught by comparing the design against the reference
setup and against what the question claims to isolate. Instrument validation and design
validation are different activities, and passing one says nothing about the other.

## The controlled stress test, and what it says about everything above (2026-08-14)

The advisor objected to the phrase "the conditions manufacture known degeneration modes":
a training intervention sets up an optimisation, it does not dictate which degeneracy
emerges, and the modes here were read off the runs afterwards. Correct. So the
degradations are now applied directly to embedding matrices, where the mechanics are exact
by construction — scale contraction, SVD rank truncation, mean injection, isotropic noise,
and interpolation toward the global mean.

The resulting invariance map (synthetic, 8 clusters, severity 0 -> 0.99):

| transformation | total var | cosine | PR | RankMe | retrieval |
| --- | --- | --- | --- | --- | --- |
| scale contraction | 1106 -> 0.11 | 0.086 -> 0.086 | 5.97 -> 5.97 | 15.9 -> 15.9 | **1.000 -> 1.000** |
| mean injection | 1106 -> 1106 | 0.086 -> **1.000** | 5.97 -> 5.97 | 15.9 -> 1.23 | **1.000 -> 1.000** |
| mean interpolation | 1106 -> 0.11 | 0.086 -> **0.999** | 5.97 -> 5.97 | 15.9 -> 1.77 | **1.000 -> 1.000** |
| rank truncation | 1106 -> 254 | 0.086 -> 0.057 | 5.97 -> 1.00 | 15.9 -> 1.00 | 1.000 -> 0.244 |
| isotropic noise | 1106 -> 1.2e7 | 0.086 -> 0.000 | 5.97 -> 30.0 | 15.9 -> 31.7 | 1.000 -> 0.125 |

**No geometric diagnostic tracks semantic content.** The three transformations producing
the classic signatures of collapse — variance down four orders, cosine at 1.000 — leave
retrieval perfectly intact. The two that actually destroy retrieval move variance and rank
in the *opposite* direction.

Two consequences that revise earlier entries in this log:

**A mean pairwise cosine of 1.000 does not imply collapse.** Under mean injection, cosine
saturates while total variance, per-feature std, participation ratio, probe and retrieval
are all exactly unchanged. Earlier entries treated cosine near 1 as collapse evidence; it
is evidence of angular concentration, which a constant shift produces without any loss.

**Scale collapse is not information loss.** Contracting embeddings by 1e4 leaves cosine,
both rank measures, the probe and retrieval bit-identical. In `none_nostopgrad` the
variance collapse and the probe collapse co-occurred, and this log has been reading the
first as evidence for the second. It never was. They are separate events, and only the
unscaled probe ever measured the second one directly.

This is the first result in the project obtained with genuine ground truth rather than by
inference from a training run — and it cost no GPU time. It also sharpens what Phase A can
claim: the training conditions produce degeneration whose *mode* must be identified from
the diagnostic vector, not assumed from the intervention's name.
