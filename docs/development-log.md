# Development log

## Phase-C C3: bounded numerical instrument study (2026-10-08)

[ours] Final protocol and tested implementation `443459f` preceded freeze
`50f6659` and all eight new endpoint records. Native 768d is primary; only
the existing training-fitted mean-injection .99 reference and FP16-gallery
condition provide a supporting stress. Canonical vectors, sample, dependencies
and 40 accepted historical files remain unchanged. No encoder/image/ANN/C4 work.

Read-only review caught a backend witness running too late and outside the
execution thread limit, missing per-row clipping context, and unchecked report
provenance/completion. Fixed before freezing, with synthetic regression tests.
Actual normalization/scoring dtypes, backend identity and a matrix cancellation
witness are checked before endpoints. Ties have an explicit ID ordering; bins
keep equal reference margins together. Attribute turnover uses entering and
departing label counts, not arbitrary neighbour pairs.

[ours] Initial invocation 6.226 seconds, peak RSS 744.38 MiB; native FP16 changes
five neighbour sets, INT8 60/75, FP32 arithmetic zero. Fixed stressed FP16 changes
195 sets. Attribute P@10 responds much less. No sufficient-check exceptions;
the prospective bound covers 592/412 native FP16 queries and no INT8/stress
queries. Completed resume performs zero endpoint calls and preserves record
bytes. 441 tests pass, one optional accelerator test skips; Ruff/lock pass.

[interpretation] These are conditional numerical responses on a previously
examined sample. Loose bounds, clipping, coarse labels and unknown sequence
dependence limit transfer claims. See `experiments/phaseC_c3/interpretation.md`;
independent replication or a multimodal extension requires a separate decision.

[ours] Final record review additionally identified 17 inherited `synthetic_val_`
validation IDs with unexplained origins. The prefix is not proof of fabrication;
no rows or accepted/frozen endpoints were changed. Source qualification is
explicit in the C3 record and open questions rather than hidden by filtering.

## Phase-C C2: cache-only matched-dimensional comparison (2026-10-08)

[measurement] Accepted C1 source `0c19a85` was audited before C2 freeze. Its
raw training SVD defines the shared train/validation projector; severe cached
validation output and geometry match exactly. No H4 replacement was needed.
C1 results and pilot records remain unchanged.

C2 implementation `5e3f3a5`, freeze `6206cb0`: exact native 768 copy; official
learned MRL 512/256/128 with mandatory L2 normalization; centered training-only
full PCA, nested shared basis and L2 normalization at matching dimensions.
Canonical cache/matrix/sample, source files and runtime versions are bound.
No dependency changes, encoder inference, BDD image decoding or new sampling.

Review caught missing transitive cache-validator source coverage, absent
balanced floors, and an overwrite-before-resume-check hazard for PCA parameters.
These were fixed before freezing endpoints. All project Python sources are
hashed; parameters are validated/reused or atomically created; mismatches fail.
The existing RankMe epsilon can put a capacity fraction just above one, so the
new fractions preserve that definition and do not silently clamp it.

[measurement] Seven endpoint records completed in 336.93 seconds with guards
forbidding network connections, model construction and image decoding. Native
probe scores exactly reproduce C1. Completed resume made zero endpoint calls
and preserved endpoint/effect/parameter hashes. All 21 final standardized fits
converged, 18 at the lower C boundary; 19 inner candidates did not converge.
All warnings remain tracked. See `experiments/phaseC_c2/results.md` and the
separate qualified hypothesis assessment; no C3/C4 work began.


## Phase-C C1-pilot (2026-10-07)

Implemented metadata-only census and immutable uniform fully-labelled sampling,
then held-out semantic endpoints and train-fitted versions of the existing five
feature interventions. The sole C0 path adjustment parameterizes its sampling
description: otherwise it would falsely record C0 lexicographic sampling for
C1's random subset. Adapter, cache, Phase-B loader and shared diagnostics remain
unchanged. No C1 dependency changes or training conditions were added.

[measurement] The fixed 384/128 sample produced finite unit-normalized 768d
vectors in 5706.03 s, CPU FP32 / batch 1 / four threads, 3.59 GiB peak RSS.
Thirty stress records and their optimization diagnostics are retained in
`experiments/phaseC_c1_pilot/`. The pinned mean-pooling/Normalize module sequence
was checked against local configuration without SentenceTransformers.
Completed-cache and analysis resumes used no-inference/no-endpoint sentinels.
392 tests passed, one optional accelerator test skipped; Ruff and lock checks
passed. Read-only review found no critical or important defect; its suggestions
added held-out transform tests, analysis dependency-version checks and clearer
sampling prose.

[open] Severe positive scaling saturates the unscaled C grid; severe mean
injection causes two unscaled final fits to hit max_iter. Results and warnings
remain visible rather than changing the protocol after seeing outcomes. The
rare-scene classes have insufficient training support. Main counts/IDs and
probe calibration need a separate decision; C2–C4 are not started. This is an
exploratory instrument pilot, not a new headline finding.

## Phase-C C0 integration (2026-10-07)

Added an external pretrained encoder path without changing the seven training
conditions, shared Trainer, Phase-B filesystem loader, or diagnostics.
Transformers 5.19.0 and pylance 0.39.0 are locked through the project workflow;
existing package versions and the cu128 Torch policy remain unchanged.
The model and dataset were already local; no download tooling was added.

[ours] The pinned processor/configuration load offline with audio_config=None.
The text+vision model contains 438,760,448 parameters. The MSI GPU is sm_61,
which the locked Torch build does not support; C0 therefore runs CPU FP32.
The ordinary suite exposed an existing accelerator test that treated visible
CUDA as usable. It is now explicitly opt-in, keeping default CI CPU-only.

Independent review caught a provenance defect: replacing a local Lance
dataset could preserve path, version, row count, and selected metadata while
changing its bytes. Optional repeat inference was insufficient to protect
resume. Completed raw image hashes are now checked against source bytes on
every resume before any new inference. A regression test first failed on
the original implementation, then passed with mandatory source verification.
Actual decoding/preprocessing dependency versions and concrete processor
classes are also recorded, alongside lock/source and weight hashes.

The C0 lexicographic sample is for integration only. The existing degradation
helper's probes are in-sample; C1 must use the underlying probe implementation
with separate train/validation vectors and train-fitted transformations.
No semantic study, Matryoshka experiment, numerical study, or cross-modal
retrieval was started. Measurements are recorded in
`experiments/phaseC_c0/smoke.json`; protocol and commands in [phase-c.md](phase-c.md).

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
  because repeated API outages blocked subagent dispatch: `6ae2e69` (HTML escaping),
  `51184f4` (the global-RNG fix), `ac07725` and `f9c4f46` (test hardening). All four were
  subsequently covered by the final whole-branch review, which found no issues in them.
  (These IDs were updated on 2026-09-27, when the history was rewritten to change the
  author email; the commits are otherwise unchanged.)
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
seeds; fixed-step endpoints and (initially) a claim rule later withdrawn; per-phenomenon reporting
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

## Corrections to earlier entries (2026-08-14, after the second audit)

Earlier entries stand as written — they record what was believed at the time. These are
the later corrections, not edits to the originals:

- **"EMA re-collapses" (32k entry) is too strong.** What was observed is angular
  reconcentration (cosine 0.294 -> 0.530) alongside a 0.021 probe decline. The stress test
  since showed those axes move independently, so semantic collapse was never established.
  Read it as angular reconcentration with a probe decline.
- **"lambda 0.1 borderline collapsed" is wrong.** Its total variance ended at 1.03x
  initialisation — above where it started. Only the angular axis distinguished it; on the
  scale axis nothing degenerated. lambda 0.05 (0.22x) genuinely contracted.
- **"cosine 1.000 means collapse" is unsafe throughout.** Mean injection drives cosine to
  1.000 with retrieval untouched at 1.000. In `none_nostopgrad` collapse and high cosine
  co-occurred; the log inferred one from the other, which does not follow.
- **The CIFAR SSL stream is NOT disjoint from probe-training images.** Both draw from the
  same 50,000 training images; only the probe's test split is held out. SSL uses no labels,
  but the report described a disjointness that does not exist.
- **The claim heuristic (2x seed SD and binomial SE) was not a valid paired analysis** and
  has been removed in favour of paired Student-t intervals over the five pre-declared
  contrasts.
- **Sweep and pilot results are exploratory Phase-1 findings**, single-seed and
  no-projector, not claim-grade evidence for any research question.

The second audit also caught what the first could not: that the eval split moved with the
training seed, and that the runner truncated `train.log` before the child's guard could
refuse. Neither is visible from reading a metric; both needed someone to ask what the
experiment was actually measuring and what the orchestration actually did.

## Post-audit exclusion sensitivity — 2026-10-08

[ours] Implementation/protocol `2feef78`, pre-execution sample/freeze `97595f3`.
The audit's exact 17 generated validation graphics were removed from queries and
galleries, with 2,000 unchanged training rows and 983 retained validation rows.
All 46 original C1/C2/C3 conditions completed in 731.693 seconds, peak process
RSS 874.43 MiB. C2 saved training PCA was reused; historical train-only C choices
were reused and final models refitted because classifier files were unavailable.
Refit training accuracy, iterations and convergence match historical diagnostics.
No model inference, source-image access, sampling or dependencies were changed.

[ours] 491 accepted historical experiment artifacts and all canonical chunks
were preserved. Completed resume: zero endpoints/probe fits, 3.721 seconds,
unchanged result/report/freeze/original verification bytes. Checks: 458 passed,
one optional accelerator skip; Ruff/lock validation passed. Source and current
status were updated separately; the historical source disclosure was not edited.

[interpretation] Pristine absolute utility/floors shift with the cohort, and some
weather BA contrasts change by over two points. Descriptive diagnostic invariances,
noise rank/utility dissociation, compression construction differences and C3
margin/turnover behavior remain visible. That does not establish equivalence or
independent replication. The current retained-fragment qualification and pilot's
six unresolved appended IDs remain explicit. See the separate
[interpretation](../experiments/phaseC_source_sensitivity/interpretation.md).


## Original COCO paired-data compression — 2026-10-09

[ours] Starting accepted `96a28b9`, bounded implementation/final protocol `141ebce`,
pre-inference source/sample/input freeze `2eee895`, passed offline contract
`7cd2c17`, complete canonical binding `b24bf28` before endpoint execution.
Original COCO2017 validation archives were acquired from the certificate-valid
same publisher-named S3 bucket (documented alias TLS transport amendment).
All 5K source image members and annotation joins validated; 1K selected contact-
sheet/caption bundles inspected by the assistant, with full coverage and stated
thumbnail limits. Original IDs/text, errors and repeated captions were retained.
No accepted BDD artifact, canonical chunk or dependency was changed.

[ours] CPU FP32/batch1/four threads, text+vision/no audio, pinned model and
mean-pooling/L2; role-specific frozen caption prefixes. Contract smoke took
272.823 s; remaining 984 image + 4920 query + 4920 document entries took
14,507.062 s, combined 4.106 h, peak RSS 3,571.05 MiB. Image 12.980 s/new row;
query/document 0.17736/0.17525 s/new row. No consecutive excessive swap windows;
resource gate passed. Separate source validation 57.328 s and interleaved
inspection window 1,312.352 s are not encoder extraction costs.

[ours] Native/MRL256/MRL128 primary T2I Hit@10 97.160/96.960/93.820%; paired
native-relative differences −0.200 pp [−0.540,+0.140], −3.340 pp
[−4.060,−2.620]. I2T Hit@10 98.400/98.100/95.900%, set recall@10
80.240/78.500/68.900%; category macro P@10 48.842/48.125/44.880%.
Native pairing shift Hit@10 1.220% T2I and 0.200% I2T is retained, including
its below-floor secondary result. Tables contain all supports, ties, raw and
normalized descriptive ranks, overlaps and paired intervals. Recorded condition-
evaluation/interval phase 11.348 s excludes preflight/loading/native references.

[ours] Completed resume performs zero inference/model loads and zero endpoint,
reference-ranking or interval calls. Protected 1,260 historical/cache files and
11,003 paired canonical files verified; pre-resume artifacts unchanged. Full
post-extraction verification: 497 passed, one optional skip (37.36 s), Ruff and
lock check passed. Final documentation/results verification is recorded separately.

[interpretation] Category and recorded-pair responses agree broadly rather than
show a strong reversal; 256d equivalence remains unresolved, 128d loss is larger,
especially when counting recovery of five captions. Three dimensions cannot
validate a general geometry predictor; same-encoder transfer is not independent
encoder replication. Unknown pretraining, incomplete relevance and near/event
relationships remain. Preserve current 983-row BDD qualifications and six pilot
IDs. Stop; any independent encoder replication requires new scope.

[ours] Final verification: 497 tests passed, one optional skip (34.31 s);
Ruff, lock check, whitespace check and local document links passed. Accepted
C1/C2/C3/pilot/sensitivity artifact diffs from `96a28b9` are empty. Completed
resume preserves all 18 pre-resume artifacts and verifies all protected hashes.

## 2026-10-09 — Jina paired replication stops at its offline loading gate

[ours] Starting `ebe17fb` on `phaseC-paired-replication`, source correction
`b4c2c97` separates tagged upstream Transformers files from locally renamed
research copies. Tested implementation `35bdf75` uses a separate locked4.49.0
extraction environment; root environment/lock unchanged. Four exact repository
revisions acquired selectively; advertised CLIP safetensors digest verified;
no nested text weights. Two recorded resolution-only wrapper amendments bind
nested config/code revisions. Original source snapshots remain unchanged.

[ours] Pre-forward freeze `8557b78` binds inputs, versions, source/code/model/cache
hashes, unchanged conditions and resource limits. Offline model construction
fails at the pinned nested model's internal tokenizer lookup without code-revision
propagation (`modeling_xlm_roberta.py:479`). Encoder forwards0; scientific
endpoints0; extraction-stage20.623s, partial peakRSS736.785MiB. Neither full-wrapper
fit nor numerical/throughput agreement was tested. Error and traceback retained
in the new namespace; no alternate loading, precision, resolution or revised
freeze was attempted. Accepted BDD/COCO results and qualifications are preserved.

[interpretation] This is a source-resolution compatibility failure, not a failed
compression replication or an encoder-quality finding. A separately reviewed
resolution-only remedy and new pre-forward freeze would be needed to retry;
no further execution is authorized by this checkpoint. Next reporting deliverable
is synthesis of completed evidence with this blocker stated plainly.

[ours] Final verification:529 root tests passed, one optional skip (34.24s);
27 isolated tests, Ruff, root/isolated lock checks, whitespace and new document
links passed. Frozen input/code/source/model/runtime and historical/cache hashes
verified; historical scientific artifacts and root dependency files have empty
diffs from `ebe17fb`. The branch remains local; no push occurred.

## 2026-10-09 — One bounded Jina tokenizer repair, second compatibility stop

[ours] From `56e9765`, tested repair `4d0b322` and separately committed
superseding pre-forward freeze `2079de9`. The old attempt/freeze remain unchanged.
The constructor failure was reproduced from its exact expression; original three
v3 tokenizer files at its pinned revision resolve it offline without author
numeric changes or outer substitution. All5000 nested-helper inputs and original
outer frozen tokens checked. Root/isolated locks and scientific rules unchanged.

[ours] Metadata-only full architecture verifies999 state keys,865,278,477
parameters with no missing/unexpected/shape-mismatched tensors. The non-scalar
sum matches the published865,278,476 exactly; the extra trained `logit_scale`
scalar is preserved. Actual loader/key/shape/tokenizer/source/pooling checks
proceed, then the all-CPU-FP32 parameter guard fails before any forward.
51.199s extraction-stage time, partial-process peakRSS4295.492MiB; no image
decode, canonical row, scientific endpoint or completed16-group contract.
Offending parameter details were not persisted; no exact dtype/device cause is
claimed. See the [closed outcome](../experiments/phaseC_paired_jina_repair/interpretation.md).

[interpretation] The declared second-blocker stop is respected: no alternative
loading, precision, architecture, limits or further model load. This attempt
establishes a compatibility failure, not a failed scientific compression finding.
Retain accepted BDD/COCO qualifications and prepare the completed-study synthesis;
audio–visual post-training belongs to its separate project.

[ours] Final verification:534 root tests passed, one optional skip (27.84s);
16 focused isolated tests passed. Ruff, root/isolated lock checks, whitespace and
new document links passed. Superseding frozen identities, all protected history,
original failed manifest/freeze and exact original inputs verified. Closed-attempt
re-entry guard rejects before a forbidden model loader; completed scientific
resumes are not applicable. Accepted artifact/dependency diffs from `56e9765`
are empty. Remote check finds no paired-replication branch; no push occurred.

## 2026-10-09 — Experimental scope closed; final synthesis and local preview

[ours] The wrap-up starts from `e902aa0` and changes presentation and documentation
only. The public story answers four questions: geometry versus utility, compression
responses across endpoints, numerical neighbour identity, and recorded positives
versus first-rank recovery and multiple-caption coverage. The current BDD tables
use the 983-row exclusion sensitivity. The source correction, retained fragment
provenance, unresolved six pilot IDs, corrected Phase-B probing and separate
uncertainty scopes remain explicit. Original C1/C2/C3 and both failed Jina attempts
are preserved. No new scientific endpoints, inference, image decoding or experiment
were executed. No Jina quality or compression claim is supported.

[ours] `viz/evidence-lock.json` binds accepted committed artifacts at `e902aa0`;
the site reader validates file and record checksums and emits static tables,
a COCO endpoint figure, compact display JSON and locally readable evidence records.
LaTeX tables use the same reader and displayed values. The final report replaces
stale draft placeholders; the provenance appendix explicitly separates frozen,
post-hoc and exploratory decisions. Legacy draft sections remain in Git history
and the repository, outside the current report build.

[ours] Verification:539 CPU/model-free tests passed, one optional accelerator skip;
Ruff and `uv lock --check` passed. Site build, internal file/fragment links,
updated Markdown links, generated LaTeX-table consistency and whitespace checks
passed. All six unique external page links returned HTTP200 at inspection.
11,003 COCO canonical file hashes,750 BDD canonical chunk hashes and491 frozen
historical artifact hashes matched. Scientific artifact/source/dependency diffs
from `e902aa0` are empty. The older sensitivity freeze's execution-code hash is
not expected to match subsequently accepted adapter revisions; it is not used
as a current-code identity check.

[ours] Firefox157.0.1 headless checks covered1440,768 and500 CSS-pixel layouts,
plus a390px same-origin frame for the phone viewport (headless outer windows have
a500px minimum). No overall horizontal overflow was observed; all four historical
charts loaded. Keyboard checks covered skip navigation, mobile contents expansion,
Tab/activation/closure, attribute selection and horizontal table scrolling.
Local preview screenshots and browser/link ledgers are in ignored `viz/dist/`.
No automated comprehensive assistive-technology audit is claimed.

[open] `latexmk`, `pdflatex` and `tectonic` are unavailable on this host. Report
inputs and generated tables were checked, but the PDF was not compiled. No
packages were installed to mask this limitation. The site builds with the
standard library and is ready for local review; it was not pushed, published or
deployed. Experiments are closed, with maintenance and transparent corrections
still possible. Audio–visual post-training belongs to a separately scoped project.


## 2026-10-09 — User-authorized publication of the closed study

The user authorized pushing and publishing the reviewed wrap-up `994937b`.
Release wording distinguishes the preserved pre-publication verification from
the GitHub Pages publication stage. The research branch and `main` receive the
accepted history by fast-forward; no scientific records, source code, caches or
root dependency files are changed. Experimental scope remains closed.
