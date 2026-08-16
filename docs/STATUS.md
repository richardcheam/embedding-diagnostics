# Where this project stands

Read this first. It is the plain-language story of what we set out to test, what actually
happened, and what is still open. Everything else is detail.

If the terms below are unfamiliar — embedding, collapse, linear probe, stop-gradient —
read [`primer.md`](primer.md) first. It builds all of them from scratch and takes about
half an hour.

Last updated: 2026-08-16, after the Phase-B pilot on BDD100K (3 seeds).

---

## Terminology: what the control condition actually is

The `none_nostopgrad` control is **not** an exactly constant encoder. Its embeddings have
measurable, consistent, non-zero variance (2.1–2.3e-4 across five seeds; minimum per-feature
std 7.4–8.4e-4). It is **severely scale-contracted and angularly concentrated**: the cloud
of points has shrunk by four to five orders of magnitude and every point now aims in nearly
the same direction.

That distinction is load-bearing, so this document tries to keep six things separate:

| | meaning |
| --- | --- |
| exact constant collapse | E(x) = c for every input; no input-dependent structure at all |
| near-constant scale contraction | structure survives, shrunk by orders of magnitude |
| angular concentration | vectors aim the same way; magnitudes may still differ |
| dimensional / rank collapse | spread survives in only a few directions |
| loss of downstream information | the representation no longer supports the task |
| numerical fragility | structure survives in float64 but may not in float16 or int8 |

These are not interchangeable, and earlier versions of this document used "dead", "fully
collapsed" and "constant" as if they were. Whether our control has lost *downstream
information* is currently unmeasured — see the probe defect below.

---

## 0. The reframe (read this before the history below)

An external advisor audit found two design flaws that change how everything below must be
read, and the project's goal (interview preparation for an ADAS-validation CIFRE, a
credible portfolio piece) pointed at a sharper question. The project is now:

> **Can you trust a self-supervised scenario embedding?** Which label-free diagnostics
> reliably detect each mode of representation degeneration — and do they still work when
> the embedding is used for driving-scenario retrieval?

The four bugs and the collapse conditions stop being the contribution and become the
**instrument**: controlled ways to manufacture known degeneration so diagnostics can be
tested against ground truth. Phase A calibrates on CIFAR-10; Phase B runs the same matrix
on BDD100K driving images with scenario-attribute retrieval as the semantic endpoint.

The audit's two design findings, both now fixed in code:

1. **SIGReg had no projector.** LeJEPA applies SIGReg to a disposable MLP head and probes
   the encoder beneath it; we applied it directly to the probed representation. The
   projector is worth ~20 points in comparable settings, so every "SIGReg didn't learn"
   result below characterises the no-projector configuration — not SIGReg, not LeJEPA.
   There are now `proj_sigreg_*` conditions with a 512-d projector.
2. **The 2x2 had a hole.** `none_stopgrad` (stop-gradient alone, no regularizer) was
   missing, so stop-gradient's effect could not be separated from SIGReg's. Added.

Plus the statistics fixes: paired seeds (registered at 3, run at 5), fixed-step endpoints instead of
best-over-checkpoints, paired Student-t 95% intervals (the 2x-seed-SD/binomial
SE), and the rank metric renamed to what it actually is (participation ratio, with RankMe
added as a genuinely different measure).

Full pre-registration: `docs/superpowers/specs/2026-08-14-phase2-scenario-reframe.md` and
the report's Section "Phase-2 pre-registration".

### The stress test already answered part of the question, with no GPU

Applying *known* degradations directly to embedding matrices (rather than inferring modes
from training runs) produced the project's first ground-truth result:

| transformation | total var | cosine | retrieval P@10 |
| --- | --- | --- | --- |
| scale contraction 1e4x | 1106 -> 0.11 | unchanged | **1.000 -> 1.000** |
| mean injection | unchanged | 0.086 -> **1.000** | **1.000 -> 1.000** |
| interpolate to the mean | 1106 -> 0.11 | 0.086 -> **0.999** | **1.000 -> 1.000** |
| isotropic noise | 1106 -> 1.2e7 | 0.086 -> 0.000 | 1.000 -> **0.125** |

**No geometric diagnostic tracks semantic content.** Everything that looks like collapse —
variance down four orders, cosine at 1.000 — can happen with retrieval perfectly intact.
The one transformation that destroyed retrieval moved variance and rank the *other* way.

So "cosine near 1" and "variance collapsed" are statements about geometry, not about
whether the representation is still useful. Sections 3-5 below were written before this was
known and should be read with it in mind.

### The prescription: a minimal sufficient panel

Everything above is a negative result — these diagnostics cannot be trusted. The
constructive version, and the thing worth taking away:

**No single label-free diagnostic covers every collapse mode. Two do. Use total variance +
RankMe.**

That is not a preference; it is an exhaustive search over subsets of five candidate
diagnostics against the five controlled degradations, reproducible with
`uv run python scripts/validate_panel.py`. Every singleton fails, because each candidate is
invariant to something:

| degradation | detected by | blind |
| --- | --- | --- |
| scale contraction | total variance, feature std | cosine, RankMe, participation ratio |
| mean injection | cosine, RankMe | **total variance**, participation ratio, feature std |
| rank truncation | all five | — |
| isotropic noise | all five | — |
| mean interpolation | all but participation ratio | participation ratio |

The two blind rows are complementary and that is the whole argument: the *scale-invariant*
metrics cannot see pure contraction, and the *centered* metrics cannot see the cloud
shifting off the origin. A sufficient panel needs one from each family. Exactly four pairs
qualify; **the participation ratio appears in none of them.**

**Two things had to be got right, and both were surprises.**

*Drift alarms do not work.* The obvious monitoring rule — alarm when a diagnostic moves far
from its value at initialisation — produced **5 false alarms out of 6 on CIFAR-10 and 4 out
of 6 on BDD100K**, including on `ema_stopgrad`, the only condition that actually learns
(total variance 2.2×, cosine 0.40× from init). Healthy self-supervised training legitimately
reshapes the geometry, so movement carries almost no signal. The panel uses **absolute**
limits near each metric's degenerate floor instead.

*Absolute limits work, cleanly.* Zero misclassifications across all 14 condition-dataset
pairs, with large margins:

| metric | collapsed | worst run that trains | margin |
| --- | --- | --- | --- |
| total variance | 0.0001–0.0002 | 0.69 (CIFAR) / 4.17 (BDD) | 3,466× / 41,716× |
| feature std | 0.0006–0.0011 | 0.057 / 0.097 | 54× / 168× |
| RankMe | 1.06–1.12 | 9.67 / 9.72 | 9× |
| cosine | 1.0000 | 0.85 / 0.64 | thinnest |

**What the panel does and does not claim.** A tripped check means the embedding has entered a
regime where your *scale-invariant* metrics — standardized probes, cosine retrieval — are no
longer trustworthy. It does **not** mean the representation carries no information: three of
the five controlled degradations leave retrieval P@10 at exactly 1.000 while wrecking the
geometry, because the residual structure still orders neighbours correctly. Confirming
actual information loss needs an unstandardized probe. The panel also does not detect
*failure to learn* — `sigreg_nostopgrad` never beat its own initialisation and is correctly
called healthy, because an untrained encoder is not a degenerate one.

**Thresholds are calibrated, not universal.** They come from this project's own runs and sit
midway in log space between the collapsed control and the worst genuinely-training
condition. Recalibrate for a different architecture or embedding dimension. And the panel
has only been validated against *total* collapse — partial degeneration is untested, and the
cosine margin is thin enough that it would likely be the first check to fail there.

---

### Phase B pilot: the blindness reproduces on driving data, and it is worse

**Status: BDD100K pilot complete** — 7 conditions × 3 seeds, 4,000 steps. This is a
*pilot*, not the claim-grade Phase-B run: 3 seeds, and it was launched to size the budget.
Read it as a strong indication, not a result. The full 5-seed matrix is still to run.

The Phase-A finding was that a collapsed encoder scores 0.413 on the standardized probe
against a 0.100 chance floor. The obvious objection is that CIFAR-10 is a toy. On driving
data the effect is **larger**:

| condition | total var | cosine | probe uns. | **probe std.** | retr. P@10 | RankMe | PR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ema_stopgrad` (healthy) | 108.6 | 0.256 | 0.769 | **0.768** | 0.676 | 78.9 | 11.2 |
| `none_stopgrad` (weakest real) | 49.1 | 0.549 | 0.735 | **0.738** | 0.628 | 9.7 | 1.6 |
| `none_nostopgrad` (**dead**) | 0.0001 | 1.000 | 0.564 | **0.737** | 0.566 | 1.1 | 26.9 |
| *floor* | — | — | 0.564 | 0.564 | 0.429 | — | — |

**The standardized probe cannot separate a degenerate encoder from a real one at all.** Dead
0.7375 versus the weakest genuinely-training run at 0.7383 — a difference of **−0.0008**
against a 2×SD threshold of 0.0116. Not separable, and consistent across all three seeds.
On CIFAR-10 the dead-to-healthy gap was 0.128; here it is **0.030**, four times smaller.

**Retrieval is dangerous rather than blind.** It does separate them (−0.062, 2×SD 0.056 —
just clears), so it is better here than on CIFAR where it could not. But the absolute
reading is the problem: a degenerate encoder retrieves at **1.34× chance on weather and 1.57× on
timeofday**. Seeing P@10 = 0.544 on weather against a 0.406 floor, nobody would suspect the
encoder had collapsed to a single point.

**The participation ratio is confidently inverted.** It reads 26.9 on the degenerate encoder
against 11.2 on the healthy baseline — ranking the degenerate encoder second of seven, above every real
run, by a margin that clears the claim rule in the wrong direction.

**What still works:** total variance (0.0001), mean pairwise cosine (1.0000), RankMe (1.06),
and the unscaled probe — which lands on **0.5635 against a majority floor of 0.5636**, exact
to four decimals in all three seeds. Perfect detection.

So the label-free panel that survives both datasets is: **total variance + RankMe +
the unscaled probe.** Standardized probing and the participation ratio fail on both, and
retrieval — the operation scenario mining actually performs — reads comfortably above
chance on an encoder with no information in it.

**Budget note.** Balanced accuracy on weather (the endpoint that actually moves) gained
+0.041 over the first 1,000 steps and +0.009 over the last 1,000: decelerating but not
flat. 4,000 steps captures ~91% of the learning. With fixed compute, spend it on **seeds
rather than steps** — the claim rule is driven by across-seed spread, so more seeds buy
credibility that longer training does not.

---

### Phase A is done, and it answered the question

> ⚠️ **Unstandardized probe endpoints are provisional and under recomputation.** The probe
> used during these runs stops before fitting on severely scale-contracted features,
> predicting the majority class and reporting chance with no warning. Geometric endpoints
> and retrieval are unaffected. See the report's Provenance section.

**Status: Phase A complete** — 7 conditions × 5 paired seeds on CIFAR-10, endpoints read at
the pre-registered final checkpoint, analysed with paired Student-t intervals. Phase B
is next; BDD100K is confirmed present on the GPU box (70k train / 10k val,
per-image-JSON layout, 61,591 + 8,801 fully labelled).

**The finding that matters — half of it withdrawn on 2026-08-16.** The control condition
with no collapse prevention (`none_nostopgrad`) degenerates until the cloud of embeddings is
essentially one point: total variance 0.0002 — four hundred thousand times below the
baseline — and mean pairwise cosine exactly 1.0000. Those two are pure arithmetic on the
embeddings and stand.

⚠️ **What does not stand:** the row below reading "unscaled linear probe 0.1066 → correct".
The probe's optimiser stops when the slope of its objective gets shallow, and the slope
scales with how large the input numbers are. At this encoder's scale (~0.001) the slope is
already below the threshold before a single step, so the probe never fits, predicts the most
common class, and returns chance — silently. Verified: on synthetic data where the labels
are perfectly recoverable *by construction*, at the same scale, the unscaled probe reads
0.096 and the standardized one reads 1.000. And the tell is in our own numbers: 0.1066 sits
on the always-guess-the-majority score of 0.1088. **Whether a degenerate representation
retains usable information is currently unmeasured**, not resolved either way.

| metric on the fully-collapsed control | reading | verdict |
| --- | --- | --- |
| total variance | 0.0002 | correct |
| mean pairwise cosine | 1.0000 | correct |
| unscaled linear probe | 0.1066 (chance = 0.100) | correct |
| RankMe | 1.12 | correct |
| **standardized linear probe** | **0.413** | **blind** |
| **cosine retrieval P@10** | **0.184 (1.8× chance)** | **blind** |
| **participation ratio** | **34.98** (healthy baseline: 26.90) | **blind, and inverted** |

The degenerate encoder scores 0.413 on the standardized linear probe — **second of all
seven conditions**, behind only the EMA baseline. It retrieves at 1.8× chance,
indistinguishable from a partially-working encoder (paired difference +0.002). Retrieval is
arithmetic (sort by angle, count matching labels) so that part stands; the probe comparison
is what is under recomputation.

⚠️ **The mechanism we gave for this is also under revision.** We wrote that standardization
and L2-normalization both "rescale floating-point noise back to readable magnitude". Those
are not the same operation and that phrasing was never derived. Standardization divides each
feature by its own standard deviation, which does amplify small per-coordinate differences.
L2-normalization divides each vector by its length, which does not amplify anything — it
removes overall size and can preserve tiny angular orderings that were already there. Any
surviving structure may be genuine angular structure rather than numerical noise. The
retrieval-robustness study (does the ordering survive float16, int8, approximate search?)
is what will settle which it is.

What makes this worth reporting is the *reach*, not the mechanism. Standardized probing is
the default SSL evaluation and cosine similarity is the default in essentially every vector
database. **A pipeline evaluated only that way cannot tell a working encoder from one that
has collapsed by five orders of magnitude.** And for scenario mining specifically: retrieval
is the operation the application actually performs, and it is the endpoint least able to
detect this. The fix is cheap — log total variance and the unscaled probe alongside — but
it has to be done on purpose.

**Second finding: geometry and semantics fail to resolve in opposite places.** Across the
four SIGReg contrasts, every geometric comparison clears the claim rule and is enormous
(169 units of variance, 0.84 of cosine) while the probe moves at most 0.035 and clears the
rule once. In the one contrast whose semantic difference is unambiguous — `none_stopgrad`
vs the collapsed control, +0.210 probe accuracy — *neither* geometric endpoint clears the
rule, because `none_stopgrad` is wildly seed-unstable (total variance across seeds: 17.1,
7.5, 16.9, 50.9, 15.4). Where geometry is measured precisely it says nothing about
semantics; where semantics differ hugely the geometry is too unstable to say so. This is
the stress-test result (below) surviving in real training under a rule fixed in advance.

Caveat that limits it: absolute detection still works. Nobody would look at cosine 1.0000
and call it healthy. What fails is using these diagnostics to **rank or compare**
configurations.

**Third finding: RankMe and participation ratio are not interchangeable.** Each is blind to
the mode the other catches — PR reads 34.98 on the collapsed-to-a-point control (RankMe:
1.12, correct), while RankMe reads 39.60 on `sigreg_nostopgrad` whose variance is 127×
below baseline (PR: 16.63, correct). PR uses the *centered* covariance spectrum so it
cannot see collapse to a non-zero constant; RankMe uses raw singular values so it can, but
SIGReg keeps its shrinking residual isotropic, which flattens the raw spectrum. Both are
called "effective rank" in the literature. They should not be substituted for each other.

**And: nothing except the EMA baseline learned.** Only `ema_stopgrad` finished above its own
random initialisation (+0.161). Every SIGReg arm finished below it. That is a statement
about SIGReg paired with masked latent prediction — not about LeJEPA, which pairs it with
multi-view invariance and no predictor. See section 5.

One deviation from the pre-registration, in the strengthening direction: it fixed 3 seeds,
we ran 5. Nothing else changed. The 3-seed pilot is retained separately as `phaseApilot`
and none of its numbers appear above.

One measurement issue surfaced while verifying the data, and it is the ADAS point in
miniature: BDD's scenario attributes are severely skewed (val weather 61% `clear`, `foggy`
13 images; scene 61% `city street`, `gas stations` 7). So always guessing the majority
class already scores 0.53-0.61, and *random* retrieval scores 0.41-0.46 — not the 0.17 you
would assume from 1/6 classes. A P@10 of 0.50 would look like triple chance and be barely
above it. Every number now carries its floor, balanced accuracy is logged alongside raw,
and classes too rare to score are counted rather than quietly averaged away. **The rare
scenario classes are the ones validation cares about, and they are exactly the ones the
metrics cannot see** — which is a finding worth reporting, not a nuisance to hide.

---

## 1. What we set out to test

Self-supervised models like JEPA can cheat. If nothing stops them, the encoder learns to
output the *same vector for every input* — the prediction task becomes trivially easy and
the representation becomes worthless. This is **collapse**.

The field prevents it with a bag of tricks: **stop-gradient** (don't let gradients flow
through the target branch), **EMA target encoders** (make the target a slow-moving copy),
masking design, and so on. They work, but nobody can prove why.

A 2025 paper, **LeJEPA**, proposed **SIGReg** — a regularizer that pushes embeddings
toward an isotropic Gaussian — and reported that it makes those tricks unnecessary.

We wanted to test two things:

- **Part 1:** does stop-gradient still matter once SIGReg is doing its job?
- **Part 2:** do cheap collapse diagnostics (variance, cosine similarity) degrade *before*
  an expensive linear probe does? If so, you could monitor training cheaply.

The original design used **four conditions** (three more were added after the audit; see section 0). The Phase-1 evidence below comes from these four:

| condition | EMA target | stop-gradient | SIGReg | role |
| --- | --- | --- | --- | --- |
| `ema_stopgrad` | yes | yes | no | classic JEPA baseline |
| `sigreg_stopgrad` | no | yes | yes | both mechanisms |
| `sigreg_nostopgrad` | no | no | yes | **LeJEPA's recipe — the thing under test** |
| `none_nostopgrad` | no | no | no | **control: nothing prevents collapse, must collapse** |

---

## 2. What is established, and at what strength

Four tiers. Nothing moves up a tier without the evidence that tier requires.

### Tier 1 — validated engineering facts

- The SIGReg loss matches the reference implementation bit-for-bit on isotropic,
  anisotropic, collapsed and mean-shifted inputs.
- The four conditions run through one shared training loop; within a seed they receive
  identical initialisation, masks and batch order (test-enforced).
- The evaluation split is fixed by `eval_split_seed`, independent of the training seed.
- 232 tests pass; the run planner refuses to overwrite existing results.

### Tier 2 — ground truth from the controlled stress test (no training involved)

**No geometric diagnostic tracks semantic content.** Applying known transformations
directly to embedding matrices: contracting by 10,000x, or driving mean pairwise cosine to
1.000, leaves retrieval P@10 at exactly 1.000. Isotropic noise — the transformation that
*does* destroy retrieval (1.000 -> 0.125) — raises total variance and both rank measures
instead of lowering them.

This is the strongest evidence the project holds, because the transformation is known
exactly rather than inferred from an optimisation.

### Tier 3 — exploratory single-seed observations (NOT claim-grade)

All from one seed, at one scale, with **no projector** — the regularizer acting directly on
the probed representation:

- Across LeJEPA's swept lambda range, no-stop-gradient arms concentrated angularly while
  stop-gradient arms stayed spread. At lambda 0.1, however, total variance ended *above*
  initialisation, so that arm was not contracting at all.
- On the one 32k run that learned then degraded, the probe peaked 2,500 steps before the
  cosine bottomed.
- No SIGReg configuration exceeded random-initialisation probe accuracy.

Tier 2 constrains how these may be read: cosine near 1 is angular concentration, which is
not by itself information loss.

### Tier 2b — Phase-A results under paired Student-t intervals

7 conditions x 5 paired seeds, CIFAR-10, endpoints fixed in advance. These are the
strongest *training* results the project holds; only the stress test (Tier 2) is stronger,
because there the transformation is known exactly rather than produced by an optimiser.

- A severely contracted encoder scores 0.413 on the standardized probe and 1.8x chance on
  cosine retrieval, while the unscaled probe correctly reads chance. Both blind protocols
  are the field defaults.
- Geometric contrasts clear the claim rule in 8 of 10 cells; the primary semantic endpoint
  in 2 of 5 — and the two axes fail in opposite places.
- RankMe and the participation ratio are each blind to the collapse mode the other detects.
- Only `ema_stopgrad` beat its own random initialisation.

Detail and exact numbers: section 0 above, and the report's Results section.

### Tier 4 — pre-registered and not yet run

Phase B: the same matrix on BDD100K with scenario-attribute retrieval as the primary
semantic endpoint. Endpoints, contrasts, and falsification criteria are fixed in advance;
see the spec and the report's pre-registration section. **No Phase-B result exists yet.**

---

## 3. Part 1: does stop-gradient still matter? — **Yes**

We swept SIGReg strength (lambda) across the range LeJEPA itself uses, running both
conditions at each value for 2,000 steps. Final mean pairwise cosine similarity — **near 0
means healthy and spread out, near 1 means collapsed**:

| SIGReg strength | **without** stop-gradient | **with** stop-gradient |
| --- | --- | --- |
| lambda 0.01 | **0.990** collapsed | 0.034 healthy |
| lambda 0.02 | **0.951** collapsed | 0.021 healthy |
| lambda 0.05 | **0.765** collapsed | 0.009 healthy |
| lambda 0.10 | 0.381 — but variance *rose* to 1.03x init, so not contracting | 0.008 healthy |

Two things to read off this:

1. **With stop-gradient, nothing ever collapses** — at any strength. Cosine stays at
   0.008–0.034 throughout.
2. **Without it, everything collapses**, and more SIGReg helps *monotonically* but does not
   rescue it inside the tested range on the angular axis. At lambda 0.1 the scale axis
   shows no degeneration at all (variance ends above where it began), so only the angular
   measure separates it from the stop-gradient arms. Stronger settings lie above the range
   LeJEPA sweeps and are untested.

So within LeJEPA's own hyperparameter range, on our setup, **SIGReg alone did not replace
stop-gradient.**

---

## 4. Part 2: do cheap diagnostics warn earlier? — **No, on the evidence so far**

This needs a run that both *learns* and then *degrades*. Only one condition did:
`ema_stopgrad` at 32,000 steps.

- Probe accuracy peaked (0.603) at **step 11,000**, then slipped to 0.582.
- Cosine similarity bottomed (0.294) at **step 13,500**, then rose to 0.530.

**The probe turned 2,500 steps earlier than the cheap diagnostic.** That is the opposite of
the hypothesis.

What the cheap diagnostic *does* give is a much louder signal once it moves — cosine shifts
by 0.236 against a probe decline of 0.021. So it tells you **how bad** things are, more
clearly, but **later**.

Confidence: low-to-moderate. One condition, one seed, and 2,500 steps is only five
checkpoints apart.

A separate observation worth noting: **the healthy baseline shows angular reconcentration
in its second half** (cosine rising, probe declining — not established as semantic
collapse, since Tier 2 shows those axes move independently). It peaks a third of the way through the budget and slowly degrades after. If
you only care about the best representation, 32,000 steps is too many.

---

## 5. The big caveat on all SIGReg results

**In our setup, SIGReg never produced useful representations at any strength.** Every
SIGReg run finished at or below its own random initialisation (~0.369 probe accuracy),
while the EMA baseline reached 0.603.

Before reading that as "SIGReg doesn't work", note that **we are not running LeJEPA.** We
combine SIGReg with I-JEPA-style *masked latent prediction*. LeJEPA combines it with
*multi-view invariance* (8 augmented views, no predictor, no teacher-student network).
Those are different systems that happen to share one component.

So the honest statement is: **SIGReg combined with masked prediction did not learn useful
features here.** That is a statement about the combination, not about LeJEPA.

---

## 6. Four bugs we found — and how

This is the part most worth learning from. Every one was found by *running* the experiment,
not by reading the code, and every one had passed the tests we had thought to write.

**1. The collapse diagnostics could not detect collapse.** The control collapsed completely
(cosine 1.000) while two of our four metrics moved the *wrong way*: effective rank rose
2.71 → 30.29 and probe accuracy rose 0.375 → 0.409. Both are **scale-invariant**, and
collapse is a loss of scale. Effective rank ends up measuring the shape of numerical noise;
the probe's feature standardization rescales that noise back to full size. Fixed by adding
`total_variance` and an unstandardized probe.

**2. SIGReg was silently disabled.** We centered the embeddings before the normality test.
The reference implementation does not. A collapsed encoder emits some constant vector, and
centering maps that *exactly onto the origin*, where the loss is flat — so the loss stayed
loudly high while its gradient decayed to nothing. **Collapse had become a stationary point
of the objective built to prevent collapse.** Found by comparing against the reference line
by line; ours now matches it bit-for-bit.

**3. The SIGReg weight was ~20x too high.** LeJEPA uses a convex combination and sweeps
lambda in 0.01–0.1; we were running the equivalent of 1.0.

**4. The change-point detector assumed a flat baseline.** It took the first quarter of a run
as reference — but on a 32k run that window *is* the learning phase, so the tolerance band
was so wide that an obvious peak-and-decline registered as "no change". A second version
assumed each metric has a fixed good direction, which is also false: total variance falls
throughout a *healthy* run, so falling variance only means degradation when the probe is
falling too.

The recurring lesson: **every one of these assumptions looked reasonable when written.**
They failed on contact with the shape real data actually has.

---

## 7. What is still open

- **Does the blindness survive on driving data?** This is Phase B and the single most
  important open question. Phase A's mechanism argument (both protocols remove scale by
  construction) should transfer; the *magnitudes* need not. On BDD100K retrieval is the
  primary endpoint, which is the one Phase A found least able to detect collapse.
- **Does a stronger SIGReg eventually replace stop-gradient?** The trend says maybe, above
  LeJEPA's swept range. Testing lambda 0.2 / 0.3 / 0.5 would settle it and is cheap.
- **Can SIGReg learn anything in this setup at all?** No configuration tried has beaten
  random initialisation. If none does at any lambda, the honest conclusion is that masked
  prediction is the wrong companion objective for it.
- **Collapse here was induced, not spontaneous.** Every degeneration we measured came from
  removing stop-gradient or applying a known transformation. Whether the same diagnostics
  catch collapse that arises on its own, mid-training, is untested.
- **Part 2 deserves a fair test.** It was asked on one condition. It needs several runs that
  genuinely learn and then degrade; only `ema_stopgrad` ever did.
- **`none_stopgrad` is bimodal across seeds** (total variance 7.5 to 50.9). We report it as
  instability; we have not investigated what distinguishes the seeds.

---

## 8. Where to find more detail

| document | contents |
| --- | --- |
| `docs/development-log.md` | every finding, with the numbers and the reasoning |
| `docs/open-questions.md` | unresolved issues, and the full SIGReg validation record |
| `report/` | the LaTeX write-up; `make -C report` builds the PDF |
| `AGENTS.md` | rules for anyone (human or agent) working in the code |
| `README.md` | how to run things |

To look at results yourself:

```bash
uv run python scripts/summarize_sweep.py --tag sweep      # the lambda sweep table
uv run python scripts/make_figures.py --tag main          # figures into report/figures/
uv run python viz/build_report.py --tag main              # interactive page
```
