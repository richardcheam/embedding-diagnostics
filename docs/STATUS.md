# Where this project stands

Read this first. It is the plain-language story of what we set out to test, what actually
happened, and what is still open. Everything else is detail.

Last updated: 2026-08-14, after the external audit and the Phase-2 reframe.

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

Plus the statistics fixes: 3 paired seeds, fixed-step endpoints instead of
best-over-checkpoints, a pre-registered claim rule (2x seed SD and the probe's binomial
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

**Status: harness ready, Phase-A runs pending. BDD100K confirmed present on the GPU box**
(70k train / 10k val, per-image-JSON layout, 61,591 + 8,801 fully labelled).

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

### Tier 4 — pre-registered and not yet run

Phase A (7 conditions x 5 paired seeds, CIFAR-10) and Phase B (same matrix on BDD100K with
scenario-attribute retrieval). Endpoints, contrasts, and falsification criteria are fixed
in advance; see the spec and the report's pre-registration section. **No Phase-A or Phase-B
result exists yet.**

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

- **Does a stronger SIGReg eventually replace stop-gradient?** The trend says maybe, above
  LeJEPA's swept range. Testing lambda 0.2 / 0.3 / 0.5 would settle it and is cheap.
- **Can SIGReg learn anything in this setup at all?** If not at any lambda, the honest
  conclusion is that masked prediction is the wrong companion objective for it.
- **Everything rests on one seed.** The probe alone has a run-to-run noise floor of ~0.005,
  and no result here has been repeated across seeds. This is the single biggest weakness.
- **Part 2 deserves a fair test.** It was asked on one condition. It needs several runs that
  genuinely learn and then degrade.

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
