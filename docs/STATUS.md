# Where this project stands

Last updated: 2026-08-17, after the Phase-B reprobe.

[measurement] Phase-C extension updated 2026-10-07: C0 integration infrastructure is
implemented and smoke-verified, with local pinned EmbeddingGemma 2 image extraction, validated
resumable FP32 caching, and unchanged geometry diagnostics. C1-pilot is
complete and exploratory: 384 train / 128 validation, frozen uniform sampling,
held-out probes and a cached-vector stress sweep. C1-main is not frozen;
C2–C4 remain deferred. The [pilot report](../experiments/phaseC_c1_pilot/README.md)
records measurements and unresolved probe calibration/convergence limits. See [phase-c.md](phase-c.md) and the compact smoke record
at `experiments/phaseC_c0/smoke.json`. Existing Phase-A/B findings below are
unchanged; no Phase-C research conclusion has been drawn.

This is the plain-language account of what the project asks, what it found, what it had to
take back, and what is still open. If a term is unfamiliar, look it up in
[`metrics.md`](metrics.md) (formulas and worked examples) or read [`primer.md`](primer.md)
first (builds everything from zero).

---

## 1. The question

Scenario mining for ADAS validation embeds millions of unlabelled driving clips and then
asks geometric questions of them: *find me more situations like this one*, *does my test set
cover fog?* That only works if the embedding is trustworthy, and you have no labels to check
it with. So:

> **Which label-free diagnostics actually tell you a self-supervised embedding has gone
> wrong — and do they still work when the embedding is used for driving-scenario
> retrieval?**

Collapse-prevention mechanisms (stop-gradient, EMA, SIGReg, a projector) are the
**instruments**: they let us manufacture known degeneration on purpose, including one
condition designed to fail completely, so the diagnostics have something to be tested
against.

---

## 2. Where it stands in one paragraph

Phase A (CIFAR-10) and Phase B (BDD100K) are both complete at 7 conditions × 5 paired
seeds. An audit in August found that the linear probe used during training was defective,
and re-measuring Phase B from its saved encoders **refuted the project's original headline
claim**. The corrected result is more interesting than the one it replaced: a representation
can be degenerate by every geometric measure while remaining nearly as decodable as a
working one. Phase A's probe numbers are still uncorrected, because that campaign predates
encoder saving.

---

## 3. The finding

### 3.1 Set-up

**The three encoders quoted throughout, and the floors.** Every table below compares the
same three conditions against a baseline. If the columns do not immediately make sense,
[`primer.md` §7.0](primer.md) walks through reading one row out loud.

| column | condition | role |
| --- | --- | --- |
| healthy | `ema_stopgrad` | our best encoder — what good looks like |
| **contracted** | `none_nostopgrad` | the control, built to fail on purpose |
| weakest real | `none_stopgrad` | trains, but badly — the demanding comparison |
| floor | — | what you score by ignoring the input: guess the most common class (probe), or return random neighbours (retrieval) |

A score is only meaningful relative to its floor, and the floors differ per attribute
because the class balance does. `timeofday` is near 50/50 so its floor is 0.483; `weather`
is 60% clear so its floor is 0.604.

The `none_nostopgrad` control has no collapse prevention at all. It degenerated as designed,
and every geometric measure agrees it is in a terrible state:

| geometric measure | healthy baseline | **contracted control** | weakest run that still trains |
| --- | --- | --- | --- |
| total variance | 107.71 | **0.0001** | 39.61 |
| mean pairwise cosine | 0.256 | **1.0000** | 0.676 |
| RankMe | 78.36 | **1.06** | 7.45 |

Total variance a million times below baseline. Every embedding pointing the same way. One
effective dimension out of 192. By any geometric account this representation is destroyed.

### 3.2 The finding: it is not destroyed

Measured with a **working** probe, the contracted representation is about as linearly
decodable as a condition that genuinely trained:

| attribute | healthy | **contracted** | weakest run that trains | majority floor |
| --- | --- | --- | --- | --- |
| weather | 0.7271 | **0.6680** | 0.6778 | 0.6041 |
| scene | 0.6560 | **0.6381** | 0.6333 | 0.6034 |
| timeofday | 0.9271 | **0.9137** | 0.9167 | 0.4833 |

On balanced accuracy — the pre-registered *primary* endpoint for BDD, since the attributes
are severely skewed — it is statistically indistinguishable from the weakest real run on all
three attributes (0.2931 vs 0.2937; 0.3053 vs 0.3038; 0.7350 vs 0.7416).

**Timeofday is the clearest case.** Majority guessing scores 0.483. The healthy encoder
scores 0.927. The contracted one — variance 1e-4, cosine 1.0000 — scores **0.914**.

> **Geometric degeneration does not imply loss of decodable information.** The two are
> dissociable, and here they are almost completely dissociated.

### 3.3 The second finding: retrieval and probing disagree

Linear decodability survives. Nearest-neighbour structure does not:

| semantic endpoint | healthy | contracted | weakest real | chance |
| --- | --- | --- | --- | --- |
| linear probe (timeofday) | 0.9271 | **0.9137** | 0.9167 | 0.483 |
| retrieval P@10 | 0.6750 | **0.5664** | 0.6329 | 0.429 |

The probe says the contracted encoder is as good as a real run. Retrieval says it is clearly
worse — and unlike the probe comparison, that gap clears the across-seed noise.

They measure different things. A linear classifier can find a direction that separates
classes even in a squashed cloud. Retrieval asks *which points are nearest*, and squashing
reorders neighbourhoods.

**For ADAS this is the important half**, because scenario mining *is* retrieval. The endpoint
that matters is the one that degrades.

### 3.4 The third finding: two "effective rank" measures invert

| | healthy | contracted | verdict |
| --- | --- | --- | --- |
| RankMe | 78.36 | **1.06** | correct |
| participation ratio | 10.99 | **25.45** | **backwards** |

The participation ratio rates the contracted encoder as using more than twice the dimensions
of the healthy one. Paired 95% interval against `none_stopgrad`: **[−32.70, −15.36]** —
large, reproducible, and in the wrong direction.

The cause is one step of arithmetic: PR subtracts the mean before measuring. When every point
sits at the same non-zero location, subtracting the mean deletes exactly the thing that went
wrong, leaving random wobble that points in every direction and so reads as
high-dimensional. RankMe does not subtract the mean, so it still sees the pile-up. Both are
routinely called "effective rank". They are not interchangeable — that algebra is known; the
reproducible inversion in real training is what this project adds.

---

## 4. What we withdrew, and why

**The original claim was:**

> ~~A collapsed encoder scores 0.413 on the standardized linear probe against a 0.100 chance
> floor, while the unstandardized probe correctly reports it dead. The field-standard
> evaluation is blind to collapse.~~

**It was wrong, and the error was ours.** The unstandardized probe never worked on contracted
features.

The probe fits a classifier by taking repeated downhill steps, stopping when the slope gets
shallow. The slope's steepness scales with the size of the input numbers. Our contracted
encoder emits numbers around 0.001, where the slope is *already* below the stopping threshold
before the first step. So the probe stopped immediately, predicted the most common class, and
returned chance — while raising no warning at all.

The tell was in our own numbers: the "dead" readings sat exactly on the majority floor
(CIFAR-10 0.1066 vs 0.1088; BDD100K timeofday 0.4832 vs 0.4833).

Re-measuring Phase B from its saved encoders under a corrected protocol moved the contracted
encoder's timeofday score from **0.4832 to 0.9137**, while every other condition moved by
0.00–0.01 and the standardized probe moved by 0.0035. The defect was specific to exactly the
case the claim depended on.

So there was no information loss for the standard evaluation to be blind *to*. The
standardized probe had been reporting correctly all along.

**Supporting evidence for the mechanism.** The corrected protocol chooses regularisation
strength on a validation split. It picks **C = 0.1 for the healthy encoder and C = 10⁵ for
the contracted one** — a factor of a million. Regularisation strength is defined relative to
feature magnitude, so a representation shrunk by orders of magnitude needs a proportionally
weaker penalty to be fitted at all. A single fixed C, as the old protocol used, cannot serve
both. That is the confound made visible.

**Also withdrawn:** the explanation that standardization and L2-normalization "both rescale
floating-point noise back to readable magnitude". They are different operations and that was
never derived. Whether the surviving retrieval signal is genuine angular structure or
numerical fragility is open (§6).

**Terminology.** The control is not "dead", "constant" or "fully collapsed" — it has
consistent non-zero variance across all five seeds. It is **severely scale-contracted and
angularly concentrated, and retains decodable structure.** Six distinct things are kept apart
in this project and must not be used as synonyms:

| | meaning |
| --- | --- |
| exact constant collapse | E(x) = c for all x; no input-dependent structure at all |
| scale contraction | structure survives, shrunk by orders of magnitude |
| angular concentration | vectors aim the same way |
| dimensional / rank collapse | spread survives in only a few directions |
| loss of downstream information | the representation no longer supports the task |
| numerical fragility | structure survives in float64 but maybe not in float16 or int8 |

---

## 5. Strength of evidence

| tier | what | status |
| --- | --- | --- |
| **Strongest** | Geometry on Phase B, 5 paired seeds, paired Student-t. Pure arithmetic, no optimiser. | confirmatory |
| **Strong** | Corrected semantics on Phase B, 5 seeds, from saved encoders on the fixed evaluation split. | confirmatory endpoints, post-hoc protocol |
| **Supporting** | Controlled synthetic degradations: which metric is blind to which transformation. | exploratory |
| **Suspect** | All Phase-A (CIFAR-10) probe numbers. Never reprobed — no encoders were saved. | do not quote |
| **Exploratory** | Panel thresholds, comparator choice, verdict scheme, severity grid — all chosen after seeing results. | labelled in the report |

The statistical protocol in force is **paired Student-t 95% intervals** over five
pre-declared contrasts, with no verdict language. An earlier "2× seed SD" decision rule was
withdrawn before any run and must not reappear; see the report's Provenance section.

---

## 6. What is open

- **Is the surviving retrieval signal real angular structure, or numerical fragility?**
  Retrieval currently runs in float64 exact search. If the ordering survives float16, int8
  and approximate nearest-neighbour search, it is structure worth characterising. If it
  dissolves, that is a fragility result. This is the next experiment.
- **Phase A probe endpoints are uncorrected.** CIFAR-10 needs re-running with encoder saving
  before any of its probe numbers can be quoted.
- **Does the dissociation hold on a genuinely strong encoder?** Our best model reaches 0.927
  on timeofday but only 0.394 balanced on weather. Everything here sits in a narrow band of
  mediocre representations.
- **Why does linear decodability survive squashing at all?** We observe it; we have not
  explained it.
- **Degeneration here was induced**, by removing stop-gradient. Whether these diagnostics
  behave the same way on collapse that arises spontaneously is untested.

---

## 7. Where to read more

| | |
| --- | --- |
| Build every concept from zero | [`primer.md`](primer.md) |
| Formulas and worked examples | [`metrics.md`](metrics.md) |
| What was fixed before runs vs chosen after | `report/sections/provenance.tex` |
| Engineering decisions and bugs found | [`development-log.md`](development-log.md) |
| Open questions in more depth | [`open-questions.md`](open-questions.md) |

---

## 8. A note on how this went

The project's most defensible output is not a finding, it is a correction. It caught, in its
own work: a SIGReg implementation bug that inverted the regularizer's behaviour, a masking
bias, a global-RNG confound between conditions, a condition that was silently inert, a
preregistered decision rule that was withdrawn and then wrongly described as preregistered,
and finally a probe defect that had produced its own headline result.

Each was found by testing the instrument rather than trusting the output. That record is
worth more than the claim it cost.
