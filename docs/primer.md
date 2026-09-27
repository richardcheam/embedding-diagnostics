# Primer: everything you need to read the report

**Read this before `report/main.pdf`.** It builds the whole chain from the beginning — what
an embedding is, why models collapse, how people measure whether a representation is any
good, and why that measurement is the thing this project attacks. Every idea is introduced
before it is used. Nothing assumes you remember an earlier section perfectly.

The payoff at the end is that the report's central result becomes a single sentence you can
say out loud. If you get there and it does not feel obvious, the fault is in this document
and you should tell me which section lost you.

**Every metric named here is defined with formula, worked example and blind spot in
[`metrics.md`](metrics.md).** If a term like RankMe, participation ratio or P@10 goes past
without landing, that is the place to look it up rather than pushing on.

Companion material lives in the original learning repo at `~/Desktop/git/jepa-learning`
(`docs/concepts/`, `docs/model_families/`, and nine notebooks). Pointers are given inline.
That repo teaches the *field*; this document teaches *this project*.

---

## 1. The problem: learning without labels

Supervised learning needs labels. Someone has to write "this image is a cat." For driving
data that is ruinous: a car collects millions of frames a day and nobody is going to label
them all.

**Self-supervised learning (SSL)** removes the labeller. You invent a task that generates
its own answer key from the data's own structure. Hide part of an image and predict the
hidden part; the answer key is the part you hid. No human involved.

You don't want the prediction itself — nobody needs a model that fills in image patches.
You want the **representation** learned along the way: the internal vector the model
computes for each input. If the model can predict the hidden region, it has presumably
learned something about what is in the image.

That vector is the **embedding**. One image goes in, a list of numbers comes out — in this
project, 192 of them. The hope is that similar images produce nearby vectors.

> Deeper: `jepa-learning/docs/concepts/embeddings.md`, notebook `01_embedding_geometry`.

### Why this matters for driving

The application behind this project is **scenario mining** for ADAS validation. You have
millions of driving clips. You want to ask: *find me more situations like this one* — night
rain with a cyclist, say — or *does my test set actually cover fog?*

Embed every clip, and both questions become geometry. "Similar situations" means "nearby
vectors." "Coverage" means "does the cloud of vectors spread over the space, or is it
bunched up?"

That is the entire premise. And it depends completely on the embeddings being trustworthy,
which is where this project comes in.

---

## 2. JEPA: predicting in latent space

There are many SSL recipes. The one here is **JEPA** — Joint-Embedding Predictive
Architecture, from Meta (I-JEPA, arXiv 2301.08243).

The distinguishing choice is *where* the prediction happens.

- **Reconstruction methods** (autoencoders, MAE) predict the hidden **pixels**. To score
  well they must model raindrop positions and exact leaf textures — detail that is
  unpredictable and mostly irrelevant.
- **JEPA** predicts the hidden region's **embedding**. Predict the *idea* of what is there,
  not its pixels.

The intuition: asked what is behind a parked van, you can confidently say "probably road,
maybe a pedestrian" but not the exact arrangement of gravel. JEPA is scored on the first
answer, so it never spends capacity on the second.

### The machinery

Four parts, and you need all four to read the report:

1. **Context encoder** — sees the *visible* patches, produces an embedding. This is the
   network you actually keep; everything else is scaffolding.
2. **Target encoder** — sees the *full* image, produces the embedding to be predicted.
3. **Predictor** — maps the context embedding toward the target embedding.
4. **Loss** — how far the prediction landed from the target. Squared distance here.

Training nudges the weights to shrink that distance.

> Deeper: `jepa-learning/docs/model_families/jepa.md`, notebook `07_tiny_jepa`.

---

## 3. Collapse: the cheat that ruins everything

Here is the flaw at the centre of this entire project.

The model is scored on *making the prediction match the target*. So ask: what is the
easiest possible way to make two things match?

**Make them constant.** If the encoder ignores its input and outputs the same vector —
say all zeros — for every image, then the prediction is always zero, the target is always
zero, the distance is always zero. **Perfect loss. Zero learning.**

This is **representation collapse**, and it is not a rare bug. It is the optimizer finding
the cheapest path, exactly as designed. Every method in this family needs an explicit
mechanism to block it.

A collapsed encoder is worthless for scenario mining. Every clip maps to the same point,
so every clip is "similar" to every other, and retrieval returns noise.

> Deeper: `jepa-learning/docs/concepts/collapse.md`, notebook `05_representation_collapse`.

### Collapse is not one failure

This distinction is the reason the project exists. "Collapsed" is used loosely to mean
several different geometric failures:

| mode | what happens | picture |
| --- | --- | --- |
| **complete collapse** | every input → the same vector | cloud shrinks to a dot |
| **scale collapse** | structure kept, but shrunk toward zero | same shape, 10,000× smaller |
| **angular concentration** | all vectors point the same direction | cloud becomes a narrow cone |
| **dimensional collapse** | spread in a few directions, flat in the rest | 3D cloud flattens to a pancake |

They are different failures, they need different detectors, and — the point of the whole
project — **some of them do not destroy the representation's usefulness at all.** Hold that
thought until §7.

---

## 4. The anti-collapse toolkit

The three mechanisms the report treats as its experimental conditions.

### Stop-gradient

Training works by computing the loss, then adjusting every weight that influenced it. The
model wants both branches constant. **Stop-gradient** cuts the target branch out of that
adjustment: the target is computed, but no blame flows back through it.

The target becomes a *fixed reference* for the step. The context encoder must move toward
the target; it cannot drag the target toward itself. The mutual race to zero is blocked.

In code it is one call — `.detach()` in PyTorch — and it is one of the great "why does this
even work" results of the field. It is empirically essential and not fully explained.

### EMA target encoder

Rather than sharing weights, keep a *second* copy of the encoder for targets, updated as a
slow-moving average of the trainable one:

```
target_weights ← 0.996 × target_weights + 0.004 × context_weights
```

The target drifts slowly behind the context encoder, so it is a stable thing to aim at
while still improving. Note it is **never trained** — it has no gradients, only this
averaging update. From BYOL and SimSiam.

> Deeper: `jepa-learning/docs/deep_dive_stop_gradient_ema.md`,
> `jepa-learning/docs/concepts/target_encoders.md`.

### SIGReg

The newest, from **LeJEPA** (arXiv 2511.08544, 2025). Rather than blocking collapse
structurally, add a term to the loss that *penalises* collapsed geometry directly: push the
distribution of embeddings toward an isotropic Gaussian — a round, evenly-spread cloud.

A collapsed cloud is a dot, which is maximally un-round, so it is penalised hard. The
appeal is that LeJEPA reports this makes the structural tricks unnecessary — no
stop-gradient, no EMA, just a well-posed objective.

**Testing that claim in a JEPA setting is where this project started.** The mechanics
(Epps-Pulley normality tests on random 1D projections) are in the report's Method section
and you can take them on faith for now.

⚠️ **Important scoping.** LeJEPA pairs SIGReg with *multi-view invariance* — eight augmented
crops, no predictor, no teacher network. This project pairs it with *masked latent
prediction*. Different systems sharing one component. Every SIGReg result here describes
that combination, not LeJEPA.

---

## 5. How do you know if a representation is good?

Now the crux. You have a trained encoder. Is it any good? You have no labels — that was the
premise.

### First, a distinction that matters more than it looks

Every check in this project is one of two kinds, and telling them apart explains almost
everything that follows.

**Kind 1 — measuring.** You take the 192 numbers the encoder produced for each image and do
arithmetic on them. How spread out are they? What's the average angle between them? How many
directions is the cloud using? Nothing is trained. Nothing is fitted. It's a calculator, and
like a ruler it either works or it's obviously broken.

**Kind 2 — fitting.** You *train a small model* on top of the embeddings and see how well it
does. This requires an optimizer: an algorithm that searches for good settings by taking
repeated steps downhill. Optimizers can fail. Worse, they can fail *quietly* — stopping early
and reporting a number that looks like a real answer.

Hold onto this. Later in this document, one Kind-2 check turns out to have been broken for
weeks while every Kind-1 check was fine. That is not a coincidence: it is a direct
consequence of one of them having moving parts and the other not.

Two families of answer follow.

### Family A: expensive, trustworthy — the linear probe

**Freeze** the encoder. Embed a set of images that *do* have labels. Fit a plain logistic
regression from embedding → label. Report accuracy on held-out data.

The linear part is the point. You are not asking "can something classify these images" —
a big enough network could classify raw pixels. You are asking whether the *encoder already
did the work*, leaving classes linearly separable.

This is **the** standard SSL evaluation. It needs labels, so it cannot be used to monitor
unlabelled production data — but it is the trustworthy reference.

**One implementation detail carries the entire report.** Before fitting, features are
almost always **standardized**: each of the 192 dimensions is divided by its standard
deviation, so all dimensions arrive at comparable size. This is ordinary practice — it
stops one large-valued dimension dominating the regularizer, and it keeps accuracy
comparable across checkpoints as embedding scale drifts.

**Remember that standardization divides out scale. It will matter enormously in §7.**

### Family B: also expensive — retrieval

Take a query embedding, find its 10 nearest neighbours, ask how many share its label.
That is **precision@10 (P@10)**.

For scenario mining this is the *more* relevant measure, because retrieval is literally the
operation the application performs. "Find clips like this one" is a nearest-neighbour query.

Nearness is measured by **cosine similarity** — the angle between two vectors, ignoring
their lengths. This is the default in essentially every vector database.

**Cosine ignores length. Also remember that for §7.**

### Family C: cheap, label-free — collapse diagnostics

These need no labels, so they can run continuously on production data. Each measures a
geometric property of the embedding cloud:

| diagnostic | measures | healthy | collapsed |
| --- | --- | --- | --- |
| **total variance** | overall spread (trace of covariance) | large | → 0 |
| **mean pairwise cosine** | average angle between embeddings | ~0 | → 1 |
| **participation ratio** | how many dimensions carry variance | high | low |
| **RankMe** | same idea, computed from raw singular values | high | low |

The last two are both called "effective rank" and both try to answer "how many dimensions
is this cloud really using." **They are computed differently and they do not agree** — the
report makes a point of this, and §7 shows why.

> Deeper: `jepa-learning/docs/concepts/representation_evaluation.md` (currently a stub —
> this section supersedes it), notebook `02_similarity_and_retrieval`.

### The dream

Probes need labels. Diagnostics don't. So: **can the cheap label-free diagnostics stand in
for the expensive labelled ones?** If yes, you can monitor an embedding pipeline forever
with no labelling budget.

That question, sharpened, is this project.

---

## 6. The experiment

Seven conditions, differing only in collapse-prevention mechanism, everything else held
fixed:

| condition | what it has | role |
| --- | --- | --- |
| `ema_stopgrad` | EMA target + stop-gradient | classic JEPA baseline |
| `none_stopgrad` | stop-gradient only | isolates stop-gradient |
| `sigreg_stopgrad` | SIGReg + stop-gradient | both |
| `sigreg_nostopgrad` | SIGReg only | **LeJEPA's claim under test** |
| `proj_sigreg_*` | as above + a projector head | tests where SIGReg should act |
| `none_nostopgrad` | **nothing** | **control: must collapse** |

The control is the instrument. It is *supposed* to fail. It gives a known-degenerate encoder to
test the diagnostics against — ground truth you otherwise never have.

### Three pieces of method vocabulary

**Seeds.** Random initialisation makes every run differ. Run each condition 5 times with
seeds 0–4 and report the spread. "Paired" means seed 3 of condition A and seed 3 of
condition B got identical initialisation, identical data order, identical masks — so the
only difference is the condition.

**Pre-registration.** Before running anything, write down which numbers you will look at
and what counts as a real difference. This blocks the temptation to hunt through results
for whatever looks best. The report's pre-registration is in the Experiments section, left
**unedited** even where reality deviated (3 seeds registered, 5 run — footnoted, not
rewritten).

**The analysis.** Paired Student-t 95% confidence intervals over five pre-declared
contrasts. For each contrast you take the per-seed differences, average them, and put an
interval around that average wide enough to cover run-to-run noise. The interval is an
*estimate*, not a test: no result gets labelled significant, real, learned, or collapsed.

An earlier draft of this project used a decision rule instead — "claim a difference only if
it exceeds twice the across-seed standard deviation" — and that rule was **withdrawn before
any data existed**, because it was not a paired analysis and worked as an arbitrary
threshold. If you see it quoted anywhere as pre-registered, that text is wrong and is being
corrected; see the report's Provenance section.

---

## 7. What we found — including the part we got wrong

This section describes a claim the project made, then refuted with its own data. That
sequence is the point: the withdrawal is better evidence of the method working than the
original claim would have been.

### 7.0 How to read every table in this section

The tables below compare the same three encoders and quote the same two baselines. Learn
these five things once and every table becomes readable.

**The three encoders.** We trained seven conditions; three of them do all the explanatory
work, so those are the ones quoted:

| what I call it | actual condition name | what it is |
| --- | --- | --- |
| **healthy** | `ema_stopgrad` | the classic recipe (EMA target + stop-gradient). Our best encoder. The "this is what good looks like" reference. |
| **contracted** | `none_nostopgrad` | the control with *no* collapse prevention. Built to fail, on purpose, so the diagnostics have something to be tested against. |
| **weakest real** | `none_stopgrad` | has stop-gradient but nothing else. It trains, but badly. **The most important comparison**, because "is the broken one worse than the *best* one" is easy; "is it worse than a *mediocre but genuine* one" is the real test. |

**The two baselines.** A raw accuracy is meaningless on its own. 0.60 might be excellent or
might be worthless — you cannot tell without knowing what doing *nothing* scores. So every
number is quoted next to its floor.

- **majority floor** — what you get by ignoring the image entirely and always guessing the
  most common answer. On BDD's `weather`, 60% of images are "clear", so guessing "clear"
  every time scores **0.604**. A probe scoring 0.61 has learned essentially nothing.
- **chance (retrieval)** — what you get by returning 10 *random* images instead of the 10
  most similar. This is **not** 1/number-of-classes: it is the probability two random images
  share a label, which on BDD is about **0.43**. (Worked out in
  [`metrics.md`](metrics.md) §5.)

The floors differ per row because the class balance differs per attribute. `timeofday` is
roughly half day / half night, so its floor is 0.48. `weather` is 60% clear, so its floor is
0.60. **Always read a number as "how far above its own floor", never as an absolute.**

**A worked reading.** Take one row from §7.4 and say it out loud:

> | attribute | healthy | contracted | weakest real | majority floor |
> | --- | --- | --- | --- | --- |
> | timeofday | 0.9271 | **0.9137** | 0.9167 | 0.4833 |

*"Guessing scores 0.483. Our best encoder scores 0.927 — so it has learned a lot. A badly
trained but genuine encoder scores 0.917. And the encoder that is supposed to be destroyed
scores 0.914 — statistically the same as the badly-trained one, and nowhere near the 0.483
you would get from learning nothing. So it has not lost the information."*

That sentence is the whole finding. Everything below is the same reading applied to more
rows.

---

### 7.1 The set-up

The control condition has no collapse prevention at all. It degenerated exactly as intended,
and all three geometric measures agree:

| geometric measure | **contracted** (the control) | healthy | what the contracted reading means |
| --- | --- | --- | --- |
| total variance | **0.0001** | 107.71 | the cloud shrank to almost nothing — a millionth of normal |
| mean pairwise cosine | **1.0000** | 0.256 | every point aims in the same direction (1.0 is the maximum) |
| RankMe | **1.06** | 78.36 | one usable dimension left out of 192 (1.0 is the floor) |

These are **Kind 1** (§5) — arithmetic, no optimizer — so they are reliable. By any
geometric account this representation is wrecked.

### 7.2 What we originally claimed

We ran the two evaluations people normally use, and they disagreed:

| evaluation, run on the **contracted** encoder | its score | its floor | what we concluded at the time |
| --- | --- | --- | --- |
| standardized probe | 0.413 | 0.100 | "blind — scores well above the floor on a broken encoder" |
| unstandardized probe | 0.107 | 0.100 | "correct — sits on the floor, so it sees the failure" |

That looked like a clean result: *the field-standard evaluation is blind to this failure.*
Since standardized probing is the default everywhere, it would have mattered.

### 7.3 Why it was wrong

Both probes are **Kind 2** — they fit a model — and the fitting was broken.

The probe trains a classifier by repeatedly stepping downhill, stopping when the slope goes
shallow, on the reasoning that a shallow slope means you have arrived. But **the slope's
steepness scales with the size of the input numbers.** Our contracted encoder emits numbers
around 0.001. At that scale the slope is already shallower than the stopping threshold
before a single step. The optimizer looked at its starting point, concluded it had arrived,
and stopped — having learned nothing.

A classifier that learned nothing predicts the most common class every time. Which scores
exactly what "chance" scores. **So it looked like a correct detection and was in fact a
failure to measure.**

The tell was in our own numbers. Those "dead" readings sat exactly on the always-guess-the-
majority score:

| dataset | our "dead" reading | always-guess-majority | difference |
| --- | --- | --- | --- |
| CIFAR-10 | 0.1066 | 0.1088 | −0.0022 |
| BDD100K (timeofday) | 0.4832 | 0.4833 | −0.0001 |

Agreeing to four decimals is not coincidence.

### 7.4 What happened when we measured properly

Phase B saved its trained encoders, so we reloaded them and re-measured with a corrected
probe. The contracted encoder's score on *timeofday* moved from **0.4832 to 0.9137** — while
every other condition moved by 0.00 to 0.01, and the standardized probe moved by 0.0035.

The full corrected picture:

| BDD attribute | healthy `ema_stopgrad` | **contracted** `none_nostopgrad` | weakest real `none_stopgrad` | floor (always guess majority) |
| --- | --- | --- | --- | --- |
| weather | 0.7271 | **0.6680** | 0.6778 | 0.6041 |
| scene | 0.6560 | **0.6381** | 0.6333 | 0.6034 |
| timeofday | 0.9271 | **0.9137** | 0.9167 | 0.4833 |

Read each row against its floor. On `weather` the floor is high (0.604) so nobody is far
above it. On `timeofday` the floor is low (0.483) and everything is far above it — which is
why that row is the clearest evidence.

Read the timeofday row slowly. Guessing scores 0.483. A healthy encoder scores 0.927. The
encoder whose variance is one ten-thousandth of normal, whose every output points the same
direction, scores **0.914**.

> **The information was never lost.** The representation is squashed almost flat, and you
> can still read the labels off it.

So the original claim collapses. There was nothing for the standardized probe to be blind
to — it had been right all along. Our unstandardized probe manufactured a fake detection out
of an optimizer failure.

### 7.5 The finding that replaces it

This is the better result, and it is what the project now says:

> **Geometric degeneration and loss of information are different things, and they come
> apart.** An encoder can be degenerate on every geometric measure and still carry nearly
> all its decodable content.

And a second, sharper observation — **the two semantic evaluations disagree with each
other:**

| semantic endpoint | healthy | **contracted** | weakest real | its floor |
| --- | --- | --- | --- | --- |
| linear probe (timeofday) | 0.9271 | **0.9137** | 0.9167 | 0.483 (majority) |
| retrieval P@10 | 0.6750 | **0.5664** | 0.6329 | 0.429 (random neighbours) |

Compare the two rows *within* the contracted column. On the probe it sits level with the
weakest real encoder (0.9137 vs 0.9167). On retrieval it drops well below it (0.5664 vs
0.6329) and is heading toward the 0.429 floor. Same encoder, two endpoints, opposite verdicts.

The probe says the contracted encoder is fine. Retrieval says it is clearly worse — and that
gap, unlike the probe's, is bigger than the run-to-run noise.

Why? They ask different questions. A linear classifier only needs *some direction* along
which the classes separate; squashing the cloud does not necessarily destroy that. Retrieval
asks *which points are closest to which*, and squashing scrambles that ordering.

**This is the half that matters for driving.** Scenario mining is retrieval — "find me more
clips like this one". The endpoint the application actually uses is the one that degrades,
and the endpoint most people report is the one that does not.

### 7.6 And the diagnostics still disagree with each other

Both of these claim to measure "how many dimensions is the encoder using", so for both,
**higher = healthier**:

| measure | healthy | **contracted** | is that the right direction? |
| --- | --- | --- | --- |
| RankMe | 78.36 | **1.06** | ✅ yes — far lower, as it should be |
| participation ratio | 10.99 | **25.45** | ❌ **no — it says the broken one is better** |

The participation ratio says the wrecked encoder is using more than twice the dimensions of
the healthy one. Cause: **it subtracts the average position before measuring.** Picture every
image landing on the same spot far from the origin. Subtract the average and that shared spot
vanishes, leaving a whisker of random wobble — and random wobble points every which way, so
it reads as "many dimensions in use". It is measuring noise and calling it structure.

RankMe skips the subtraction, so it still sees everything piled in one place.

**No single label-free measurement is enough**, either. Break embeddings in five known ways
and two blind spots turn out to be complementary: shrink everything and the scale-invariant
measures notice nothing; shift the whole cloud sideways and the centred measures notice
nothing. You need one from each family. We recommend **total variance + RankMe**.

### 7.7 Where that leaves things

**Known:** how to describe the shape of a degenerate representation precisely; which
shape-measures lie and why; that shape and information come apart; that probing and
retrieval come apart.

**Unknown:** whether the surviving retrieval signal is genuine angular structure or an
artifact of exact float64 arithmetic that would vanish in a real vector database (float16,
int8, approximate search). That is the next experiment.

**Not yet redone:** the CIFAR-10 probe numbers. That campaign saved no encoders, so it needs
a re-run. Do not quote its probe values.

## 8. What to read, in order

**If you have 15 minutes** — read §7 above again, then open the interactive demo:

```bash
uv run python viz/build_report.py --tag phaseA_s0 --scorecard-tag phaseA --seeds 0,1,2,3,4 \
  --title "embedding-diagnostics: can you trust a self-supervised embedding?"
open viz/dist/report.html
```

It shows you the standardized probe and cosine retrieval first and asks you to spot the
degenerate encoder. You can't. Then it reveals total variance and RankMe, which you can.
Seeing it beats reading it. (The demo's scorecard still uses the withdrawn probe reading in
one row — it is being rebuilt after the recomputation.)

**If you have an evening** — this primer, then `docs/STATUS.md` (plain-language status and
what is still open), then the report's Results section. Skip the report's Method section on
the first pass; come back to it once the result is solid in your head.

**If you want the foundations properly** — go to the other repo and follow its curriculum:

```
~/Desktop/git/jepa-learning/docs/learning_guide.md
```

Priority order for *this* project specifically:

1. `docs/concepts/embeddings.md` + notebook `01_embedding_geometry` — what a vector space is
2. notebook `02_similarity_and_retrieval` — cosine, nearest neighbours, P@10
3. `docs/concepts/collapse.md` + notebook `05_representation_collapse` — **the core concept**
4. `docs/deep_dive_stop_gradient_ema.md` — why stop-gradient works
5. `docs/model_families/jepa.md` + notebook `07_tiny_jepa` — the architecture end to end

The notebooks are the highest-value items. They expose intermediate values, so you can
watch an encoder collapse rather than read that it does.

**Primary sources**, if you want them: I-JEPA (arXiv 2301.08243), LeJEPA (arXiv 2511.08544),
RankMe (arXiv 2210.02885), BYOL (arXiv 2006.07733).

---

## 9. Say it in one sentence

If you can say this without notes, you can defend the project:

> A self-supervised encoder can degenerate until its embeddings have a millionth of their
> normal spread and every one points the same direction — and still be almost as linearly
> decodable as a working encoder. Geometric collapse and information loss are different
> things. What *does* degrade is retrieval, which is the operation scenario mining actually
> performs and the one people report least.

Two follow-ups you should expect:

> **"Didn't you originally claim the opposite?"**
> Yes. I claimed the standard evaluation was blind to collapse, because our unstandardized
> probe read chance on the degenerate encoder. That probe was broken: on features that small
> its optimizer stopped before fitting and returned the majority-class guess, which is
> numerically identical to chance. I found it by auditing my own protocol, re-measured from
> the saved encoders, and the reading moved from 0.483 to 0.914. The withdrawal is in the
> report's Provenance section with commit references.

> **"So what's left?"**
> Three things, all of which survive because they involve no fitted model: the dissociation
> between geometry and information; the dissociation between probing and retrieval; and a
> reproducible inversion where the participation ratio rates a degenerate encoder as
> healthier than a working one. Plus a methods contribution — the probe failure mode is a
> real trap, it is silent, and it will bite anyone evaluating a low-scale representation.

## 10. Why "reprobe", and why Phase A costs more

Two separate facts that interact:

**Why redo the measurement at all?** Because the probe was broken (§7.2). Any number taken
with a broken instrument has to be retaken.

**Why is Phase B cheap and Phase A expensive?** Because of what was saved.

Training produces a *trained encoder* — the actual learned weights. Saving them (`encoder.pt`)
means you can reload it later and measure it again, without repeating the training. Phase B
saved them. Phase A ran before that feature existed and did not.

| | measurement broken? | encoder saved? | cost to fix |
| --- | --- | --- | --- |
| Phase B (BDD) | yes | **yes** | minutes — reload and re-measure |
| Phase A (CIFAR) | yes | **no** | a re-run — the encoder must be retrained |

Think of it as weighing a hundred objects on a broken scale. You must re-weigh them all
either way. If you still have the objects, that's an afternoon. If you threw them out, you
have to make them again.

```bash
uv run python scripts/reprobe.py --tag phaseB --seeds 0,1,2,3,4 --data-root ../100k --compare
```

That reloads each saved encoder, re-embeds the same evaluation images, refits the probes
under the corrected protocol, and writes the results *beside* the originals — never over
them, because the difference between the old and new numbers is itself the evidence that the
correction mattered.
