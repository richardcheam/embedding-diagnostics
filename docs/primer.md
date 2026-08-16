# Primer: everything you need to read the report

**Read this before `report/main.pdf`.** It builds the whole chain from the beginning — what
an embedding is, why models collapse, how people measure whether a representation is any
good, and why that measurement is the thing this project attacks. Every idea is introduced
before it is used. Nothing assumes you remember an earlier section perfectly.

The payoff at the end is that the report's central result becomes a single sentence you can
say out loud. If you get there and it does not feel obvious, the fault is in this document
and you should tell me which section lost you.

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

## 7. What we found — and what we had to take back

This section changed on 2026-08-16 after an audit. One of the two headline results turned
out to rest on a broken measurement. Rather than delete it, the story is told as it
happened, because how a result gets withdrawn is worth more than the result was.

### 7.1 What we set out to show

The control condition — the one with no collapse prevention at all — degenerated exactly as
designed. Its measurements:

| measurement | reading | what it means |
| --- | --- | --- |
| total variance | 0.0002 | the cloud of points shrank to almost nothing |
| mean pairwise cosine | 1.0000 | every point ends up pointing the same direction |
| RankMe | 1.12 | effectively one dimension left, out of 192 |

Those three agree: this encoder maps essentially every image to the same place. All three
are **Kind 1** — pure arithmetic — so they are as trustworthy as a ruler.

Then we ran the two evaluations people normally use, and they disagreed with each other:

| evaluation | reading | chance | our original reading of it |
| --- | --- | --- | --- |
| standardized linear probe | 0.413 | 0.100 | "blind — it can't see the collapse" |
| unstandardized linear probe | 0.107 | 0.100 | "correct — it reports the encoder as dead" |

That looked like a clean, useful result: *the standard way of evaluating these models is
blind to this failure.* Since standardized probing is the field default, that mattered.

### 7.2 Why we took half of it back

Both probes are **Kind 2** — they fit a model — and the fitting was broken.

The probe trains a small classifier by taking repeated downhill steps. It stops when the
slope gets shallow enough, on the reasoning that a shallow slope means you've arrived.

But the slope's steepness scales with how big the input numbers are. Our collapsed encoder
emits numbers around 0.001. On numbers that small, the slope is *already* shallower than the
stopping threshold before a single step is taken. So the optimizer looked at the starting
point, decided it had arrived, and stopped — having learned nothing.

A classifier that has learned nothing predicts the most common class every time. On
10-class CIFAR-10 that scores about 0.10. **Which looks exactly like "chance."**

Here is the test that settles it. Take data where the label information is *definitely
present* — we build it in on purpose — but at that same tiny scale:

```
unstandardized probe : 0.096   ← reports "dead"
standardized probe   : 1.000   ← reports "perfect"
```

Same data. Full information. No collapse whatsoever. Our exact headline pattern, reproduced
with nothing wrong. Under a corrected setup both read 1.000.

And it reached the real runs. On both datasets the control's unstandardized score sits
exactly on top of the always-guess-the-most-common-class score:

| dataset | unstandardized probe | always-guess-majority | difference |
| --- | --- | --- | --- |
| CIFAR-10 | 0.1066 | 0.1088 | −0.0022 |
| BDD100K | 0.5635 | 0.5636 | −0.0001 |

Agreeing to four decimal places is not a coincidence. That is the signature of a classifier
that only ever guessed the majority class.

**So the claim "the unstandardized probe correctly detects collapse" is withdrawn.** Not
disproven — *unmeasured*. The instrument was broken, so we do not currently know whether a
collapsed representation still holds usable information.

⚠️ Note carefully what this does **not** say. It does not say the encoder is fine. It does
not say the standardized probe was right after all. It says one of our two instruments was
faulty, so that particular comparison tells us nothing until it is redone.

### 7.3 What still stands, and why

Everything in §7.1 — the geometric measurements — is **Kind 1**. No optimizer, nothing to
fail. Also unaffected: retrieval (find the 10 nearest neighbours, count how many share the
label — sorting and counting, no fitting).

Two further results also stand, both Kind 1.

**Two "effective rank" measures disagree, and one is backwards.** On the collapsed control:

| measure | reading | healthy baseline | verdict |
| --- | --- | --- | --- |
| RankMe | 1.12 | 85.72 | correct |
| participation ratio | **34.98** | 26.90 | **backwards** |

The participation ratio calls the degenerate encoder healthier than the healthy encoder. The cause is
one step of arithmetic: **it subtracts the average position before measuring.**

Picture every image landing on the same spot, far from the origin — say at (5, 5, 5, …).
Subtract the average, and that shared spot vanishes. What's left is a whisker of random
measurement wobble. Random wobble points every which way, so it looks like the cloud is
using lots of dimensions, so it reads as healthy. It is measuring noise and calling it
structure.

RankMe doesn't subtract the average, so it still sees that everything is piled in one place.
Both are called "effective rank" in the literature. They are not interchangeable.

**No single label-free measurement is enough.** We break embeddings in five known ways and
check which measurements notice. Two blind spots turn out to be complementary:

- shrink everything 1000× → cosine, RankMe and the participation ratio notice *nothing*
  (they ignore size, deliberately)
- shove the whole cloud sideways → total variance and the participation ratio notice
  *nothing* (they subtract position, deliberately)

So you need at least one from each family. We recommend **total variance + RankMe**. Of the
four pairs that work, the participation ratio is in none of them.

**Watching for change doesn't work either.** The obvious monitoring rule — "warn me if a
number drifts a long way from where it started" — fired on 5 of 6 healthy CIFAR runs and 4
of 6 healthy BDD runs, including the only condition that actually learned anything. Healthy
training reshapes the geometry just as much as collapse does. Fixed thresholds are needed
instead.

### 7.4 Where that leaves the project

**Known:** how to describe the *shape* of a degenerate representation precisely, which
shape-measurements lie to you, and why they lie.

**Unknown:** whether a degenerate representation still holds usable information. That needs
a working probe.

**Being done:** Phase B saved its trained encoders, so the probes can simply be re-run on
them — minutes, no retraining. Phase A did not save encoders, so it needs a re-run. See §10.

## 8. What to read, in order

**If you have 15 minutes** — read §7 above again, then open the interactive demo:

```bash
uv run python viz/build_report.py --tag phaseA_s0 --scorecard-tag phaseA --seeds 0,1,2,3,4 \
  --title "jepa-lens: can you trust a self-supervised embedding?"
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

Two versions, and you want the second.

**What we can say today:**

> Self-supervised encoders can degenerate until every input maps to nearly the same place,
> and the label-free measurements people use to detect that disagree with each other — the
> participation ratio actually rates a degenerate encoder as *healthier* than a working one,
> because it subtracts the average position before measuring and so deletes the very thing
> that went wrong. No single measurement covers every failure mode; two do.

**What we cannot yet say**, and previously claimed:

> ~~The standard evaluation protocol is blind to collapse while the unstandardized probe
> detects it.~~ Withdrawn. The unstandardized probe's optimizer stopped before fitting on
> such small-scale features, so it reported chance regardless of what was there. Being
> recomputed.

The follow-up you should expect, and its answer:

> **"So your headline result fell over?"**
> Half of it did. The geometric half stands and is what the project is now built on. The
> semantic half was measured with a broken instrument, and I found that out by auditing my
> own protocol rather than by publishing it and being corrected. The withdrawal is
> documented in the report's Provenance section with the commit history, including the fact
> that an earlier draft described a withdrawn decision rule as pre-registered.

That last answer is worth more in an interview than the original claim was. Anyone can
report a finding; being able to show how you caught your own error is the harder signal.

---

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
