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
premise. Two families of answer.

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

The control is the instrument. It is *supposed* to fail. It gives a known-dead encoder to
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

**The claim rule.** A difference is claimed only if it exceeds **twice the across-seed
standard deviation**. If run-to-run noise is ±5 and conditions differ by 3, that is noise.
This rule is deliberately strict; the report applies it as written, and it kills three
results that would have looked significant otherwise.

---

## 7. The finding

Everything above was setup. Here is the result.

Look at `none_nostopgrad` — the control with no collapse prevention. It collapsed exactly
as designed: total variance `0.0002` (400,000× below the healthy baseline), mean pairwise
cosine `1.0000` (every embedding pointing the same direction). **It is dead. A single point.**

Now evaluate it:

| metric | reading | chance floor | verdict |
| --- | --- | --- | --- |
| total variance | 0.0002 | — | ✅ correct |
| mean pairwise cosine | 1.0000 | — | ✅ correct |
| **unscaled** linear probe | 0.107 | 0.100 | ✅ correct — dead |
| RankMe | 1.12 | — | ✅ correct |
| **standardized linear probe** | **0.413** | 0.100 | ❌ **blind** |
| **cosine retrieval P@10** | **0.184** | 0.100 | ❌ **blind** |
| **participation ratio** | **34.98** | (healthy: 26.90) | ❌ **blind, inverted** |

**A dead encoder scores 0.413 on the standard SSL evaluation** — which ranks it
**second of all seven conditions**, behind only the healthy EMA baseline and above every
other arm in the experiment. It retrieves at 1.8× chance, statistically indistinguishable
from a partially-working encoder.

### Why

Go back to the two things I asked you to remember.

- Standardization **divides each feature by its standard deviation**.
- Cosine similarity **ignores vector length**.

The collapse was a loss of *scale*. Both protocols remove *scale* before measuring. They
surgically delete the exact axis along which the failure occurred.

What is left? A collapsed encoder's output is not mathematically identical across inputs —
it is the same to about four decimal places, with tiny residual differences from
floating-point arithmetic. That residue is still a **deterministic function of the input**.
Standardization rescales it back to unit size, and a linear model reads it perfectly
happily.

You are classifying numerical noise. And it works, because the noise is systematic.

**⚠️ Provenance, stated plainly:** the *mechanism* is not a discovery. Standardization
removes scale because that is what standardization is for; it was predicted in advance and
documented in the code before these runs. What this project contributes is the
**magnitude** — that it reaches 0.413 on 10-class CIFAR-10 — and the demonstration that it
survives a pre-registered protocol.

### Why it matters

Both blind protocols are **the defaults**. Standardized linear probing is *the* SSL
evaluation. Cosine similarity is the default in essentially every vector database.

> A pipeline evaluated only the standard way cannot distinguish a working encoder from one
> that has collapsed by five orders of magnitude.

The fix is cheap — also log total variance and the unscaled probe — but it must be done
deliberately. And for scenario mining the sting is that **retrieval**, the operation the
application actually performs, is among the endpoints least able to detect this.

### The second finding

Geometry and semantics fail to resolve in **opposite places**. Where the geometric
difference is huge and precisely measured (169 units of variance), the probe barely moves.
Where the semantic difference is unmistakable (+0.210 probe accuracy), neither geometric
endpoint clears the claim rule, because that condition is wildly unstable across seeds
(variance 7.5 → 50.9).

Where you can measure geometry precisely, it tells you nothing about meaning. Where meaning
differs enormously, geometry is too unstable to say so.

**Necessary caveat, which the report also carries:** absolute detection still works. Nobody
would see cosine 1.0000 and call it healthy. What fails is using these diagnostics to
**rank or compare** configurations — which is the job they are usually given.

### The third finding

RankMe and participation ratio are **not interchangeable**, despite both being called
effective rank:

- **Participation ratio** uses the *centered* covariance — it subtracts the mean first. So
  it cannot see a collapse to a non-zero constant: subtract the mean and you are left with
  noise that looks full-dimensional. Hence 34.98 on the dead encoder.
- **RankMe** uses raw singular values, so it catches that. But SIGReg keeps its residual
  variance *round* while shrinking it, which flattens the raw spectrum, so RankMe reads
  39.60 on `sigreg_nostopgrad` — whose variance is 127× below baseline.

Each is blind to the mode the other catches.

---

## 8. What to read, in order

**If you have 15 minutes** — read §7 above again, then open the interactive demo:

```bash
uv run python viz/build_report.py --tag phaseA_s0 --scorecard-tag phaseA --seeds 0,1,2,3,4 \
  --title "jepa-lens: can you trust a self-supervised embedding?"
open viz/dist/report.html
```

It shows you the two blind charts first and asks you to spot the dead encoder. You can't.
Then it reveals the two that work. Seeing it beats reading it.

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

If you can say this without notes, you can defend the project in an interview:

> Self-supervised encoders can collapse to a constant, and the two standard ways of
> checking — a standardized linear probe and cosine retrieval — both mathematically remove
> scale, so both rate a totally collapsed encoder as mediocre-but-fine rather than dead;
> we measured that at 0.413 probe accuracy against a 0.100 chance floor, under a
> pre-registered protocol with five paired seeds.

Then the follow-up you should expect, and its answer:

> **"Isn't that obvious? Standardization removes scale by definition."**
> Yes — the mechanism is definitional and we say so. What was not known was the magnitude,
> that it beats normally-training conditions, and that retrieval is blinded too. The
> contribution is the measurement and the protocol, not the insight.
