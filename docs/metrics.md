# Every metric in this project, with the maths and a worked example

Companion to [`primer.md`](primer.md). The primer tells the story; this defines the
vocabulary. Each entry gives the formula, what it means in words, a small example you can
check by hand, and — importantly — **what it is blind to**, because every one of them has a
blind spot and the project is largely about those.

Notation throughout: **Z** is the embedding matrix, one row per image, one column per
dimension. In this project Z has 192 columns. Write $z_i$ for row $i$.

All numeric examples below are computed by the real code, not by hand-waving. Reproduce any
of them with `uv run python` and the snippets shown.

---

## The running example

Four images, two dimensions, so everything stays checkable:

```
Z = [[3, 0],
     [1, 0],
     [0, 2],
     [0, 4]]
```

---

## 1. Total variance

**Formula.** Centre each column (subtract its mean), then

$$\mathrm{totvar} = \operatorname{tr}(\mathrm{Cov}(Z)) = \sum_k \lambda_k$$

where $\lambda_k$ are the eigenvalues of the covariance matrix. Equivalently: the sum of
each dimension's variance.

**In words.** How spread out the cloud of points is, in total. Big = points are far apart.
Near zero = everything piled in one place.

**Worked example.** Column means are $(1, 1.5)$. Centred:

```
[[ 2.0, -1.5],
 [ 0.0, -1.5],
 [-1.0,  0.5],
 [-1.0,  2.5]]
```

Covariance $= \begin{pmatrix} 2.0 & -2.0 \\ -2.0 & 3.667\end{pmatrix}$, eigenvalues
$(0.667,\ 5.0)$, so **total variance = 5.667**.

**Blind to:** everything about *shape*. You can rotate the cloud, or shove it a mile from
the origin, and total variance does not move. It only sees size.

**Why this project cares.** It is the only one of the four geometric measures that detects
pure shrinkage. Our degenerate control reads $2\times10^{-4}$ against a healthy $\approx 108$.

---

## 2. Mean pairwise cosine similarity

**Formula.** For two vectors,

$$\cos(z_i, z_j) = \frac{z_i \cdot z_j}{\lVert z_i\rVert\,\lVert z_j\rVert}$$

and the metric is the average over all pairs $i \neq j$.

**In words.** The average *angle* between embeddings, ignoring their lengths. Cosine is 1
when two vectors point the same way, 0 when perpendicular, −1 when opposite.

**Worked example.** $z_1 = (3,0)$ and $z_3 = (0,2)$ are perpendicular:
$\cos = \frac{3\cdot 0 + 0\cdot 2}{3 \times 2} = 0$. Whereas $(3,0)$ and $(1,0)$ point
identically: $\cos = \frac{3}{3 \times 1} = 1$.

**Blind to:** length, entirely. Multiply every vector by $10^{-4}$ and cosine is unchanged.

**Why this project cares.** Values near 1 mean every image is being mapped in the same
direction — *angular concentration*. Our degenerate control reads exactly 1.0000; the
healthy baseline reads 0.256.

---

## 3. Participation ratio (sometimes called "effective rank")

**Formula.** From the eigenvalues $\lambda_k$ of the **centred** covariance,

$$\mathrm{PR} = \frac{\left(\sum_k \lambda_k\right)^2}{\sum_k \lambda_k^2}$$

**In words.** Roughly "how many dimensions is the cloud actually using". If variance is
spread evenly over $d$ dimensions, PR $= d$. If one dimension dominates, PR $\to 1$.

**Worked example.** Eigenvalues $(0.667, 5.0)$:

$$\mathrm{PR} = \frac{(0.667 + 5.0)^2}{0.667^2 + 5.0^2} = \frac{32.11}{25.44} = 1.26$$

Out of 2 possible dimensions, it is effectively using 1.26 — the second eigenvalue is much
smaller, so the cloud is nearly a line.

**Blind to:** anything that survives centring. **This is the important one.** Because it
subtracts the mean first, PR cannot see a collapse to a single non-zero point.

**The failure, concretely.** Suppose every image maps to nearly $(5,5,5,\dots)$:

```python
dead = 5.0 + 1e-4 * rng.normal(size=(500, 32))
```

Centring deletes the shared position, leaving only tiny random wobble. Random wobble is
spread evenly over all directions, so PR reads **30 out of 32** — "using almost every
dimension", i.e. maximally healthy. It is measuring noise.

**Measured in this project:** PR reads **25.45 on the degenerate control** versus **10.99 on
the healthy baseline** (Phase B, 5 seeds). It ranks the broken encoder as more than twice as
healthy as the working one. The paired 95% interval against `none_stopgrad` is
$[-32.70,\ -15.36]$ — a large, reproducible, *backwards* result.

---

## 4. RankMe

Established metric, not ours — Garrido et al., ICML 2023 ([arXiv:2210.02885](https://arxiv.org/abs/2210.02885)).

**Formula.** Take the singular values $\sigma_k$ of the **raw** (uncentred) matrix $Z$,
normalise them into a probability distribution, and exponentiate the entropy:

$$p_k = \frac{\sigma_k}{\sum_j \sigma_j}, \qquad
\mathrm{RankMe} = \exp\left(-\sum_k p_k \log p_k\right)$$

**In words.** Same question as PR — how many dimensions are in use — but answered from the
raw matrix and via entropy. Its floor is exactly 1 (one effective dimension).

**Worked example.** Singular values of the raw $Z$ are $(4.472,\ 3.162)$. Normalised:
$p = (0.586,\ 0.414)$. Entropy:

$$H = -(0.586\log 0.586 + 0.414\log 0.414) = 0.678$$

$$\mathrm{RankMe} = e^{0.678} = 1.97$$

**Blind to:** scale (it normalises), like PR and cosine. It cannot see pure shrinkage.

**The crucial difference from PR.** RankMe does **not** centre. So when every point piles up
at the same non-zero location, that shared position shows up as one enormous singular value
dominating all others, and RankMe correctly collapses toward 1. On the same `dead` array
above where PR says 30, **RankMe says 1.00**.

> Both are routinely called "effective rank". They are computed from different matrices —
> one centred, one not — and they disagree exactly when it matters. This is documented
> algebra, not a discovery of ours; our contribution is a concrete case where they invert.

---

## 5. Retrieval precision at k (P@k)

**Formula.** For each sample $i$, let $N_k(i)$ be its $k$ nearest neighbours by cosine
similarity, excluding itself. Then

$$\mathrm{P@}k = \frac{1}{n}\sum_{i=1}^{n} \frac{1}{k}\left|\{\,j \in N_k(i) : y_j = y_i\,\}\right|$$

**In words.** "Ask each image for its 10 most similar images; what fraction come back with
the same label?" This is exactly the operation scenario mining performs — *find me more
situations like this one*.

**Worked example.** Six points, two classes:

```
class 0: (1.0, 0.0)  (0.9, 0.1)  (0.8, 0.2)
class 1: (0.0, 1.0)  (0.1, 0.9)  (0.2, 0.8)
```

Each point's 2 nearest neighbours by angle are its two classmates, so **P@2 = 1.0**.

**The chance floor is not 1/K.** A random neighbour matches with probability equal to the
class self-match probability:

$$\mathrm{chance} = \sum_c f_c^2$$

where $f_c$ is class $c$'s frequency. **This is the single easiest number to get wrong in
this project.**

**Worked example.** Labels with frequencies $0.7, 0.2, 0.1$:

$$\mathrm{chance} = 0.7^2 + 0.2^2 + 0.1^2 = 0.49 + 0.04 + 0.01 = 0.54$$

Not $1/3 = 0.33$. On BDD100K the real chance floors are **0.41–0.46**, so a P@10 of 0.50
looks like "1.5× chance" but is barely above the floor.

**Measured here:** healthy 0.675, degenerate 0.566, chance 0.429. So even a badly degenerate
encoder retrieves at 1.32× chance.

---

## 6. Linear probe

**What it is.** Freeze the encoder. Take embeddings of images that *do* have labels. Fit
multinomial logistic regression from embedding → label. Report accuracy on held-out data.

The classifier is deliberately *linear* — the weakest useful model. The question is not "can
anything classify these images" but "did the encoder already do the work, leaving the classes
linearly separable".

**Two variants, and the difference is the whole story.**

- **Standardized:** divide each dimension by its standard deviation first
  ($x' = x/\sigma$). This is the field default.
- **Unstandardized:** leave the numbers as the encoder emitted them.

**Why the variants disagree.** Standardization removes scale by construction. So on a
representation that has *shrunk*, the standardized probe is measuring shape only.

⚠️ **But our comparison of the two is currently withdrawn.** The optimiser (lbfgs) stops
when the gradient norm falls below a tolerance; the gradient scales with feature magnitude;
so on features around $10^{-3}$ it stopped before fitting, predicted the majority class, and
returned chance — silently. See `primer.md` §7.2. The corrected protocol selects
regularisation on a validation split and records convergence.

---

## 7. Baselines you must quote beside any accuracy

**Majority rate.** Accuracy of always predicting the most common class.

$$\mathrm{majority} = \max_c f_c$$

Labels `[0]*7 + [1]*2 + [2]*1` → majority = **0.7**. A probe scoring 0.70 there has learned
nothing whatsoever.

**Balanced accuracy.** Mean per-class recall — accuracy computed per class, then averaged,
so each class counts equally regardless of size.

$$\mathrm{balanced} = \frac{1}{|C|}\sum_{c \in C} \frac{\text{correct in class } c}{\text{total in class } c}$$

**Worked example.** 8 samples of class 0, 2 of class 1. Predict class 0 every time:

| | value |
| --- | --- |
| raw accuracy | 8/10 = **0.80** |
| recall, class 0 | 8/8 = 1.00 |
| recall, class 1 | 0/2 = 0.00 |
| **balanced accuracy** | (1.00 + 0.00)/2 = **0.50** |

Raw accuracy says 80%. Balanced accuracy says 50% — chance for two classes. **This is why
balanced accuracy is the primary reading on BDD100K**, where weather is 61% `clear`.

**Macro-F1.** Like balanced accuracy but also penalises *over*-predicting a class. Balanced
accuracy only looks at recall, so a model that predicts "clear" very often can still score
well on the clear class; F1 combines precision and recall and catches that.

**Chance-adjusted score.** Rescales so 0 = chance and 1 = perfect:

$$\mathrm{adjusted} = \frac{\text{score} - \text{chance}}{1 - \text{chance}}$$

P@10 of 0.50 against a 0.46 floor → $(0.50-0.46)/(1-0.46) = 0.074$. Seven percent of the
available headroom, not "1.09× chance".

---

## 8. Statistical vocabulary

**Seed.** The random number that fixes initial weights, data order, and masking. Same seed
⇒ byte-identical setup. Different seed ⇒ a different draw of the same experiment.

**Paired.** Seed 3 of condition A and seed 3 of condition B share initialisation, data
order and evaluation images. So their difference isolates the condition. Comparing seed 3 of
A with seed 1 of B would confound the condition with the draw.

**Paired Student-t 95% interval.** For $n$ paired seeds, take the per-seed differences
$d_i$, then

$$\bar d \pm t_{0.975,\,n-1}\cdot \frac{s_d}{\sqrt{n}}$$

with $s_d$ the sample standard deviation (ddof 1). With $n=5$, $t_{0.975,4} = 2.776$.

**How to read it.** "The difference is somewhere in this range, with 95% confidence." An
interval excluding zero is evidence of a consistent difference. **A wide interval is not a
null result** — with 5 seeds the half-width is about 1.24 seed SDs, so effects smaller than
roughly 1.3 seed SDs cannot be resolved at all.

**Why no p-values or "significant".** The project's pre-registration prohibits verdict
language. Intervals are reported as estimates. Five contrasts × several endpoints without
multiplicity correction would make any significance claim misleading anyway.

---

## 9. Quick reference — what is blind to what

| | scale ↓ | angle → same | rank ↓ | shift off origin |
| --- | --- | --- | --- | --- |
| total variance | **sees** | blind | partly | blind |
| mean pairwise cosine | blind | **sees** | partly | **sees** |
| participation ratio | blind | blind | **sees** | blind |
| RankMe | blind | partly | **sees** | **sees** |

The two blind columns are complementary, which is why **no single metric is sufficient** and
why the recommended pair is **total variance + RankMe** — one from the scale-sensitive
family, one from the uncentred family. See `scripts/validate_panel.py`.
