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
