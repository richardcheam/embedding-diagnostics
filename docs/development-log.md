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
