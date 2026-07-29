# Open questions

## SIGReg implementation fidelity

`src/jepa_lens/training/sigreg.py` was written from a summary-level
understanding of LeJEPA (arXiv 2511.08544), not an end-to-end reading.

**Not yet done:** compare against the reference implementation at
https://github.com/rbalestr-lab/lejepa — specifically the slice-direction
sampling, the frequency grid, the weighting function, and the batch-size
scaling.

**Until then:** results are labelled `[interpretation]`, not `[replication]`,
and the report must not call this a faithful reproduction of LeJEPA.

**If a discrepancy is found:** record it here, fix the implementation, and note
in the report whether any already-collected results were affected.

## SIGReg fails to penalize anisotropy (found during Task 8 TDD, unresolved)

While implementing Task 8, `tests/test_sigreg.py::test_gaussian_scores_lower_than_heavily_anisotropic`
failed consistently (5/5 seeds tried) against the implementation given verbatim
in the task-8 brief: a heavily anisotropic embedding batch (15 of 16 dims
scaled by 0.01) scored a *lower* (better) SIGReg loss than a genuinely
isotropic Gaussian batch of the same shape.

Root cause, as far as it was diagnosed without editing the implementation: the
per-slice standardization step

```python
mean = projections.mean(dim=0, keepdim=True)
std = projections.std(dim=0, keepdim=True).clamp_min(1e-6)
standardized = (projections - mean) / std
```

rescales every projected slice to unit variance *before* comparing its
characteristic function to a standard normal. This discards the per-direction
variance signal that isotropy is defined by — a projection dominated by a
single near-Gaussian coordinate, after being forcibly rescaled to unit
variance, looks just as "standard normal" as a projection of a genuinely
isotropic batch. Measured directly: for the anisotropic test fixture, raw
per-slice projection std ranged from ~0.01 to ~0.6 across 64 random
directions (mean ≈ 0.21) before standardization — that spread is exactly the
anisotropy signal SIGReg should be penalizing, and it is erased before the
characteristic-function comparison runs.

This was **not fixed** by tuning constants — per project policy, a
regularizer should not be silently adjusted until a test passes, since that
would invalidate any study built on it. `sigreg_loss` was committed as
specified in the task-8 brief (5 of 6 brief tests pass); this anisotropy test
currently fails and is left failing / xfail pending a decision.

**Before this loss is used for real training runs:** either (a) the
standardization step needs to be reconsidered against the actual LeJEPA
reference implementation (see above — this may be the same fidelity gap), or
(b) the test's expectation needs re-examination if per-slice standardization
turns out to be intentional in the reference method for a reason not
understood here. Do not treat SIGReg-regularized runs as trustworthy isotropy
control until this is resolved.
