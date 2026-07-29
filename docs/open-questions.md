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

## SIGReg fails to penalize anisotropy (found during Task 8 TDD, resolved)

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
would invalidate any study built on it. `sigreg_loss` was first committed as
specified in the task-8 brief (5 of 6 brief tests passing), with this
anisotropy test left failing and documented above rather than silently
patched.

### Resolution

The coordinator confirmed the diagnosis above was correct and corrected the
brief. The fix, applied in a follow-up commit on `feat/harness`:

1. **Removed per-slice standardization.** `sigreg_loss` now only centers each
   projection (`embeddings - embeddings.mean(dim=0, keepdim=True)`) before
   computing its empirical characteristic function; it no longer divides by
   the per-slice std. This restores the per-direction variance signal that
   standardization had erased, so the loss now measures "is this a *standard*
   isotropic Gaussian" rather than merely "is each marginal Gaussian-shaped."
2. **Removed the `* batch` scaling on the returned loss.** The Epps-Pulley
   test statistic carries a factor of `n` for its asymptotic null
   distribution, but multiplying a training loss by batch size just makes the
   loss magnitude — and therefore the effective meaning of a `sigreg_weight`
   hyperparameter — depend on batch size, which is undesirable for a
   regularizer used across configs with different batch sizes.
3. **Added `test_wrong_scale_is_penalized`**, which pins that the loss is
   sensitive to both inflated and shrunk scale relative to the standard
   Gaussian — the exact property the old standardization step made
   invisible.

All 7 tests in `tests/test_sigreg.py` now pass, including
`test_gaussian_scores_lower_than_heavily_anisotropic`.

### Why this raises rather than lowers the priority of the fidelity check above

This bug was caught by a behavioral test (isotropic vs. anisotropic
comparison), not by a review of the maths against the paper or the reference
implementation. That is exactly the failure mode the "SIGReg implementation
fidelity" section above warns about: a summary-level understanding of a
method can silently encode a wrong instinct (here, "standardize before
comparing distributions," which is reasonable for a shape test but wrong for
an isotropy test) that only surfaces if someone happens to write a test that
exercises the specific property it breaks. The test suite here is not
exhaustive — there is no guarantee another such gap isn't still present in,
e.g., the frequency grid, the slice-direction sampling, or the weighting
function. If anything, this episode is a concrete demonstration of why
`sigreg_loss` must still be validated against
https://github.com/rbalestr-lab/lejepa before any result produced with it is
described as a reproduction of LeJEPA, faithful or otherwise. The module
docstring's unvalidated-provenance warning stands unchanged.
