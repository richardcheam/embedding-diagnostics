"""Sketched Isotropic Gaussian Regularization (SIGReg).

SIGReg is introduced by LeJEPA (arXiv 2511.08544). It is not a contribution of
this project.

PROVENANCE. This module was first written from a summary-level understanding of
the paper. It has since been checked line by line against the reference
implementation at https://github.com/rbalestr-lab/lejepa (`lejepa/univariate/
epps_pulley.py::EppsPulley` and `lejepa/multivariate/slicing.py::
SlicingUnivariateTest`) and rewritten to match it. Remaining known deviations
are listed at the bottom of this docstring.

Method:

1. Draw `num_slices` random directions in embedding space, normalized to unit
   L2 norm.
2. Project the embeddings onto each direction, giving univariate samples.
3. Compare each projection's empirical characteristic function against that of
   a standard 1D Gaussian, via the Epps-Pulley goodness-of-fit statistic, and
   average over slices.

The Cramer-Wold device motivates the sketching: a distribution is determined by
its 1D projections, so matching many random 1D marginals to a standard normal
avoids estimating a high-dimensional density directly.

DO NOT CENTER THE EMBEDDINGS. An earlier version of this file subtracted the
per-feature mean before projecting. That was our own invention, not from the
paper, and it silently destroyed the regularizer. A collapsed encoder emits
some constant vector c; centering maps that exactly onto the origin, where the
characteristic function is flat (`cos(0) = 1`, derivative zero), so the loss
saturated while its gradient decayed to nothing. Measured on a batch collapsing
to a constant: the reference keeps a restoring-force gradient norm of ~3.6
throughout, while the centered version fell from 2.5e-4 to 2.5e-6. Collapse had
become a stationary point of the very objective meant to prevent it.

Known deviations from the reference, all deliberate:

- The reference integrates its projection directions from a `global_step`
  counter synchronized across ranks; we take a `torch.Generator` instead, since
  this project runs one process per GPU and needs the generator threaded
  explicitly for reproducibility (see docs/development-log.md).
- The reference supports distributed all-reduce of the characteristic function
  across ranks. We omit it: this project never shards a batch, precisely
  because these are batch-level statistics.
- The reference offers a `clip_value` that zeroes small per-slice statistics.
  We do not use it.
"""

from __future__ import annotations

import torch


def sigreg_loss(
    embeddings: torch.Tensor,
    num_slices: int = 1024,
    num_freqs: int = 17,
    freq_max: float = 3.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Penalize departure of the embedding distribution from isotropic Gaussian.

    Args:
        embeddings: (batch, embed_dim) tensor. Gradients flow through this. Pass
            the encoder's raw output; do not center or normalize it first.
        num_slices: number of random 1D projections. Reference default 1024.
        num_freqs: integration points over [0, freq_max], including 0. Must be
            odd, matching the reference's assertion. Reference default 17.
        freq_max: largest frequency in the integration grid. Reference default 3.
        generator: optional generator for reproducible slice directions.

    Returns:
        Non-negative scalar, scaled by batch size as the Epps-Pulley statistic
        is. Zero would mean the projections are exactly standard normal at every
        integration point.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"expected (batch, embed_dim), got shape {tuple(embeddings.shape)}")
    if num_freqs % 2 == 0:
        raise ValueError(f"num_freqs must be odd for the trapezoid rule, got {num_freqs}")

    batch, embed_dim = embeddings.shape
    device = embeddings.device

    # Draw on the generator's own device, then move. torch.randn rejects a
    # generator whose device differs from the target, and the strategy holds a
    # CPU generator so slice directions stay reproducible across machines.
    draw_device = generator.device if generator is not None else device
    directions = torch.randn(embed_dim, num_slices, generator=generator, device=draw_device)
    directions = directions.to(device=device, dtype=embeddings.dtype)
    directions = directions / directions.norm(p=2, dim=0).clamp_min(1e-12)

    # No centering, no standardization. See the module docstring.
    projections = embeddings @ directions

    # Trapezoid rule over [0, freq_max]. Only t >= 0 is evaluated; the
    # contribution from -t is identical by symmetry and is folded in by
    # doubling the interior weights, with half weight at each endpoint.
    frequencies = torch.linspace(0.0, freq_max, num_freqs, device=device)
    step = freq_max / (num_freqs - 1)
    weights = torch.full((num_freqs,), 2.0 * step, device=device)
    weights[0] = step
    weights[-1] = step
    target_real = torch.exp(-0.5 * frequencies**2)
    weights = weights * target_real

    angles = projections.unsqueeze(-1) * frequencies
    empirical_real = torch.cos(angles).mean(dim=0)
    empirical_imag = torch.sin(angles).mean(dim=0)

    squared_error = (empirical_real - target_real).square() + empirical_imag.square()

    # Scaled by batch size, following the Epps-Pulley statistic and the
    # reference implementation.
    return (squared_error @ weights).mean() * batch
