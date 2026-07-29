"""Sketched Isotropic Gaussian Regularization (SIGReg).

SIGReg is introduced by LeJEPA (arXiv 2511.08544). It is not a contribution of
this project.

IMPORTANT — implementation provenance. This module was written from a
summary-level understanding of the method, not from an end-to-end reading of
the paper, and has NOT yet been validated against the reference implementation
at https://github.com/rbalestr-lab/lejepa. Until that validation happens, do
not describe results produced with it as a faithful reproduction of LeJEPA.
Any discrepancy found during validation must be recorded in the report.

Method as implemented here:

1. Draw random unit directions ("slices") in embedding space.
2. Project embeddings onto each slice, giving univariate samples.
3. Compare each projection's empirical characteristic function against that of
   a standard 1D Gaussian, following the Epps-Pulley goodness-of-fit test.

The Cramer-Wold device motivates the sketching: a distribution is determined by
its 1D projections, so matching many random 1D marginals to a standard normal
avoids estimating a high-dimensional density directly.
"""

from __future__ import annotations

import torch


def sigreg_loss(
    embeddings: torch.Tensor,
    num_slices: int = 64,
    num_freqs: int = 16,
    freq_max: float = 5.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Penalize departure of the embedding distribution from isotropic Gaussian.

    Args:
        embeddings: (batch, embed_dim) tensor. Gradients flow through this.
        num_slices: number of random 1D projections.
        num_freqs: number of frequencies at which the characteristic functions
            are compared.
        freq_max: largest frequency in the comparison grid.
        generator: optional generator for reproducible slice directions.

    Returns:
        Non-negative scalar. Zero would mean the projections are exactly
        standard normal at every sampled frequency.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"expected (batch, embed_dim), got shape {tuple(embeddings.shape)}")

    batch, embed_dim = embeddings.shape
    device = embeddings.device

    directions = torch.randn(embed_dim, num_slices, generator=generator, device=device)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)

    projections = embeddings @ directions

    # Standardize each slice so the test measures distributional shape. Without
    # this the loss would be dominated by scale, which weight decay already
    # constrains.
    mean = projections.mean(dim=0, keepdim=True)
    std = projections.std(dim=0, keepdim=True).clamp_min(1e-6)
    standardized = (projections - mean) / std

    frequencies = torch.linspace(freq_max / num_freqs, freq_max, num_freqs, device=device)
    angles = standardized.unsqueeze(-1) * frequencies.view(1, 1, -1)

    empirical_real = torch.cos(angles).mean(dim=0)
    empirical_imag = torch.sin(angles).mean(dim=0)
    target_real = torch.exp(-0.5 * frequencies**2).view(1, -1)

    squared_error = (empirical_real - target_real) ** 2 + empirical_imag**2

    # Epps-Pulley weights the discrepancy by a Gaussian kernel so low
    # frequencies, where the estimate is most reliable, dominate.
    weights = torch.exp(-0.5 * frequencies**2).view(1, -1)
    weighted = (squared_error * weights).sum(dim=1) / weights.sum()

    # Scaling by batch size follows the Epps-Pulley statistic and keeps the
    # penalty comparable across batch sizes.
    return weighted.mean() * batch
