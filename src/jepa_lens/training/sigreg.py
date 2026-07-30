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

    embed_dim = embeddings.shape[1]
    device = embeddings.device

    # Draw on the generator's own device, then move. torch.randn rejects a
    # generator whose device differs from the target ("Expected a 'cuda' device
    # type for generator but found 'cpu'"), and the strategy holds a CPU
    # generator so that slice directions stay reproducible across machines —
    # this project is developed on CPU and trained on GPU, and a run should not
    # depend on which one produced it.
    draw_device = generator.device if generator is not None else device
    directions = torch.randn(embed_dim, num_slices, generator=generator, device=draw_device)
    directions = directions.to(device=device, dtype=embeddings.dtype)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)

    # Center only. Do NOT standardize each slice: the variance of a projection
    # varies by direction exactly when the distribution is anisotropic, so
    # rescaling each slice to unit variance destroys the signal isotropy is
    # defined by. With per-slice standardization this loss measures only
    # Gaussian *shape*, and a heavily anisotropic batch scores better than an
    # isotropic one — the opposite of the intent.
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    projections = centered @ directions

    frequencies = torch.linspace(freq_max / num_freqs, freq_max, num_freqs, device=device)
    angles = projections.unsqueeze(-1) * frequencies.view(1, 1, -1)

    empirical_real = torch.cos(angles).mean(dim=0)
    empirical_imag = torch.sin(angles).mean(dim=0)
    target_real = torch.exp(-0.5 * frequencies**2).view(1, -1)

    squared_error = (empirical_real - target_real) ** 2 + empirical_imag**2

    # Epps-Pulley weights the discrepancy by a Gaussian kernel so low
    # frequencies, where the estimate is most reliable, dominate.
    weights = torch.exp(-0.5 * frequencies**2).view(1, -1)
    weighted = (squared_error * weights).sum(dim=1) / weights.sum()

    # No batch-size scaling. The Epps-Pulley test statistic carries a factor of
    # n for its asymptotic null distribution, but as a training loss that only
    # makes the magnitude — and therefore the meaning of `sigreg_weight` —
    # depend on batch size.
    return weighted.mean()
