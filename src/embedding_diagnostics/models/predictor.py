"""Position-conditioned MLP predictor.

I-JEPA (arXiv 2301.08243) conditions its predictor on the position of the
target being predicted, so the predictor knows *where* it is predicting rather
than producing one position-agnostic guess. This module keeps that conditioning
but uses an MLP rather than a narrow transformer, which is sufficient for a
study whose subject is the collapse behaviour of the encoder.
"""

from __future__ import annotations

import torch
from torch import nn


class MLPPredictor(nn.Module):
    """Predict a target patch embedding from pooled context plus target position."""

    def __init__(self, embed_dim: int, hidden_dim: int, num_patches: int) -> None:
        super().__init__()
        self.target_pos_embed = nn.Parameter(torch.zeros(num_patches, embed_dim))
        nn.init.trunc_normal_(self.target_pos_embed, std=0.02)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self, pooled_context: torch.Tensor, target_indices: torch.Tensor
    ) -> torch.Tensor:
        """Predict embeddings for the requested target patch positions.

        Args:
            pooled_context: (batch, embed_dim) pooled context representation.
            target_indices: (batch, num_targets) long tensor of patch indices.

        Returns:
            (batch, num_targets, embed_dim) predicted target embeddings.
        """
        num_targets = target_indices.shape[1]
        positions = self.target_pos_embed[target_indices]
        context = pooled_context.unsqueeze(1).expand(-1, num_targets, -1)
        return self.net(torch.cat([context, positions], dim=-1))
