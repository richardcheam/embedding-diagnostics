"""A small Vision Transformer encoder with patch-level masking.

Architecture follows the standard ViT recipe (Dosovitskiy et al., 2020) at a
scale chosen to fit CIFAR-10 experiments on a single GPU. Masked encoding —
encoding only the visible patches rather than substituting mask tokens —
follows I-JEPA (arXiv 2301.08243).
"""

from __future__ import annotations

import torch
from torch import nn


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with multi-head self-attention and an MLP."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(tokens)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class ViTEncoder(nn.Module):
    """Patch-embed, add position embeddings, optionally drop masked patches, encode."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("patch_size must evenly divide image_size")
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        images: torch.Tensor,
        keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode images, optionally restricted to the patches `keep_mask` selects.

        Args:
            images: (batch, 3, height, width).
            keep_mask: optional bool tensor (batch, num_patches). True keeps the
                patch. Every row must keep the same number of patches so the
                result stays a dense tensor.

        Returns:
            Tuple of (tokens, pooled). `tokens` is (batch, kept, embed_dim);
            `pooled` is the mean over kept tokens, (batch, embed_dim).
        """
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed

        if keep_mask is not None:
            counts = keep_mask.sum(dim=1)
            if not bool((counts == counts[0]).all()):
                raise ValueError("keep_mask must keep the same number of patches per sample")
            batch, _, dim = tokens.shape
            indices = keep_mask.nonzero(as_tuple=True)[1].view(batch, int(counts[0]))
            tokens = torch.gather(tokens, 1, indices.unsqueeze(-1).expand(-1, -1, dim))

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens, tokens.mean(dim=1)
