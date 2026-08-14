"""Disposable projector between the encoder and the distributional regularizer.

Established practice, not a contribution of this project. VICReg, SimCLR-family
methods, and LeJEPA all apply their embedding-space losses to the output of a
small MLP head rather than to the representation that is actually kept: the
projector absorbs the distortions the loss demands, so the encoder underneath
can keep structure the loss would otherwise squeeze out. Removing the projector
is reported to cost on the order of 20 accuracy points in SimCLR/VICReg
(arXiv 2212.11491; SSL Cookbook, arXiv 2304.12210). LeJEPA's launcher uses
`projector_arch="MLP", projector_dim=512`.

This project ran its first experiments WITHOUT a projector — SIGReg acted
directly on the representation being probed — and no SIGReg configuration beat
its own random initialisation. Whether the projector is what unlocks
SIGReg-without-stop-gradient is precisely the Phase-2 question, so the module
is config-gated (`model.projector_dim: 0` disables it) rather than always on.

The projector's output feeds ONLY the regularizer. The probe, the collapse
diagnostics, and the prediction task all operate on the encoder output. The
projector is discarded at evaluation.
"""

from __future__ import annotations

from torch import nn


class Projector(nn.Module):
    """MLP head whose output the regularizer sees and evaluation never does."""

    def __init__(self, embed_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, embedding):
        return self.net(embedding)
