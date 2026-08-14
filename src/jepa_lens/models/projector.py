"""Disposable projector between the encoder and the distributional regularizer.

Established practice, not a contribution of this project. VICReg, SimCLR-family
methods, and LeJEPA all apply their embedding-space losses to the output of a
small MLP head rather than to the representation that is actually kept: the
projector absorbs the distortions the loss demands, so the encoder underneath
can keep structure the loss would otherwise squeeze out. Discarding it is
reported to cost substantial accuracy in SimCLR and VICReg (arXiv 2212.11491;
SSL Cookbook, arXiv 2304.12210).

NAMING (advisor item 5). This is a PROJECT-SPECIFIC DISPOSABLE MLP BUFFER
INSPIRED BY LeJEPA, not a reproduction of LeJEPA's projector. The reference at
github.com/rbalestr-lab/lejepa, commit c293d291, `MINIMAL.md:88`, builds

    self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

i.e. three linear layers with BatchNorm1d and a 4x hidden expansion over a
512-d encoder. Ours is two linear layers, GELU, no normalization, over a 192-d
encoder. Matching an output dimension establishes nothing about architectural
fidelity, and no claim of an exact match is made anywhere in this project.

The magnitude of the projector effect reported for SimCLR/VICReg should NOT be
expected here: those are multi-view contrastive and variance-covariance systems,
whereas this is a masked latent-prediction hybrid with a different companion
objective. The literature establishes that the placement matters, not how much
it matters in this setting — which is what the projector conditions measure.

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
