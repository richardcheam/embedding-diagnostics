"""Collapse-prevention strategies, the one piece that differs between conditions.

Every condition runs through the same `Trainer`. If each condition had its own
training loop, a difference in results could come from an incidental difference
in the loop rather than from the mechanism under test. Keeping the swappable
surface this narrow is a correctness requirement of the experiment, not a
stylistic preference.

Condition table:

    condition            uses_ema_target  detaches_target  regularizer
    ema_stopgrad         yes              yes              none
    sigreg_stopgrad      no               yes              SIGReg
    sigreg_nostopgrad    no               no               SIGReg
    none_nostopgrad      no               no               none

The fourth condition is the sanity control: one shared encoder, no detachment,
no regularizer, so nothing at all prevents the constant solution. It must
collapse. If it does not, the measurement apparatus cannot detect the
phenomenon under study and no other condition's result can be trusted.

Note on why it is not an EMA condition: with a frozen EMA target encoder, the
target branch reaches no trainable parameter, so `detaches_target=False` would
have no numerical effect and the condition would silently duplicate
`ema_stopgrad`. Removing collapse prevention requires removing the EMA split
too.

EMA target encoders and stop-gradient come from the BYOL/SimSiam lineage and
are used by I-JEPA (arXiv 2301.08243). SIGReg comes from LeJEPA
(arXiv 2511.08544). Neither is a contribution of this project.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .sigreg import sigreg_loss


@dataclass
class LossOutput:
    """Total loss plus a breakdown for logging."""

    total: torch.Tensor
    components: dict[str, float] = field(default_factory=dict)


class CollapsePreventionStrategy(ABC):
    """One collapse-prevention mechanism, plugged into the shared trainer."""

    def __init__(self, uses_ema_target: bool, detaches_target: bool) -> None:
        self.uses_ema_target = uses_ema_target
        self.detaches_target = detaches_target

    @abstractmethod
    def compute_loss(
        self,
        prediction: torch.Tensor,
        target_latent: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> LossOutput:
        """Combine the prediction loss with any regularizer.

        Args:
            prediction: (batch, num_targets, embed_dim) predicted target embeddings.
            target_latent: (batch, num_targets, embed_dim) encoder targets, already
                detached by the trainer if `detaches_target` is True.
            context_embedding: (batch, embed_dim) pooled context representation.
                Regularizers apply here because this branch always carries
                gradient — regularizing a detached target would be a no-op in
                the stop-gradient conditions.
        """

    def post_step_update(self, context_encoder: nn.Module, target_encoder: nn.Module) -> None:
        """Hook run after `optimizer.step()`. Default is no update."""
        return None


def _prediction_loss(prediction: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(prediction, target_latent)


class EMAStrategy(CollapsePreventionStrategy):
    """Classic JEPA: a separate EMA target encoder, no explicit regularizer."""

    def __init__(self, uses_ema_target: bool, detaches_target: bool, ema_decay: float) -> None:
        super().__init__(uses_ema_target, detaches_target)
        self.ema_decay = ema_decay

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target_latent: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> LossOutput:
        loss = _prediction_loss(prediction, target_latent)
        return LossOutput(total=loss, components={"prediction": float(loss.detach())})

    @torch.no_grad()
    def post_step_update(self, context_encoder: nn.Module, target_encoder: nn.Module) -> None:
        decay = self.ema_decay
        for target_param, context_param in zip(
            target_encoder.parameters(), context_encoder.parameters(), strict=True
        ):
            target_param.mul_(decay).add_(context_param, alpha=1.0 - decay)
        for target_buffer, context_buffer in zip(
            target_encoder.buffers(), context_encoder.buffers(), strict=True
        ):
            target_buffer.copy_(context_buffer)


class SIGRegStrategy(CollapsePreventionStrategy):
    """LeJEPA-style: SIGReg on the context embeddings, shared encoder."""

    def __init__(
        self,
        uses_ema_target: bool,
        detaches_target: bool,
        weight: float,
        num_slices: int,
        num_freqs: int,
        freq_max: float,
        seed: int = 0,
    ) -> None:
        super().__init__(uses_ema_target, detaches_target)
        self.weight = weight
        self.num_slices = num_slices
        self.num_freqs = num_freqs
        self.freq_max = freq_max
        # A dedicated generator, NOT the global RNG. Drawing slice directions
        # from the global RNG would consume it once per step in the SIGReg
        # conditions only, and the DataLoader reseeds its shuffle and worker
        # augmentation from that same global RNG each epoch — so the SIGReg and
        # non-SIGReg conditions would diverge in data order and augmentation.
        # That is a plumbing-induced difference between conditions, which is
        # precisely what the shared-loop design exists to rule out.
        self.generator = torch.Generator().manual_seed(seed)

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target_latent: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> LossOutput:
        prediction_term = _prediction_loss(prediction, target_latent)
        regularizer = sigreg_loss(
            context_embedding,
            num_slices=self.num_slices,
            num_freqs=self.num_freqs,
            freq_max=self.freq_max,
            generator=self.generator,
        )
        weighted = self.weight * regularizer
        total = prediction_term + weighted
        # Two keys on purpose. `sigreg` is the raw isotropy penalty — a
        # diagnostic worth reading on its own, independent of how hard it is
        # being weighted. `sigreg_weighted` is its actual contribution to
        # `total`, so that prediction + sigreg_weighted == total holds for any
        # weight. Reporting only the raw value would misattribute the loss
        # whenever sigreg_weight != 1.
        return LossOutput(
            total=total,
            components={
                "prediction": float(prediction_term.detach()),
                "sigreg": float(regularizer.detach()),
                "sigreg_weighted": float(weighted.detach()),
            },
        )


class NoPreventionStrategy(CollapsePreventionStrategy):
    """No collapse prevention at all: the sanity control, expected to collapse."""

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target_latent: torch.Tensor,
        context_embedding: torch.Tensor,
    ) -> LossOutput:
        loss = _prediction_loss(prediction, target_latent)
        return LossOutput(total=loss, components={"prediction": float(loss.detach())})


def build_strategy(
    strategy_config: dict[str, Any],
    seed: int = 0,
) -> CollapsePreventionStrategy:
    """Construct the strategy named by `strategy_config['name']`.

    `seed` is the run seed, threaded in so SIGReg's slice-direction generator
    tracks the run rather than being pinned to a constant. It is passed
    separately because `strategy_config` is only the `strategy:` block and does
    not carry the top-level seed.
    """
    name = strategy_config["name"]
    uses_ema = bool(strategy_config["uses_ema_target"])
    detaches = bool(strategy_config["detaches_target"])

    if name == "none_nostopgrad":
        return NoPreventionStrategy(uses_ema, detaches)
    if name == "ema_stopgrad":
        return EMAStrategy(uses_ema, detaches, float(strategy_config["ema_decay"]))
    if name in {"sigreg_stopgrad", "sigreg_nostopgrad"}:
        return SIGRegStrategy(
            uses_ema,
            detaches,
            weight=float(strategy_config["sigreg_weight"]),
            num_slices=int(strategy_config["sigreg_num_slices"]),
            num_freqs=int(strategy_config["sigreg_num_freqs"]),
            freq_max=float(strategy_config["sigreg_freq_max"]),
            seed=seed,
        )
    raise ValueError(f"unknown strategy {name!r}")
