"""The single training loop shared by all four conditions.

Only the `CollapsePreventionStrategy` differs between conditions. Everything
else — architecture, data, optimizer, schedule, seeding, measurement — is
identical by construction, so an observed difference is attributable to the
mechanism under test rather than to an incidental difference in the loop.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from ..data import sample_block_masks
from ..diagnostics.metrics import collapse_metrics
from ..diagnostics.probe import linear_probe_accuracy
from ..diagnostics.projection import project_2d
from ..logging_utils import RunLogger
from ..models.predictor import MLPPredictor
from ..models.vit import ViTEncoder
from .strategy import LossOutput, build_strategy


@dataclass
class ForwardOutput:
    """Intermediates from one forward pass, exposed so tests can inspect them."""

    loss_output: LossOutput
    target_latent: torch.Tensor
    context_embedding: torch.Tensor


class Trainer:
    """Owns the models, optimizer, strategy, and measurement for one condition."""

    def __init__(self, config: dict[str, Any], device: str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        torch.manual_seed(config["seed"])

        model_config = config["model"]
        self.context_encoder = ViTEncoder(
            image_size=model_config["image_size"],
            patch_size=model_config["patch_size"],
            embed_dim=model_config["embed_dim"],
            depth=model_config["depth"],
            num_heads=model_config["num_heads"],
            mlp_ratio=model_config["mlp_ratio"],
        ).to(self.device)

        self.strategy = build_strategy(config["strategy"], seed=config["seed"])

        if self.strategy.uses_ema_target:
            # A separate encoder updated only by EMA, never by the optimizer.
            self.target_encoder = copy.deepcopy(self.context_encoder).to(self.device)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
        else:
            # LeJEPA removes the teacher-student split: one encoder, both branches.
            self.target_encoder = self.context_encoder

        self.predictor = MLPPredictor(
            embed_dim=model_config["embed_dim"],
            hidden_dim=model_config["predictor_hidden_dim"],
            num_patches=self.context_encoder.num_patches,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self.context_encoder.parameters()) + list(self.predictor.parameters()),
            lr=config["optim"]["lr"],
            weight_decay=config["optim"]["weight_decay"],
        )
        self.generator = torch.Generator().manual_seed(config["seed"])
        self.step = 0

    def _learning_rate(self) -> float:
        """Linear warmup then cosine decay."""
        optim_config = self.config["optim"]
        warmup = optim_config["warmup_steps"]
        total = optim_config["total_steps"]
        base = optim_config["lr"]
        if self.step < warmup:
            return base * (self.step + 1) / warmup
        progress = (self.step - warmup) / max(total - warmup, 1)
        return base * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    def forward_pass(self, images: torch.Tensor) -> ForwardOutput:
        """Run one forward pass and return the loss plus inspectable intermediates."""
        images = images.to(self.device)
        masking = self.config["masking"]
        context_mask, target_indices = sample_block_masks(
            batch_size=images.shape[0],
            grid_size=self.context_encoder.grid_size,
            num_target_blocks=masking["num_target_blocks"],
            block_scale=tuple(masking["target_block_scale"]),
            block_aspect=tuple(masking["target_block_aspect"]),
            generator=self.generator,
        )
        context_mask = context_mask.to(self.device)
        target_indices = target_indices.to(self.device)

        _, context_embedding = self.context_encoder(images, context_mask)
        prediction = self.predictor(context_embedding, target_indices)

        if self.strategy.detaches_target:
            with torch.no_grad():
                target_tokens, _ = self.target_encoder(images)
            target_latent = target_tokens.detach()
        else:
            target_tokens, _ = self.target_encoder(images)
            target_latent = target_tokens

        batch = images.shape[0]
        gather_index = target_indices.unsqueeze(-1).expand(-1, -1, target_latent.shape[-1])
        target_latent = torch.gather(target_latent, 1, gather_index)
        assert target_latent.shape[0] == batch

        loss_output = self.strategy.compute_loss(prediction, target_latent, context_embedding)
        return ForwardOutput(loss_output, target_latent, context_embedding)

    def train_step(self, images: torch.Tensor) -> dict[str, float]:
        """One optimizer step, then the strategy's post-step hook."""
        self.context_encoder.train()
        self.predictor.train()

        # Capture once. `_learning_rate()` reads `self.step`, which is
        # incremented below, so recomputing it after the step would report the
        # NEXT step's rate alongside this step's loss.
        learning_rate = self._learning_rate()
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

        self.optimizer.zero_grad(set_to_none=True)
        output = self.forward_pass(images)
        output.loss_output.total.backward()
        self.optimizer.step()
        self.strategy.post_step_update(self.context_encoder, self.target_encoder)
        self.step += 1

        metrics = dict(output.loss_output.components)
        metrics["total"] = float(output.loss_output.total.detach())
        metrics["lr"] = learning_rate
        return metrics

    @torch.no_grad()
    def encode_all(self, images: torch.Tensor, batch_size: int = 256) -> np.ndarray:
        """Pooled embeddings from the context encoder, in eval mode."""
        self.context_encoder.eval()
        chunks = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(self.device)
            _, pooled = self.context_encoder(batch)
            chunks.append(pooled.cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def evaluate(self, probe_train: tuple, probe_test: tuple) -> dict[str, Any]:
        """Cheap diagnostics plus the expensive probe, on the frozen encoder."""
        train_images, train_labels = probe_train
        test_images, test_labels = probe_test

        train_features = self.encode_all(train_images)
        test_features = self.encode_all(test_images)

        record: dict[str, Any] = collapse_metrics(test_features)
        record["probe_accuracy"] = linear_probe_accuracy(
            train_features, train_labels, test_features, test_labels, seed=self.config["seed"]
        )
        record["projection"] = project_2d(
            test_features,
            max_samples=self.config["logging"]["projection_samples"],
            seed=self.config["seed"],
        )
        return record

    def fit(
        self,
        ssl_loader,
        probe_train: tuple,
        probe_test: tuple,
        logger: RunLogger,
    ) -> None:
        """Train to `total_steps`, checkpointing diagnostics along the way."""
        total_steps = self.config["optim"]["total_steps"]
        checkpoint_every = self.config["logging"]["checkpoint_every"]
        condition = self.config["strategy"]["name"]

        train_metrics: dict[str, float] = {}
        data_iterator = iter(ssl_loader)

        while self.step < total_steps:
            try:
                images, _ = next(data_iterator)
            except StopIteration:
                data_iterator = iter(ssl_loader)
                images, _ = next(data_iterator)

            step_at_checkpoint = self.step % checkpoint_every == 0
            if step_at_checkpoint:
                record = self.evaluate(probe_train, probe_test)
                record["step"] = self.step
                record["condition"] = condition
                record.update({f"loss_{k}": v for k, v in train_metrics.items()})
                logger.log(record)

            train_metrics = self.train_step(images)

        record = self.evaluate(probe_train, probe_test)
        record["step"] = self.step
        record["condition"] = condition
        record.update({f"loss_{k}": v for k, v in train_metrics.items()})
        logger.log(record)


def count_parameters(module: nn.Module) -> int:
    """Trainable parameter count, reported at run start."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
