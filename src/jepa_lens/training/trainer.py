"""The single training loop shared by every condition.

Only the `CollapsePreventionStrategy` differs between conditions. Everything
else — architecture, data, optimizer, schedule, seeding, measurement — is
identical by construction, so an observed difference is attributable to the
mechanism under test rather than to an incidental difference in the loop.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..data import sample_block_masks
from ..diagnostics.metrics import collapse_metrics
from ..diagnostics.probe import chance_adjusted, linear_probe_scores, retrieval_chance
from ..diagnostics.projection import project_2d
from ..diagnostics.retrieval import (
    retrieval_macro_precision_at_k,
    retrieval_precision_at_k,
)
from ..logging_utils import RunLogger
from ..models.predictor import MLPPredictor
from ..models.projector import Projector
from ..models.vit import ViTEncoder
from .strategy import LossOutput, build_strategy


@dataclass
class ForwardOutput:
    """Intermediates from one forward pass, exposed so tests can inspect them."""

    loss_output: LossOutput
    target_latent: torch.Tensor
    context_embedding: torch.Tensor
    reg_embedding: torch.Tensor


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

        # Disposable buffer between the encoder and the regularizer, matching
        # LeJEPA's own setup (projector_dim=512). When present, SIGReg acts on
        # its output and the probed representation is never regularized
        # directly; when projector_dim is 0 the regularizer sees the pooled
        # context embedding exactly as before. The probe and the diagnostics
        # always read the encoder — never the projector.
        projector_dim = int(model_config.get("projector_dim", 0))
        self.projector = (
            Projector(
                embed_dim=model_config["embed_dim"],
                hidden_dim=int(model_config.get("projector_hidden_dim", projector_dim)),
                out_dim=projector_dim,
            ).to(self.device)
            if projector_dim > 0
            else None
        )

        trainable = list(self.context_encoder.parameters()) + list(self.predictor.parameters())
        if self.projector is not None:
            trainable += list(self.projector.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable,
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

        reg_embedding = (
            self.projector(context_embedding) if self.projector is not None else context_embedding
        )
        loss_output = self.strategy.compute_loss(prediction, target_latent, reg_embedding)
        return ForwardOutput(loss_output, target_latent, context_embedding, reg_embedding)

    def train_step(self, images: torch.Tensor) -> dict[str, float]:
        """One optimizer step, then the strategy's post-step hook."""
        self.context_encoder.train()
        self.predictor.train()
        if self.projector is not None:
            self.projector.train()

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
        """Cheap diagnostics, probes, and retrieval, on the frozen encoder.

        Labels may be a single array (CIFAR-10 classes) or a dict of named
        attribute arrays (BDD100K weather/scene/timeofday). With a dict, each
        attribute is probed and retrieved separately under suffixed keys, and
        the across-attribute means are logged under the canonical keys so every
        downstream tool — figures, summaries, seed aggregation — reads both
        dataset families identically.
        """
        train_images, train_labels = probe_train
        test_images, test_labels = probe_test

        train_features = self.encode_all(train_images)
        test_features = self.encode_all(test_images)

        record: dict[str, Any] = collapse_metrics(test_features)

        if isinstance(train_labels, dict):
            label_sets = {name: (train_labels[name], test_labels[name]) for name in train_labels}
        else:
            label_sets = {None: (train_labels, test_labels)}

        scaled, unscaled, retrieved = [], [], []
        balanced, floors, macro_f1s, adjusted = [], [], [], []
        for name, (fit_labels, eval_labels) in label_sets.items():
            suffix = f"_{name}" if name is not None else ""
            # Both probes. The standardized one stays comparable across
            # checkpoints; the unstandardized one is the only probe that
            # registers scale collapse, because standardizing rescales a
            # collapsed encoder's numerical noise back to unit variance. A
            # widening gap between them is itself a signal.
            probe = linear_probe_scores(
                train_features, fit_labels, test_features, eval_labels, seed=self.config["seed"]
            )
            probe_raw = linear_probe_scores(
                train_features,
                fit_labels,
                test_features,
                eval_labels,
                seed=self.config["seed"],
                standardize=False,
            )
            precision = retrieval_precision_at_k(test_features, eval_labels, k=10)
            macro_precision, _ = retrieval_macro_precision_at_k(test_features, eval_labels, k=10)
            # Floors travel with every number. On BDD100K's `scene`, always
            # guessing "city street" scores 0.71 and random retrieval scores
            # about 0.60 — an accuracy reported without its floor is unreadable.
            chance = retrieval_chance(eval_labels)

            # Optimisation metadata travels with every probe number, for both
            # variants. Without it a reader cannot tell a genuine chance-level
            # reading from a probe whose optimiser stopped before it moved --
            # the two are indistinguishable in the accuracy alone. See
            # diagnostics/probe.py.
            for variant, scores in (("", probe), ("_unscaled", probe_raw)):
                for key in ("selected_C", "n_iter", "converged", "underfit_train",
                            "train_accuracy", "feature_scale"):
                    record[f"probe_{key}{variant}{suffix}"] = scores[key]

            if name is not None:
                record[f"probe_accuracy{suffix}"] = probe["accuracy"]
                record[f"probe_accuracy_unscaled{suffix}"] = probe_raw["accuracy"]
                record[f"probe_balanced{suffix}"] = probe["balanced_accuracy"]
                record[f"probe_balanced_unscaled{suffix}"] = probe_raw["balanced_accuracy"]
                record[f"probe_majority{suffix}"] = probe["majority"]
                record[f"probe_macro_f1{suffix}"] = probe["macro_f1"]
                record[f"retrieval_p10{suffix}"] = precision
                record[f"retrieval_macro_p10{suffix}"] = macro_precision
                record[f"retrieval_chance{suffix}"] = chance
                # Normalized so 0 is chance and 1 is perfect. Raw P@10 is not
                # comparable across attributes with different class priors, so
                # only the adjusted form may be averaged.
                record[f"retrieval_adjusted{suffix}"] = chance_adjusted(precision, chance)
                record[f"scored_classes{suffix}"] = probe["scored_classes"]
                record[f"dropped_classes{suffix}"] = probe["dropped_classes"]
            scaled.append(probe["accuracy"])
            unscaled.append(probe_raw["accuracy"])
            balanced.append(probe["balanced_accuracy"])
            macro_f1s.append(probe["macro_f1"])
            retrieved.append(precision)
            adjusted.append(chance_adjusted(precision, chance))
            floors.append((probe["majority"], chance))

        record["probe_accuracy"] = float(np.mean(scaled))
        record["probe_accuracy_unscaled"] = float(np.mean(unscaled))
        record["retrieval_p10"] = float(np.mean(retrieved))
        record["probe_balanced"] = float(np.mean(balanced))
        record["probe_macro_f1"] = float(np.mean(macro_f1s))
        # Across-attribute aggregate uses the CHANCE-ADJUSTED retrieval, since
        # raw P@10 from attributes with different priors cannot be averaged
        # meaningfully. Secondary summary; the per-attribute keys are primary.
        record["retrieval_adjusted"] = float(np.mean(adjusted))
        record["probe_majority"] = float(np.mean([f[0] for f in floors]))
        record["retrieval_chance"] = float(np.mean([f[1] for f in floors]))
        # Headroom: how far above its own floor each family sits. On skewed
        # attributes this is the only interpretable version of the number.
        record["probe_over_majority"] = float(np.mean(scaled)) - record["probe_majority"]
        record["retrieval_over_chance"] = float(np.mean(retrieved)) - record["retrieval_chance"]
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

        if self.config["logging"].get("save_artifacts", True):
            self.save_artifacts(logger.run_dir, probe_test, probe_train)

    def save_artifacts(
        self, run_dir: Path, probe_test: tuple, probe_train: tuple | None = None
    ) -> None:
        """Persist the final encoder and its evaluation embeddings.

        Without this a run is answerable only by the diagnostics that happened
        to be implemented on the day it ran: `metrics.jsonl` keeps a 2D PCA of
        500 samples, which is too few dimensions for any rank measure to mean
        anything. A diagnostic thought of later, a different probe, a
        re-evaluation on a different attribute subset, or the controlled
        degradation suite applied to real embeddings rather than synthetic
        clusters — all of those need the representation itself, and retraining
        the matrix to recover it costs far more than the disk.

        Two files, both small: the encoder is a few MB and re-embeds anything
        later; the embeddings are what the degradation suite consumes directly.
        Both are gitignored — results stay reproducible from `metrics.jsonl`,
        which remains the committed record.
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.context_encoder.state_dict(), run_dir / "encoder.pt")

        test_images, test_labels = probe_test
        features = self.encode_all(test_images)
        # Both splits. Refitting a probe offline needs the TRAIN embeddings too,
        # and storing only test forced recovery to go back through encoder.pt
        # and the raw dataset -- which works, but needs the data present.
        train_features = self.encode_all(probe_train[0]) if probe_train is not None else None
        # Labels are a plain array on CIFAR-10 and a dict of named attributes
        # on BDD100K; savez flattens the dict so each attribute lands under
        # its own key and neither dataset family needs special handling later.
        arrays = {"features": features}
        if train_features is not None:
            arrays["train_features"] = train_features
            train_labels = probe_train[1]
            if isinstance(train_labels, dict):
                arrays.update({f"train_labels_{k}": v for k, v in train_labels.items()})
            else:
                arrays["train_labels"] = train_labels
        if isinstance(test_labels, dict):
            arrays.update({f"labels_{name}": value for name, value in test_labels.items()})
        else:
            arrays["labels"] = test_labels
        np.savez_compressed(run_dir / "embeddings.npz", **arrays)


def count_parameters(module: nn.Module) -> int:
    """Trainable parameter count, reported at run start."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
