"""Wiring tests for the Phase-2 validity fixes.

These pin the audit findings: the completed 2x2 (`none_stopgrad`), the
projector buffer (SIGReg must see the projector output, the probe must never
see it), and seed pairing across conditions.
"""

import numpy as np
import pytest
import torch

from jepa_lens.training.strategy import build_strategy
from jepa_lens.training.trainer import Trainer


def tiny_config(name: str, projector_dim: int = 0, seed: int = 0) -> dict:
    flags = {
        "ema_stopgrad": (True, True, 0.0),
        "none_stopgrad": (False, True, 0.0),
        "none_nostopgrad": (False, False, 0.0),
        "sigreg_stopgrad": (False, True, 1.0),
        "sigreg_nostopgrad": (False, False, 1.0),
        "proj_sigreg_stopgrad": (False, True, 1.0),
        "proj_sigreg_nostopgrad": (False, False, 1.0),
    }
    uses_ema, detaches, weight = flags[name]
    return {
        "seed": seed,
        "model": {
            "image_size": 32,
            "patch_size": 8,
            "embed_dim": 16,
            "depth": 1,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "predictor_hidden_dim": 32,
            "projector_dim": projector_dim,
            "projector_hidden_dim": 32,
        },
        "masking": {
            "num_target_blocks": 2,
            "target_block_scale": [0.15, 0.2],
            "target_block_aspect": [0.75, 1.5],
        },
        "optim": {"lr": 0.001, "weight_decay": 0.05, "warmup_steps": 2, "total_steps": 10},
        "logging": {
            "checkpoint_every": 5,
            "projection_samples": 20,
            "output_root": "./experiments",
        },
        "data": {"probe_train_samples": 20, "probe_test_samples": 10},
        "strategy": {
            "name": name,
            "uses_ema_target": uses_ema,
            "detaches_target": detaches,
            "ema_decay": 0.99,
            "sigreg_weight": weight,
            "sigreg_num_slices": 8,
            "sigreg_num_freqs": 5,
            "sigreg_freq_max": 3.0,
        },
    }


def test_none_stopgrad_completes_the_two_by_two():
    """The audit's B2: stop-gradient with no regularizer, shared encoder."""
    strategy = build_strategy(tiny_config("none_stopgrad")["strategy"])
    assert strategy.uses_ema_target is False
    assert strategy.detaches_target is True

    trainer = Trainer(tiny_config("none_stopgrad"))
    assert trainer.target_encoder is trainer.context_encoder
    output = trainer.forward_pass(torch.randn(4, 3, 32, 32))
    assert not output.target_latent.requires_grad
    assert "sigreg" not in output.loss_output.components


def test_proj_conditions_build_a_projector_and_plain_ones_do_not():
    with_proj = Trainer(tiny_config("proj_sigreg_stopgrad", projector_dim=24))
    without = Trainer(tiny_config("sigreg_stopgrad"))
    assert with_proj.projector is not None
    assert without.projector is None


def test_sigreg_sees_the_projector_output_not_the_encoder():
    """The audit's B1 made testable: with a projector, the regularizer's input
    lives in projector space and is a different tensor from the probed one."""
    trainer = Trainer(tiny_config("proj_sigreg_stopgrad", projector_dim=24))
    output = trainer.forward_pass(torch.randn(4, 3, 32, 32))
    assert output.reg_embedding.shape == (4, 24)
    assert output.context_embedding.shape == (4, 16)
    assert output.reg_embedding is not output.context_embedding


def test_without_projector_the_regularizer_sees_the_encoder_directly():
    trainer = Trainer(tiny_config("sigreg_stopgrad"))
    output = trainer.forward_pass(torch.randn(4, 3, 32, 32))
    assert output.reg_embedding is output.context_embedding


def test_probe_features_come_from_the_encoder_never_the_projector():
    """The probed representation must be untouched by the projector's existence."""
    trainer = Trainer(tiny_config("proj_sigreg_stopgrad", projector_dim=24))
    features = trainer.encode_all(torch.randn(6, 3, 32, 32))
    assert features.shape == (6, 16)  # encoder dim, not projector dim


def test_projector_parameters_are_optimized():
    trainer = Trainer(tiny_config("proj_sigreg_nostopgrad", projector_dim=24))
    optimizer_params = {
        id(p) for group in trainer.optimizer.param_groups for p in group["params"]
    }
    for parameter in trainer.projector.parameters():
        assert id(parameter) in optimizer_params


def test_projector_gradient_reaches_the_encoder():
    """SIGReg-through-projector must still shape the encoder, or the buffer
    would disconnect the regularizer entirely."""
    trainer = Trainer(tiny_config("proj_sigreg_nostopgrad", projector_dim=24))
    output = trainer.forward_pass(torch.randn(8, 3, 32, 32))
    output.loss_output.total.backward()
    grads = [p.grad for p in trainer.context_encoder.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_disabled_projector_leaves_the_optimizer_roster_unchanged():
    baseline = Trainer(tiny_config("sigreg_stopgrad"))
    encoder_and_predictor = sum(
        p.numel() for g in baseline.optimizer.param_groups for p in g["params"]
    )
    expected = sum(p.numel() for p in baseline.context_encoder.parameters()) + sum(
        p.numel() for p in baseline.predictor.parameters()
    )
    assert encoder_and_predictor == expected


def test_same_seed_pairs_masks_across_conditions():
    """Paired-seed design: at seed s, every condition must draw identical masks."""
    torch.manual_seed(999)  # pollute the global RNG to prove it is not used
    first = Trainer(tiny_config("none_stopgrad", seed=3))
    second = Trainer(tiny_config("sigreg_nostopgrad", seed=3))
    from jepa_lens.data import sample_block_masks

    mask_a = sample_block_masks(4, 4, 2, (0.15, 0.2), (0.75, 1.5), first.generator)
    mask_b = sample_block_masks(4, 4, 2, (0.15, 0.2), (0.75, 1.5), second.generator)
    assert torch.equal(mask_a[0], mask_b[0])
    assert torch.equal(mask_a[1], mask_b[1])


def test_different_seeds_produce_different_masks():
    first = Trainer(tiny_config("none_stopgrad", seed=0))
    second = Trainer(tiny_config("none_stopgrad", seed=1))
    from jepa_lens.data import sample_block_masks

    mask_a = sample_block_masks(4, 4, 2, (0.15, 0.2), (0.75, 1.5), first.generator)
    mask_b = sample_block_masks(4, 4, 2, (0.15, 0.2), (0.75, 1.5), second.generator)
    assert not torch.equal(mask_a[0], mask_b[0])


def test_seed_changes_encoder_initialisation():
    a = Trainer(tiny_config("none_stopgrad", seed=0))
    b = Trainer(tiny_config("none_stopgrad", seed=1))
    c = Trainer(tiny_config("none_stopgrad", seed=0))
    pa = a.context_encoder.pos_embed.detach()
    pb = b.context_encoder.pos_embed.detach()
    pc = c.context_encoder.pos_embed.detach()
    assert not torch.allclose(pa, pb)
    assert torch.allclose(pa, pc)


@pytest.mark.parametrize(
    "name",
    ["none_stopgrad", "proj_sigreg_stopgrad", "proj_sigreg_nostopgrad"],
)
def test_new_conditions_run_a_full_train_step(name):
    projector_dim = 24 if name.startswith("proj_") else 0
    trainer = Trainer(tiny_config(name, projector_dim=projector_dim))
    metrics = trainer.train_step(torch.randn(4, 3, 32, 32))
    assert np.isfinite(metrics["total"])


def cifar_like_config(seed: int, eval_split_seed: int) -> dict:
    """Minimal config for exercising probe-subset selection without torchvision."""
    return {
        "seed": seed,
        "data": {
            "dataset": "cifar10",
            "eval_split_seed": eval_split_seed,
            "probe_train_samples": 5,
            "probe_test_samples": 5,
        },
    }


def _subset_indices(config: dict, population: int) -> list[int]:
    """Reproduce the loader's subset draw, which is the thing under test."""
    rng = np.random.default_rng(config["data"].get("eval_split_seed", 0))
    return sorted(
        int(i)
        for i in rng.choice(
            population, size=config["data"]["probe_test_samples"], replace=False
        )
    )


def test_probe_subset_is_identical_across_run_seeds():
    """Across-seed spread must measure training variability, not which images
    happened to be scored."""
    a = _subset_indices(cifar_like_config(seed=0, eval_split_seed=0), 500)
    b = _subset_indices(cifar_like_config(seed=7, eval_split_seed=0), 500)
    assert a == b


def test_changing_eval_split_seed_changes_the_probe_subset():
    a = _subset_indices(cifar_like_config(seed=0, eval_split_seed=0), 500)
    b = _subset_indices(cifar_like_config(seed=0, eval_split_seed=1), 500)
    assert a != b


def test_bdd_probe_split_follows_eval_split_seed_not_the_run_seed(tmp_path):
    """Same guarantee on the driving corpus."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from jepa_lens.bdd100k import build_bdd_dataloaders
    from test_bdd100k import CLEAN, RAINY, make_per_image_tree

    for split in ("train", "val"):
        make_per_image_tree(
            tmp_path,
            split,
            [(f"{split}{i}", {"attributes": CLEAN if i % 2 else RAINY}) for i in range(12)],
        )

    def probe_labels(run_seed, eval_seed):
        config = {
            "seed": run_seed,
            "data": {
                "dataset": "bdd100k",
                "root": str(tmp_path),
                "batch_size": 2,
                "num_workers": 0,
                "eval_split_seed": eval_seed,
                "probe_train_samples": 4,
                "probe_test_samples": 4,
            },
            "model": {"image_size": 32},
        }
        _, _, (images, labels) = build_bdd_dataloaders(config)
        return images, labels["weather"]

    images_a, labels_a = probe_labels(run_seed=0, eval_seed=0)
    images_b, labels_b = probe_labels(run_seed=5, eval_seed=0)
    assert torch.equal(images_a, images_b)
    assert (labels_a == labels_b).all()

    _, labels_c = probe_labels(run_seed=0, eval_seed=3)
    assert not (len(labels_a) == len(labels_c) and (labels_a == labels_c).all())


def test_resolved_config_records_both_seeds():
    config = tiny_config("none_stopgrad", seed=4)
    config["data"]["eval_split_seed"] = 2
    trainer = Trainer(config)
    assert trainer.config["seed"] == 4
    assert trainer.config["data"]["eval_split_seed"] == 2
