import numpy as np
import pytest
import torch

from jepa_lens.training.trainer import Trainer

CONDITIONS = ["ema_stopgrad", "sigreg_stopgrad", "sigreg_nostopgrad", "none_nostopgrad"]


def tiny_config(name: str) -> dict:
    flags = {
        "ema_stopgrad": (True, True, 0.0),
        "sigreg_stopgrad": (False, True, 1.0),
        "sigreg_nostopgrad": (False, False, 1.0),
        "none_nostopgrad": (False, False, 0.0),
    }
    uses_ema, detaches, weight = flags[name]
    return {
        "seed": 0,
        "model": {
            "image_size": 32,
            "patch_size": 8,
            "embed_dim": 16,
            "depth": 1,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "predictor_hidden_dim": 32,
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
            "sigreg_num_freqs": 4,
            "sigreg_freq_max": 5.0,
        },
    }


@pytest.mark.parametrize("name", CONDITIONS)
def test_train_step_returns_finite_loss(name):
    trainer = Trainer(tiny_config(name))
    metrics = trainer.train_step(torch.randn(4, 3, 32, 32))
    assert np.isfinite(metrics["total"])
    assert "prediction" in metrics


@pytest.mark.parametrize("name", CONDITIONS)
def test_detach_flag_controls_target_gradient(name):
    """The condition table's `detaches_target` column must be honoured."""
    config = tiny_config(name)
    trainer = Trainer(config)
    output = trainer.forward_pass(torch.randn(4, 3, 32, 32))
    expected_detached = config["strategy"]["detaches_target"]
    assert output.target_latent.requires_grad is not expected_detached


def test_ema_conditions_build_a_separate_target_encoder():
    trainer = Trainer(tiny_config("ema_stopgrad"))
    assert trainer.target_encoder is not trainer.context_encoder


def test_sigreg_conditions_share_one_encoder():
    """LeJEPA removes the teacher-student split entirely."""
    trainer = Trainer(tiny_config("sigreg_nostopgrad"))
    assert trainer.target_encoder is trainer.context_encoder


def test_control_condition_target_branch_carries_gradient():
    """The control must actually couple both branches, or it cannot collapse.

    A frozen EMA target would leave the target branch attached to no trainable
    parameter, making `detaches_target=False` inert and silently duplicating
    `ema_stopgrad`. The control therefore shares one encoder.
    """
    trainer = Trainer(tiny_config("none_nostopgrad"))
    assert trainer.target_encoder is trainer.context_encoder
    output = trainer.forward_pass(torch.randn(4, 3, 32, 32))
    assert output.target_latent.requires_grad


def test_ema_target_encoder_is_not_in_the_optimizer():
    trainer = Trainer(tiny_config("ema_stopgrad"))
    optimizer_params = {
        id(param) for group in trainer.optimizer.param_groups for param in group["params"]
    }
    target_params = {id(param) for param in trainer.target_encoder.parameters()}
    assert optimizer_params.isdisjoint(target_params)


def test_ema_target_encoder_changes_after_a_step():
    trainer = Trainer(tiny_config("ema_stopgrad"))
    before = trainer.target_encoder.pos_embed.detach().clone()
    trainer.train_step(torch.randn(4, 3, 32, 32))
    after = trainer.target_encoder.pos_embed.detach().clone()
    assert not torch.allclose(before, after)


def test_encode_all_returns_pooled_embeddings():
    trainer = Trainer(tiny_config("ema_stopgrad"))
    features = trainer.encode_all(torch.randn(6, 3, 32, 32))
    assert features.shape == (6, 16)
    assert isinstance(features, np.ndarray)


def test_evaluate_returns_all_diagnostic_keys():
    trainer = Trainer(tiny_config("ema_stopgrad"))
    rng = np.random.default_rng(0)
    probe_train = (torch.randn(20, 3, 32, 32), rng.integers(0, 3, 20))
    probe_test = (torch.randn(10, 3, 32, 32), rng.integers(0, 3, 10))

    record = trainer.evaluate(probe_train, probe_test)
    for key in ["effective_rank", "mean_feature_std", "probe_accuracy", "projection"]:
        assert key in record
    assert len(record["projection"]) == 10


def test_logged_lr_matches_the_rate_actually_applied():
    """The logged rate must be the one this step used, not the next step's.

    `_learning_rate()` reads `self.step`, so recomputing it after the
    increment silently reports a rate one schedule step ahead of the loss it
    is logged beside. Compared against an independently computed expected
    rate per step (rather than the previous step's logged value) because
    with this config's `warmup_steps=2`, the last warmup step and the first
    cosine step happen to land on the same rate (0.001) by construction of
    a continuous schedule, so a consecutive-steps-differ assumption is not
    reliably true here.
    """
    trainer = Trainer(tiny_config("ema_stopgrad"))
    for _ in range(4):
        expected_rate = trainer._learning_rate()
        metrics = trainer.train_step(torch.randn(4, 3, 32, 32))
        applied_during = trainer.optimizer.param_groups[0]["lr"]
        assert metrics["lr"] == pytest.approx(expected_rate)
        assert applied_during == pytest.approx(expected_rate)
