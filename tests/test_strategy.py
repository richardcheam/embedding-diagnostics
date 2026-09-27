import pytest
import torch
from torch import nn

from embedding_diagnostics.training.strategy import build_strategy

CONDITIONS = ["ema_stopgrad", "sigreg_stopgrad", "sigreg_nostopgrad", "none_nostopgrad"]


def make_config(name: str) -> dict:
    flags = {
        "ema_stopgrad": (True, True, 0.0),
        "sigreg_stopgrad": (False, True, 1.0),
        "sigreg_nostopgrad": (False, False, 1.0),
        "none_nostopgrad": (False, False, 0.0),
    }
    uses_ema, detaches, weight = flags[name]
    return {
        "name": name,
        "uses_ema_target": uses_ema,
        "detaches_target": detaches,
        "ema_decay": 0.99,
        "sigreg_weight": weight,
        "sigreg_num_slices": 8,
        "sigreg_num_freqs": 5,
        "sigreg_freq_max": 3.0,
    }


@pytest.mark.parametrize("name", CONDITIONS)
def test_all_conditions_produce_finite_loss(name):
    strategy = build_strategy(make_config(name))
    output = strategy.compute_loss(
        prediction=torch.randn(8, 5, 16),
        target_latent=torch.randn(8, 5, 16),
        reg_embedding=torch.randn(8, 16),
    )
    assert torch.isfinite(output.total)
    assert "prediction" in output.components


@pytest.mark.parametrize("name", CONDITIONS)
def test_flags_match_condition_table(name):
    config = make_config(name)
    strategy = build_strategy(config)
    assert strategy.uses_ema_target is config["uses_ema_target"]
    assert strategy.detaches_target is config["detaches_target"]


def test_sigreg_conditions_add_a_regularizer_component():
    strategy = build_strategy(make_config("sigreg_stopgrad"))
    output = strategy.compute_loss(
        torch.randn(8, 5, 16), torch.randn(8, 5, 16), torch.randn(8, 16)
    )
    assert "sigreg" in output.components
    assert output.components["sigreg"] > 0.0


def test_ema_conditions_have_no_regularizer_component():
    strategy = build_strategy(make_config("ema_stopgrad"))
    output = strategy.compute_loss(
        torch.randn(8, 5, 16), torch.randn(8, 5, 16), torch.randn(8, 16)
    )
    assert "sigreg" not in output.components


def test_sigreg_components_reconcile_with_total_at_non_unit_weight():
    """Components must break down the total they claim to describe.

    Weights of 0.0 and 1.0 hide weighting bugs because weighted and unweighted
    values coincide there, so this uses 3.0 deliberately.
    """
    config = make_config("sigreg_stopgrad")
    config["sigreg_weight"] = 3.0
    strategy = build_strategy(config)
    output = strategy.compute_loss(
        torch.randn(32, 5, 16), torch.randn(32, 5, 16), torch.randn(32, 16)
    )
    reconstructed = output.components["prediction"] + output.components["sigreg_weighted"]
    assert abs(reconstructed - float(output.total)) < 1e-5
    # The raw penalty is retained as its own diagnostic and differs from the
    # weighted contribution at this weight.
    assert output.components["sigreg"] != output.components["sigreg_weighted"]


def test_sigreg_gradient_reaches_context_embedding():
    """SIGReg must regularize a branch that carries gradient, or it is a no-op."""
    strategy = build_strategy(make_config("sigreg_stopgrad"))
    context_embedding = torch.randn(32, 16, requires_grad=True)
    output = strategy.compute_loss(
        torch.randn(32, 5, 16), torch.randn(32, 5, 16), context_embedding
    )
    output.total.backward()
    assert context_embedding.grad is not None
    assert context_embedding.grad.abs().sum() > 0


def test_ema_update_moves_target_toward_context():
    strategy = build_strategy(make_config("ema_stopgrad"))
    context = nn.Linear(4, 4)
    target = nn.Linear(4, 4)
    with torch.no_grad():
        context.weight.fill_(1.0)
        target.weight.fill_(0.0)

    strategy.post_step_update(context, target)
    assert 0.0 < float(target.weight.detach().mean()) < 1.0


def test_non_ema_strategy_update_is_a_noop():
    strategy = build_strategy(make_config("sigreg_stopgrad"))
    context = nn.Linear(4, 4)
    target = nn.Linear(4, 4)
    with torch.no_grad():
        target.weight.fill_(0.0)
    strategy.post_step_update(context, target)
    assert float(target.weight.detach().abs().sum()) == 0.0


def test_unknown_strategy_name_raises():
    with pytest.raises(ValueError, match="unknown strategy"):
        build_strategy({"name": "nonsense", "uses_ema_target": False, "detaches_target": True})


def test_sigreg_does_not_consume_the_global_rng():
    """Slice directions must come from a dedicated generator.

    Consuming the global RNG here would advance it once per step in the SIGReg
    conditions only. The DataLoader reseeds its shuffle and worker augmentation
    from the global RNG each epoch, so data order would diverge between
    conditions -- a plumbing difference masquerading as a mechanism difference.

    Inputs are built BEFORE seeding so only `compute_loss` runs between the
    seed and the probe draw; otherwise the test's own tensor construction would
    consume the global RNG and mask the effect.
    """
    strategy = build_strategy(make_config("sigreg_stopgrad"))
    prediction = torch.randn(8, 5, 16)
    target = torch.randn(8, 5, 16)
    context = torch.randn(8, 16)

    torch.manual_seed(1234)
    baseline = torch.randn(3).tolist()

    torch.manual_seed(1234)
    strategy.compute_loss(prediction, target, context)
    after_loss = torch.randn(3).tolist()

    assert baseline == after_loss


def test_sigreg_slice_directions_advance_between_calls():
    """Successive steps must not reuse identical slice directions.

    The dedicated generator is stateful, so reusing it across calls advances
    it. If it did not, every training step would probe the embedding
    distribution along the same fixed directions.
    """
    strategy = build_strategy(make_config("sigreg_stopgrad"))
    prediction = torch.randn(8, 5, 16)
    target = torch.randn(8, 5, 16)
    context = torch.randn(8, 16)

    first = strategy.compute_loss(prediction, target, context).components["sigreg"]
    second = strategy.compute_loss(prediction, target, context).components["sigreg"]
    assert first != second
