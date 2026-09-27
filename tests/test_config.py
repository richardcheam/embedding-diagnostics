from pathlib import Path

import pytest

from embedding_diagnostics.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_base_values_survive_merge():
    config = load_config("ema_stopgrad", CONFIGS)
    assert config["data"]["dataset"] == "cifar10"
    assert config["model"]["patch_size"] == 4


def test_condition_overrides_strategy_block():
    config = load_config("sigreg_nostopgrad", CONFIGS)
    assert config["strategy"]["name"] == "sigreg_nostopgrad"
    assert config["strategy"]["uses_ema_target"] is False
    assert config["strategy"]["detaches_target"] is False


def test_merge_is_deep_not_shallow():
    """Overriding one strategy key must not delete sibling keys."""
    config = load_config("none_nostopgrad", CONFIGS)
    assert config["strategy"]["uses_ema_target"] is False
    assert config["strategy"]["detaches_target"] is False
    assert "ema_decay" in config["strategy"]


def test_all_four_conditions_load():
    names = ["ema_stopgrad", "sigreg_stopgrad", "sigreg_nostopgrad", "none_nostopgrad"]
    for name in names:
        config = load_config(name, CONFIGS)
        assert config["strategy"]["name"] == name


def test_unknown_condition_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist", CONFIGS)
