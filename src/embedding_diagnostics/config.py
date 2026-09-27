"""Config loading: a shared base plus a per-condition override, deep-merged."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` onto `base`, returning a new dict.

    Nested dicts merge key-by-key so an override that sets one key inside a
    block does not discard that block's sibling keys.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    condition: str, configs_dir: Path, base: str = "base.yaml"
) -> dict[str, Any]:
    """Load a base config and deep-merge `conditions/<condition>.yaml` onto it.

    `base` selects the dataset family (`base.yaml` for CIFAR-10, `bdd.yaml` for
    BDD100K); condition files only override the strategy/model blocks, so the
    same seven conditions run unchanged on either dataset.
    """
    base_path = Path(configs_dir) / base
    override_path = Path(configs_dir) / "conditions" / f"{condition}.yaml"
    if not override_path.exists():
        raise FileNotFoundError(f"no config for condition {condition!r}: {override_path}")
    base = yaml.safe_load(base_path.read_text())
    override = yaml.safe_load(override_path.read_text()) or {}
    return deep_merge(base, override)
