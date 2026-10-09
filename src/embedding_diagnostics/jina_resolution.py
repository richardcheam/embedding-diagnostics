"""Resolution and tensor-inventory checks for pinned Jina-CLIP-v2 author sources.

Koukounas et al., arXiv:2412.08802v2. No numeric model operation is reimplemented.
The nested tokenizer remains its original v3 tokenizer, not the outer CLIP object.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path


def constructor_tokenizer_expression(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "XLMRobertaModel")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assignment = next(
        n
        for n in init.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "tokenizer" for t in n.targets)
    )
    return compile(ast.Expression(assignment.value), "<pinned constructor tokenizer>", "eval")


def tokenizer_identity(tokenizer):
    return {
        "class": type(tokenizer).__name__,
        "backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "special_tokens_map": tokenizer.special_tokens_map,
        "model_max_length": tokenizer.model_max_length,
        "vocabulary_size": len(tokenizer),
    }


def bind_nested_tokenizer(snapshot):
    """Require original v3 files at their local immutable snapshot before construction."""
    snapshot = Path(snapshot)
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    if any(not (snapshot / n).is_file() for n in names):
        raise ValueError("complete intended nested tokenizer files required; no outer fallback")
    cfg = json.loads((snapshot / "tokenizer_config.json").read_text())
    if cfg.get("tokenizer_class") != "XLMRobertaTokenizer" or cfg.get("auto_map"):
        raise ValueError("unexpected nested tokenizer class or remote tokenizer mapping")
    from transformers import XLMRobertaTokenizerFast

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(snapshot, local_files_only=True)
    return tokenizer


def verify_nested_tokenizer(model, expected):
    actual = model.text_model.transformer.roberta.tokenizer
    if tokenizer_identity(actual) != tokenizer_identity(expected):
        raise ValueError("constructor nested tokenizer differs from its intended original source")
    if Path(actual.name_or_path).resolve() != Path(expected.name_or_path).resolve():
        raise ValueError("nested tokenizer original source path mismatch")
    return tokenizer_identity(actual)


def tensor_coverage(model, inventory, *, reported=865278476):
    """Require exact state-key/shape coverage, including the scalar contrastive temperature."""
    expected = {k: list(v.shape) for k, v in model.state_dict().items()}
    missing = sorted(set(expected) - set(inventory))
    unexpected = sorted(set(inventory) - set(expected))
    mismatched = {
        k: {"expected": expected[k], "stored": inventory[k]["shape"]}
        for k in set(expected) & set(inventory)
        if expected[k] != inventory[k]["shape"]
    }
    if missing or unexpected or mismatched:
        raise ValueError(f"tensor coverage mismatch: {missing=} {unexpected=} {mismatched=}")
    stored = sum(math.prod(v["shape"]) for v in inventory.values())
    parameters = sum(p.numel() for p in model.parameters())
    scalars = {k: math.prod(v["shape"]) for k, v in inventory.items() if v["shape"] == []}
    if scalars != {"logit_scale": 1} or stored != reported + 1 or parameters != stored:
        raise ValueError("checkpoint inventory discrepancy not explained by logit_scale scalar")
    return {
        "state_tensors": len(expected),
        "parameters": parameters,
        "stored_elements": stored,
        "reported_elements": reported,
        "scalar_elements": sum(scalars.values()),
        "scalar_keys": scalars,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": mismatched,
        "explanation": "Reported count exactly equals non-scalar tensor elements; "
        "complete checkpoint "
        "also contains the trained scalar logit_scale parameter. All state keys/shapes "
        "and parameters accounted for; no tensor discarded. Hub counting cause unverified.",
    }
