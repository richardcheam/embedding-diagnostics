import json
from types import SimpleNamespace

import pytest


def test_original_constructor_call_attempts_unpinned_config_resolution(tmp_path):
    from embedding_diagnostics.jina_resolution import constructor_tokenizer_expression

    source = """class XLMRobertaModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name_or_path, trust_remote_code=True
        )
    def forward(self, input_ids): return input_ids
"""
    calls = []

    class Tokens:
        @staticmethod
        def from_pretrained(path, **kw):
            calls.append((path, kw))
            if not (tmp_path / "tokenizer_config.json").exists():
                raise OSError("unpinned external config-code lookup")
            return "intended nested tokenizer"

    expression = constructor_tokenizer_expression(source)
    with pytest.raises(OSError, match="unpinned"):
        eval(
            expression,
            {"AutoTokenizer": Tokens, "self": SimpleNamespace(name_or_path=str(tmp_path))},
        )
    assert calls == [(str(tmp_path), {"trust_remote_code": True})]
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "XLMRobertaTokenizer"})
    )
    assert (
        eval(
            expression,
            {"AutoTokenizer": Tokens, "self": SimpleNamespace(name_or_path=str(tmp_path))},
        )
        == "intended nested tokenizer"
    )


def test_missing_nested_files_rejected_without_outer_fallback(tmp_path):
    from embedding_diagnostics.jina_resolution import bind_nested_tokenizer

    with pytest.raises(ValueError, match="nested tokenizer"):
        bind_nested_tokenizer(tmp_path)


def test_complete_tensor_coverage_includes_scalar_and_rejects_discrepancies():
    import torch

    from embedding_diagnostics.jina_resolution import tensor_coverage

    model = torch.nn.Module()
    model.register_parameter("logit_scale", torch.nn.Parameter(torch.empty((), device="meta")))
    model.register_parameter("weight", torch.nn.Parameter(torch.empty(2, 3, device="meta")))
    inventory = {
        "logit_scale": {"shape": [], "dtype": "F16"},
        "weight": {"shape": [2, 3], "dtype": "F16"},
    }
    audit = tensor_coverage(model, inventory, reported=6)
    assert audit["parameters"] == audit["stored_elements"] == 7
    assert audit["scalar_elements"] == 1 and audit["missing_keys"] == audit["unexpected_keys"] == []
    with pytest.raises(ValueError, match="coverage"):
        tensor_coverage(model, {"weight": inventory["weight"]}, reported=6)
    with pytest.raises(ValueError, match="inventory"):
        tensor_coverage(model, inventory, reported=5)


def test_real_local_tokenizer_resolution_bypasses_external_config(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import Unigram
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoConfig

    from embedding_diagnostics.jina_resolution import bind_nested_tokenizer

    backend = Tokenizer(
        Unigram(
            [(w, -1.0) for w in ("<s>", "<pad>", "</s>", "<unk>", "caption", "▁caption")], unk_id=3
        )
    )
    backend.pre_tokenizer = Whitespace()
    backend.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "xlm-roberta", "auto_map": {"AutoConfig": "external-main--Config"}}
        )
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "XLMRobertaTokenizer", "model_max_length": 8194})
    )
    (tmp_path / "special_tokens_map.json").write_text(
        json.dumps(
            {"bos_token": "<s>", "pad_token": "<pad>", "eos_token": "</s>", "unk_token": "<unk>"}
        )
    )

    def forbidden(*a, **kw):
        raise AssertionError("external config lookup")

    monkeypatch.setattr(AutoConfig, "from_pretrained", forbidden)
    tokenizer = bind_nested_tokenizer(tmp_path)
    assert tokenizer.get_vocab()["caption"] == 4
    assert tokenizer("caption")["input_ids"]
    assert tokenizer.model_max_length == 8194


def test_unexpected_and_shape_mismatched_checkpoint_keys_fail():
    import torch

    from embedding_diagnostics.jina_resolution import tensor_coverage

    model = torch.nn.Linear(2, 2, device="meta")
    with pytest.raises(ValueError, match="coverage"):
        tensor_coverage(model, {"extra": {"shape": [1]}})
    with pytest.raises(ValueError, match="coverage"):
        tensor_coverage(model, {"weight": {"shape": [3, 2]}, "bias": {"shape": [2]}})
