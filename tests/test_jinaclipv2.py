from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from embedding_diagnostics.models.jinaclipv2 import JinaCLIPv2, verify_settings


def config():
    return SimpleNamespace(
        use_text_flash_attn=False,
        use_vision_xformers=False,
        vision_config=SimpleNamespace(x_attention=False, image_size=512),
        text_config=SimpleNamespace(
            default_instruction_task=None,
            default_lora_task="retrieval.query",
            hf_model_config_kwargs={"use_flash_attn": False},
        ),
    )


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.config = config()
        self.text_model = SimpleNamespace(
            default_loraid=0, default_instruction=None, lora_adaptation_map={"retrieval.query": 0}
        )
        self.calls = []

    def encode_text(self, captions, **kw):
        self.calls.append((list(captions), kw))
        return torch.ones((len(captions), kw.get("truncate_dim") or 1024)) / np.sqrt(
            kw.get("truncate_dim") or 1024
        )

    def encode_image(self, images, **kw):
        self.calls.append((len(images), kw))
        return torch.ones((len(images), kw.get("truncate_dim") or 1024)) / np.sqrt(
            kw.get("truncate_dim") or 1024
        )


def adapter():
    a = JinaCLIPv2.__new__(JinaCLIPv2)
    a.model = FakeModel()
    a.tokenizer = lambda texts, **kw: {"input_ids": torch.tensor([[0, 12, 2]] * len(texts))}
    return a


def test_plain_caption_contract_preserves_default_adapter():
    a = adapter()
    x = a.encode_captions(["caption two", "caption one"])
    assert x.shape == (2, 1024) and x.dtype == np.float32
    texts, kw = a.model.calls[0]
    assert texts == ["caption two", "caption one"] and kw["task"] is None
    assert kw["truncation"] is False and kw["batch_size"] == 1
    assert kw["device"] == "cpu" and kw["normalize_embeddings"] is True
    verify_settings(a.model)


def test_images_and_prefix_contract():
    a = adapter()
    image = Image.new("RGB", (32, 40))
    assert a.encode_images([image]).shape == (1, 1024)
    assert a.encode_images([image], dimension=128).shape == (1, 128)
    assert a.model.calls[-1][1]["truncate_dim"] == 128
    with pytest.raises(ValueError):
        a.encode_images([image], dimension=512)


@pytest.mark.parametrize("failure", ["dtype", "device_flags", "adapter", "instruction"])
def test_actual_numerical_settings_reject_bad_values(failure):
    model = FakeModel()
    if failure == "dtype":
        model.half()
    if failure == "device_flags":
        model.config.use_text_flash_attn = True
    if failure == "adapter":
        model.text_model.default_loraid = None
    if failure == "instruction":
        model.text_model.default_instruction = "a prefix"
    with pytest.raises(ValueError):
        verify_settings(model)


def test_empty_and_over_context_fail_before_forward():
    a = adapter()
    with pytest.raises(ValueError):
        a.encode_captions([""])
    a.tokenizer = lambda texts, **kw: {"input_ids": torch.zeros((1, 8195), dtype=torch.long)}
    with pytest.raises(ValueError):
        a.encode_captions(["too long"])
    assert not a.model.calls


def test_local_author_import_without_model_construction(tmp_path, monkeypatch):
    import sys

    from embedding_diagnostics.models.jinaclipv2 import author_classes

    (tmp_path / "__init__.py").write_text("")
    for name, cls in [
        ("configuration_clip", "JinaCLIPConfig"),
        ("modeling_clip", "JinaCLIPModel"),
        ("processing_clip", "JinaCLIPImageProcessor"),
    ]:
        (tmp_path / (name + ".py")).write_text(f"class {cls}: pass\n")
    for name in list(sys.modules):
        if name.startswith("_jina_pinned_author"):
            monkeypatch.delitem(sys.modules, name)
    try:
        classes = author_classes(tmp_path)
        assert [c.__name__ for c in classes] == [
            "JinaCLIPConfig",
            "JinaCLIPModel",
            "JinaCLIPImageProcessor",
        ]
    finally:
        for name in list(sys.modules):
            if name.startswith("_jina_pinned_author"):
                del sys.modules[name]


def test_model_construction_requires_explicit_offline_before_source_import(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    with pytest.raises(ValueError, match="offline"):
        JinaCLIPv2(tmp_path, {})


def test_executed_dynamic_source_must_match_pinned_bytes(tmp_path, monkeypatch):
    import sys

    from embedding_diagnostics.models.jinaclipv2 import executed_source_hashes

    original = tmp_path / "original"
    original.mkdir()
    resolved = tmp_path / "resolved"
    resolved.mkdir()
    (original / "modeling.py").write_text("# pinned numerical source\n")
    copy = tmp_path / "modeling.py"
    copy.write_bytes((original / "modeling.py").read_bytes())
    monkeypatch.setitem(
        sys.modules,
        "transformers_modules.jinaai.test.modeling",
        SimpleNamespace(__file__=str(copy)),
    )
    roots = {"resolved_clip": resolved, "jinaai/xlm-roberta-flash-implementation": original}
    assert str(copy) in executed_source_hashes(roots)
    copy.write_text("# modified\n")
    with pytest.raises(ValueError, match="source mismatch"):
        executed_source_hashes(roots)


def test_constructor_enforces_fp32_full_wrapper_and_checkpoint_coverage(tmp_path, monkeypatch):
    import embedding_diagnostics.models.jinaclipv2 as module

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    cfg = config()
    cfg.to_json_string = lambda: "{}"
    c = SimpleNamespace(from_pretrained=lambda *a, **k: cfg)
    calls = []
    model = FakeModel()
    model.text_model.pooler = type("MeanPooler", (), {})()
    model.text_model.proj = torch.nn.Identity()
    model.vision_model = SimpleNamespace(fc_norm=None, head=torch.nn.Identity())

    def load(*a, **kw):
        calls.append(kw)
        return model, {"missing_keys": []}

    mc = SimpleNamespace(from_pretrained=load)
    monkeypatch.setattr(module, "author_classes", lambda *a: (c, mc, None))
    monkeypatch.setattr(
        module,
        "processor_tokenizer",
        lambda *a: (
            lambda *a, **k: {},
            SimpleNamespace(
                size=512, resize_mode="shortest", interpolation="bicubic", to_dict=lambda: {}
            ),
        ),
    )
    monkeypatch.setattr(module, "executed_source_hashes", lambda *a: {})
    roots = {"resolved_clip": tmp_path, "jinaai/jina-embeddings-v3": tmp_path}
    a = JinaCLIPv2(tmp_path, roots)
    assert calls[0]["torch_dtype"] == torch.float32
    assert calls[0]["low_cpu_mem_usage"] and calls[0]["local_files_only"]
    assert calls[0]["use_safetensors"] and calls[0]["output_loading_info"]
    assert a.provenance["default_loraid"] == 0 and a.provenance["text_task"] is None
    mc.from_pretrained = lambda *a, **kw: (model, {"missing_keys": ["weight"]})
    with pytest.raises(ValueError, match="coverage"):
        JinaCLIPv2(tmp_path, roots)
