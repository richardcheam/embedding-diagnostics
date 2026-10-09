"""Author-API Jina-CLIP-v2 adapter, Koukounas et al. arXiv:2412.08802v2.

Uses mean text/CLS image pooling and the trained default LoRA without instruction
prefixes. Author code is pinned; numerical integration is validated only by the
separately frozen contract smoke, not by these model-free unit tests.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from embedding_diagnostics.jina_compression import validate_jina
from embedding_diagnostics.jina_resolution import (
    bind_nested_tokenizer,
    tensor_coverage,
    verify_nested_tokenizer,
)

TEXT_CODE_REVISION = "bd55a5ec8e6c0fb1d6c26efb4b6a4a74ce8a88d3"
TEXT_CONFIG_REVISION = "ab036b023d30b4d1138c4c3bfa9f0c445ab455d6"


def author_classes(code_root):
    """Import reviewed local wrapper package; no model construction or forward."""
    root = Path(code_root)
    name = "_jina_pinned_author"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    if Path(sys.modules[name].__file__).resolve() != (root / "__init__.py").resolve():
        raise ValueError("author source root changed within this process")
    cfg = importlib.import_module(name + ".configuration_clip")
    model = importlib.import_module(name + ".modeling_clip")
    proc = importlib.import_module(name + ".processing_clip")
    return cfg.JinaCLIPConfig, model.JinaCLIPModel, proc.JinaCLIPImageProcessor


def executed_source_hashes(code_roots):
    """Verify imported author/dynamic Python bytes against immutable reviewed sources."""
    allowed = {}
    for key in ("resolved_clip", "jinaai/xlm-roberta-flash-implementation"):
        for p in Path(code_roots[key]).glob("*.py"):
            allowed[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    records = {}
    for name, module in list(sys.modules.items()):
        if not (
            name.startswith("_jina_pinned_author")
            or name.startswith("transformers_modules.jinaai.")
        ):
            continue
        path = getattr(module, "__file__", None)
        if not path or not Path(path).is_file():
            continue
        path = Path(path)
        if path.name == "__init__.py" and name.startswith("transformers_modules."):
            if path.read_bytes().strip():
                raise ValueError("unexpected dynamic package initializer")
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if allowed.get(path.name) != sha:
            raise ValueError(f"executed author source mismatch: {path}")
        records[str(path)] = sha
    return records


def processor_tokenizer(snapshot, code_root):
    from transformers import AutoTokenizer

    _, _, processor = author_classes(code_root)
    tok = AutoTokenizer.from_pretrained(snapshot, use_fast=True, local_files_only=True)
    proc = processor.from_pretrained(snapshot, local_files_only=True)
    return tok, proc


def verify_settings(model):
    config = model.config
    if (
        config.use_text_flash_attn
        or config.use_vision_xformers
        or config.vision_config.x_attention
        or config.vision_config.image_size != 512
        or config.text_config.hf_model_config_kwargs["use_flash_attn"]
    ):
        raise ValueError("CPU native attention and official512 processor required")
    if (
        model.text_model.default_loraid != 0
        or model.text_model.default_instruction is not None
        or model.text_model.lora_adaptation_map != {"retrieval.query": 0}
    ):
        raise ValueError("checkpoint trained default adapter without instruction required")
    params = list(model.parameters())
    if not params or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in params):
        raise ValueError("all parameters must be CPU float32")


def author_config(snapshot, code_roots, config_class):
    config = config_class.from_pretrained(snapshot, local_files_only=True)
    config.use_text_flash_attn = False
    config.use_vision_xformers = False
    config.text_config.hf_model_name_or_path = str(code_roots["jinaai/jina-embeddings-v3"])
    config.text_config.jina_text_config_revision = TEXT_CONFIG_REVISION
    config.text_config.jina_text_code_revision = TEXT_CODE_REVISION
    config.text_config.hf_model_config_kwargs["use_flash_attn"] = False
    config.vision_config.x_attention = False
    return config


class JinaCLIPv2:
    def __init__(self, snapshot: Path, code_roots: dict[str, Path]):
        if any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
            raise ValueError("explicit offline nested resolution required before model imports")
        intended_nested = bind_nested_tokenizer(code_roots["jinaai/jina-embeddings-v3"])
        config_class, model_class, _ = author_classes(code_roots["resolved_clip"])
        config = author_config(snapshot, code_roots, config_class)
        self.model, info = model_class.from_pretrained(
            snapshot,
            config=config,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True,
            use_safetensors=True,
            output_loading_info=True,
        )
        if any(
            info.get(k)
            for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        ):
            raise ValueError(f"checkpoint loading coverage failure: {info}")
        inventory_path = Path(snapshot) / "model.safetensors"
        from safetensors import safe_open

        with safe_open(inventory_path, framework="pt", device="cpu") as stored:
            inventory = {
                k: {
                    "shape": list(stored.get_slice(k).get_shape()),
                    "dtype": stored.get_slice(k).get_dtype(),
                }
                for k in stored.keys()
            }
        coverage = tensor_coverage(self.model, inventory)
        nested_identity = verify_nested_tokenizer(self.model, intended_nested)
        self.model.eval()
        if type(self.model.text_model.pooler).__name__ != "MeanPooler":
            raise ValueError("official pad-masked mean text pooler required")
        if not isinstance(self.model.text_model.proj, torch.nn.Identity):
            raise ValueError("checkpoint identity text projection required")
        if self.model.vision_model.fc_norm is not None or not isinstance(
            self.model.vision_model.head, torch.nn.Identity
        ):
            raise ValueError("checkpoint CLS vision pooling and identity head required")
        executed = executed_source_hashes(code_roots)
        self.tokenizer, self.processor = processor_tokenizer(snapshot, code_roots["resolved_clip"])
        self.model.tokenizer = self.tokenizer
        self.model.preprocess = self.processor
        if (
            self.processor.size != 512
            or self.processor.resize_mode != "shortest"
            or self.processor.interpolation != "bicubic"
        ):
            raise ValueError("official 512px shortest-side bicubic processor required")
        verify_settings(self.model)
        self.activation_dtypes = set()

        def hook(_, __, output):
            def walk(value):
                if torch.is_tensor(value) and value.is_floating_point():
                    self.activation_dtypes.add(str(value.dtype))
                elif isinstance(value, dict):
                    for v in value.values():
                        walk(v)
                elif isinstance(value, (list, tuple)):
                    for v in value:
                        walk(v)

            walk(output)

        self.hooks = [m.register_forward_hook(hook) for m in self.model.modules()]
        self.provenance = {
            "model": "jinaai/jina-clip-v2",
            "device": "cpu",
            "dtype": "float32",
            "executed_source_sha256": executed,
            "tensor_coverage": coverage,
            "nested_tokenizer_identity": nested_identity,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "load_mode": "official full wrapper; low_cpu_mem_usage safetensors",
            "loading_info": info,
            "configuration": json.loads(config.to_json_string()),
            "processor": self.processor.to_dict(),
            "text_task": None,
            "default_loraid": 0,
            "normalization": "official native L2; prefixes derived with FP64 L2",
            "pooling": {"image": "CLS", "caption": "pad-masked mean including specials"},
        }

    def _options(self, dimension):
        if dimension not in (1024, 256, 128):
            raise ValueError("unsupported frozen dimension")
        return {
            "batch_size": 1,
            "device": "cpu",
            "truncate_dim": None if dimension == 1024 else dimension,
            "normalize_embeddings": True,
            "convert_to_tensor": True,
            "show_progress_bar": False,
        }

    def _result(self, value, count, dimension):
        if not torch.is_tensor(value) or value.dtype != torch.float32 or value.device.type != "cpu":
            raise ValueError("native inference output must be CPU float32")
        x = value.detach().numpy()
        if dimension == 1024:
            return validate_jina(x, count)
        if (
            x.shape != (count, dimension)
            or not np.isfinite(x).all()
            or not np.allclose(np.linalg.norm(x, axis=1), 1, rtol=1e-5, atol=1e-6)
        ):
            raise ValueError("invalid official prefix output")
        return x

    def encode_images(self, images, *, dimension=1024):
        options = self._options(dimension)
        with torch.inference_mode(), torch.autocast("cpu", enabled=False):
            x = self.model.encode_image(images, **options)
        return self._result(x, len(images), dimension)

    def encode_captions(self, captions, *, dimension=1024):
        if not captions or any(not isinstance(s, str) or not s.strip() for s in captions):
            raise ValueError("nonempty plain captions required")
        tokens = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=False)
        if tokens["input_ids"].shape[1] > 8194:
            raise ValueError("caption exceeds frozen context")
        options = self._options(dimension)
        with torch.inference_mode(), torch.autocast("cpu", enabled=False):
            x = self.model.encode_text(
                captions, task=None, max_length=8194, truncation=False, **options
            )
        return self._result(x, len(captions), dimension)
