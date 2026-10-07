"""FP32 image adapter for Google's EmbeddingGemma 2 (2026 model card).

Official semantics: https://huggingface.co/google/embeddinggemma-2 and
https://huggingface.co/docs/transformers/model_doc/embedding_gemma2 .
Projected token outputs are attention-mask mean pooled, then L2 normalized,
matching the pinned checkpoint's 1_Pooling and 2_Normalize modules.
This is integration infrastructure, not a validation of pretrained research
claims. No modality prefixes are added to image inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
DIMENSION = 768


def validate_embeddings(matrix: np.ndarray, count: int) -> np.ndarray:
    """Require a finite, nonzero FP32 [N, 768] canonical matrix."""
    matrix = np.asarray(matrix)
    if matrix.dtype != np.float32 or matrix.shape != (count, DIMENSION) or count < 1:
        raise ValueError(
            f"expected float32 [{count}, {DIMENSION}], got {matrix.dtype} {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings contain non-finite values")
    if (np.linalg.norm(matrix, axis=1) <= 1e-12).any():
        raise ValueError("embeddings contain zero norms")
    return matrix


def pool_embeddings(hidden: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Checkpoint-defined masked mean pooling followed by L2 normalization."""
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or (mask.sum(1) <= 0).any():
        raise ValueError("invalid token outputs or attention mask")
    hidden = hidden.float().masked_fill(~mask.bool().unsqueeze(-1), 0)
    pooled = hidden.sum(1) / mask.sum(1, keepdim=True)
    # Validate before normalizing so exact zeros cannot be hidden by epsilon.
    validate_embeddings(pooled.detach().cpu().numpy(), len(pooled))
    result = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy()
    return validate_embeddings(result, len(result))


def cuda_preflight() -> dict:
    """Reject unsupported GPU architectures before allocating model weights."""
    if not torch.cuda.is_available():
        return {"usable": False, "reason": "CUDA unavailable"}
    capability = torch.cuda.get_device_capability(0)
    arches = torch.cuda.get_arch_list()
    # Conservatively require a compiled architecture on this major family.
    supported = any(a.startswith(f"sm_{capability[0]}") for a in arches)
    return {"usable": supported, "capability": list(capability), "compiled_arches": arches,
            "name": torch.cuda.get_device_name(0),
            "reason": "supported" if supported else
            "GPU architecture absent from locked Torch build"}


class EmbeddingGemma2:
    """External reference encoder; never a CollapsePreventionStrategy.

    Loading always uses local_files_only, even when source is a model ID.
    A Hub ID must therefore already exist in the selected HF cache.
    """

    def __init__(self, source: str | Path = MODEL_ID, *, revision: str = MODEL_REVISION,
                 device: str = "cpu", dtype: torch.dtype = torch.float32):
        if dtype != torch.float32:
            raise ValueError("Phase C canonical inference requires float32; float16 is forbidden")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("model revision must be an exact 40-character commit")
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if device == "cuda" and not cuda_preflight()["usable"]:
            raise ValueError(f"CUDA preflight failed: {cuda_preflight()}")

        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, EmbeddingGemma2Config, EmbeddingGemma2Model

        source = str(source)
        path = Path(source)
        if not path.is_dir():
            path = Path(snapshot_download(source, revision=revision, local_files_only=True))
        path = path.absolute()
        if path.parent.name == "snapshots" and path.name != revision:
            raise ValueError("local snapshot revision does not match requested revision")
        pooling = json.loads((path / "1_Pooling" / "config.json").read_text())
        if pooling != {"embedding_dimension": 768, "pooling_mode": "mean", "include_prompt": True}:
            raise ValueError("checkpoint pooling configuration differs from supported semantics")
        modules = json.loads((path / "modules.json").read_text())
        if not any(m["type"].endswith("Normalize") for m in modules):
            raise ValueError("checkpoint does not declare normalization")
        config = EmbeddingGemma2Config.from_pretrained(
            path, local_files_only=True, audio_config=None,
        )
        if config.audio_config is not None or config.text_config.embedding_dim != DIMENSION:
            raise ValueError("unexpected model configuration")
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model, loading = EmbeddingGemma2Model.from_pretrained(
            path, config=config, local_files_only=True, dtype=torch.float32,
            attn_implementation="eager", output_loading_info=True,
        )
        if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"incomplete checkpoint loading: {loading}")
        if self.model.audio_tower is not None or self.model.embed_audio is not None:
            raise ValueError("audio encoder was not excluded")
        self.model.eval().to(device)
        if any(p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError("model parameters are not all float32")
        self.device = device
        file_hashes = {}
        for name in ("config.json", "processor_config.json", "preprocessor_config.json",
                     "tokenizer.json", "tokenizer_config.json", "1_Pooling/config.json",
                     "modules.json"):
            file_hashes[name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
        weights_hashes = {}
        for weight_file in sorted(path.glob("*.safetensors")):
            with weight_file.open("rb") as handle:
                weights_hashes[weight_file.name] = hashlib.file_digest(handle, "sha256").hexdigest()
        if not weights_hashes:
            raise ValueError("local checkpoint has no safetensors weights")
        self.provenance = {
            "model_id": MODEL_ID, "model_revision": revision, "model_source": source,
            "revision_verification": "HF snapshot directory" if path.parent.name == "snapshots"
            else "caller-asserted revision; local weight hashes recorded",
            "model_configuration": config.to_dict(),
            "enabled_modalities": ["text", "vision"], "input_modality": "image",
            "processor_configuration": self.processor.to_dict(),
            "processor_class": type(self.processor).__name__,
            "image_processor_class": type(self.processor.image_processor).__name__
            if hasattr(self.processor, "image_processor") else None,
            "source_configuration_sha256": file_hashes,
            "weights_sha256": weights_hashes,
            "dtype": "float32", "device": device, "native_embedding_dimension": DIMENSION,
            "normalization_policy": "attention-mask mean pooling; L2 unit norm",
            "attention_implementation": "eager",
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "audio_excluded": True,
        }

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """One image per sample; return plain CPU FP32 normalized vectors."""
        if not images:
            raise ValueError("image batch cannot be empty")
        inputs = self.processor(images=[[im] for im in images], return_tensors="pt")
        inputs = {k: v.to(device=self.device, dtype=torch.float32) if v.is_floating_point()
                  else v.to(self.device) for k, v in inputs.items()}
        with torch.autocast(device_type=self.device, enabled=False):
            output = self.model(**inputs)
        return pool_embeddings(output.last_hidden_state, inputs["attention_mask"])
