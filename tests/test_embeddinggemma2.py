"""Model-free checks of dtype, masked pooling, and output validation."""

import numpy as np
import pytest
import torch

from embedding_diagnostics.models.embeddinggemma2 import (
    EmbeddingGemma2,
    pool_embeddings,
    validate_embeddings,
)


def test_float16_rejected_before_loading():
    with pytest.raises(ValueError, match="float32"):
        EmbeddingGemma2("unused", revision="a" * 40, dtype=torch.float16)


def test_unpinned_revision_rejected_before_loading():
    with pytest.raises(ValueError, match="revision"):
        EmbeddingGemma2("unused", revision="main")


def test_masked_mean_pooling_and_normalization():
    hidden = torch.zeros(2, 3, 768)
    hidden[0, 0, :2] = torch.tensor([2., 0.])
    hidden[0, 1, :2] = torch.tensor([0., 2.])
    hidden[0, 2] = float("nan")  # masked padding must not contaminate the mean
    hidden[1, :, 0] = 3
    result = pool_embeddings(hidden, torch.tensor([[1, 1, 0], [1, 1, 1]]))
    np.testing.assert_allclose(result[0, :2], [2**-0.5, 2**-0.5], rtol=1e-6)
    assert result[1, 0] == 1
    assert result.dtype == np.float32 and result.shape == (2, 768)


@pytest.mark.parametrize("matrix", [
    np.ones((2, 512), dtype=np.float32),
    np.full((2, 768), np.nan, dtype=np.float32),
    np.full((2, 768), np.inf, dtype=np.float32),
    np.zeros((2, 768), dtype=np.float32),
    np.ones((2, 768), dtype=np.float16),
])
def test_invalid_embeddings_rejected(matrix):
    with pytest.raises(ValueError):
        validate_embeddings(matrix, 2)


def test_empty_attention_mask_rejected():
    with pytest.raises(ValueError, match="mask"):
        pool_embeddings(torch.ones(1, 3, 768), torch.zeros(1, 3))


def test_adapter_load_and_inference_contract(tmp_path, monkeypatch):
    """Mock only heavyweight weights/processor; run the real adapter."""
    import json
    from types import SimpleNamespace

    import transformers
    from PIL import Image

    from embedding_diagnostics.models.embeddinggemma2 import MODEL_REVISION

    (tmp_path / "1_Pooling").mkdir()
    files = {
        "config.json": {"model_type": "embedding_gemma2", "text_config": {}},
        "1_Pooling/config.json": {"embedding_dimension": 768, "pooling_mode": "mean",
                                  "include_prompt": True},
        "modules.json": [
            {"type": "sentence_transformers.base.modules.transformer.Transformer"},
            {"type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling"},
            {"type": "sentence_transformers.base.modules.normalize.Normalize"},
        ],
        "processor_config.json": {}, "preprocessor_config.json": {},
        "tokenizer.json": {}, "tokenizer_config.json": {},
    }
    for name, payload in files.items():
        (tmp_path / name).write_text(json.dumps(payload))
    (tmp_path / "model.safetensors").write_bytes(b"fixture weights")

    class Processor:
        def to_dict(self):
            return {"image_seq_length": 280}

        def __call__(self, *, images, return_tensors):
            assert len(images) == 2 and all(len(im) == 1 for im in images)
            assert return_tensors == "pt"
            return {"attention_mask": torch.tensor([[1, 0], [1, 1]]),
                    "pixel_values": torch.ones(2, 2, 3, dtype=torch.float64)}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.audio_tower = None
            self.embed_audio = None

        def forward(self, attention_mask, pixel_values):
            assert not self.training and not torch.is_grad_enabled()
            assert pixel_values.dtype == torch.float32
            hidden = torch.zeros(2, 2, 768)
            hidden[:, :, 0] = 3
            hidden[0, 1] = 99
            return SimpleNamespace(last_hidden_state=hidden)

    def config_loader(path, *, local_files_only, audio_config):
        assert local_files_only and audio_config is None
        return SimpleNamespace(audio_config=None, text_config=SimpleNamespace(embedding_dim=768),
                               to_dict=lambda: {"audio_config": None})

    def model_loader(path, *, config, local_files_only, dtype, attn_implementation,
                     output_loading_info):
        assert config.audio_config is None and local_files_only and output_loading_info
        assert dtype == torch.float32 and attn_implementation == "eager"
        return Model(), {"missing_keys": [], "mismatched_keys": [], "error_msgs": []}

    def processor_loader(path, *, local_files_only):
        assert local_files_only
        return Processor()

    monkeypatch.setattr(transformers.EmbeddingGemma2Config, "from_pretrained", config_loader)
    monkeypatch.setattr(transformers.EmbeddingGemma2Model, "from_pretrained", model_loader)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", processor_loader)
    adapter = EmbeddingGemma2(tmp_path, revision=MODEL_REVISION)
    result = adapter.encode([Image.new("RGB", (8, 8))] * 2)
    assert result.shape == (2, 768) and result.dtype == np.float32
    assert result[:, 0].tolist() == [1., 1.]
    assert np.count_nonzero(result) == 2
    assert adapter.provenance["model_revision"] == MODEL_REVISION
    assert adapter.provenance["audio_excluded"]
    import hashlib

    assert adapter.provenance["weights_sha256"] == {
        "model.safetensors": hashlib.sha256(b"fixture weights").hexdigest(),
    }
