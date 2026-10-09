import json

import numpy as np
import pytest

from embedding_diagnostics.jina_protocol import (
    logical_views,
    resource_gate,
    validate_inputs,
    verify_hashes,
)


def inputs():
    return {
        "image": [{"id": 10, "parent_image_id": 10, "role": "image", "input_sha256": "a" * 64}],
        "caption": [
            {
                "id": i,
                "parent_image_id": 10,
                "role": "caption",
                "rendered": "same caption",
                "input_sha256": "a" * 64,
                "token_ids": [0, 3, 2],
            }
            for i in range(5)
        ],
    }


def test_shared_role_identity_and_exact_relevance():
    x = np.ones((1, 1024), dtype=np.float32) / 32
    c = np.ones((5, 1024), dtype=np.float32) / 32
    matrices, rows = logical_views(x, c, inputs())
    assert matrices["query"] is matrices["document"]
    assert rows["query"] is rows["document"]
    assert [r["parent_image_id"] for r in rows["query"]] == [10] * 5
    bad = inputs()
    bad["caption"][0]["parent_image_id"] = 99
    with pytest.raises(ValueError):
        logical_views(x, c, bad)


def test_exact_five_caption_and_order_checks():
    validate_inputs(inputs())
    bad = inputs()
    bad["caption"].pop()
    with pytest.raises(ValueError):
        validate_inputs(bad)
    bad = inputs()
    bad["caption"][1]["id"] = 0
    with pytest.raises(ValueError):
        validate_inputs(bad)


def test_hash_guard_preserves_historical_bytes(tmp_path):
    import hashlib

    p = tmp_path / "historical"
    p.write_bytes(b"accepted")
    h = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()}
    verify_hashes(h)
    p.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_hashes(h)


def test_active_pressure_requires_two_consecutive_windows():
    assert resource_gate(4608, [])
    assert not resource_gate(4609, [])
    high = {"seconds": 60, "swap_io_bytes": 256 * 2**20}
    low = {"seconds": 60, "swap_io_bytes": 0}
    assert resource_gate(4000, [high, low, high])
    assert not resource_gate(4000, [high, high])
    assert json.dumps(inputs())


def test_swap_monitor_samples_loading_windows_and_stops(monkeypatch):
    from embedding_diagnostics.jina_protocol import SwapMonitor

    clock = [0.0]
    swap = [0]
    m = SwapMonitor(clock=lambda: clock[0], counter=lambda: swap[0])
    clock[0] = 60
    swap[0] = 256 * 2**20
    m.sample()
    clock[0] = 120
    swap[0] = 512 * 2**20
    m.sample()
    assert len(m.windows) == 2
    assert not resource_gate(4000, m.windows)
    with m:
        assert m.thread.is_alive()
    assert not m.thread.is_alive()


def test_resolution_patch_only_changes_pinning_and_rejects_source_mutation(tmp_path):
    from embedding_diagnostics.jina_protocol import digest, resolve_sources

    src = tmp_path / "original"
    src.mkdir()
    files = {
        "modeling_clip.py": "        revision=None,\n",
        "hf_model.py": "                self.config = AutoConfig.from_pretrained(\n"
        "                    model_name_or_path,\n",
        "numerics.py": "def pooling(x): return x\n",
    }
    for n, t in files.items():
        (src / n).write_text(t)
    a = {
        "roots": {"jinaai/jina-clip-implementation": str(src)},
        "files": {str(src / n): {"sha256": digest(src / n)} for n in files},
    }
    records = resolve_sources(a, tmp_path / "resolved")
    assert records["numerics.py"]["diff"] == ""
    assert (tmp_path / "resolved/modeling_clip.py").read_text().count("code_revision=") == 1
    assert "revision=revision," in (tmp_path / "resolved/hf_model.py").read_text()
    (src / "numerics.py").write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        resolve_sources(a, tmp_path / "second")
