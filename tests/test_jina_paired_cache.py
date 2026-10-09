import json

import numpy as np
import pytest

from embedding_diagnostics.jina_paired_cache import JinaPairedCache, load_jina_cache


def rows():
    return [
        {"id": i, "parent_image_id": 1, "role": "caption", "input_sha256": "a" * 64} for i in (2, 4)
    ]


def matrix(n=1):
    return np.ones((n, 1024), dtype=np.float32) / 32


def test_prefix_resume_and_complete_metadata(tmp_path):
    with JinaPairedCache(tmp_path, {"model": "jina"}, rows()) as c:
        c.append(matrix())
    with JinaPairedCache(tmp_path, {"model": "jina"}, rows()) as c:
        assert c.completed == 1
        c.append(matrix())
    x, m = load_jina_cache(tmp_path)
    assert x.shape == (2, 1024) and m["rows"] == rows()
    with pytest.raises(ValueError, match="mismatch"):
        JinaPairedCache(tmp_path, {"model": "other"}, rows())
    with pytest.raises(ValueError):
        JinaPairedCache(tmp_path, {"model": "jina"}, rows()[::-1])


def test_checksum_and_row_alignment(tmp_path):
    with JinaPairedCache(tmp_path, {}, rows()) as c:
        c.append(matrix(2))
    manifest = tmp_path / "manifest.json"
    d = json.loads(manifest.read_text())
    d["rows"][0]["id"] = 8
    manifest.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="alignment"):
        load_jina_cache(tmp_path)


def test_historical_file_unchanged_and_duplicate_ids_rejected(tmp_path):
    old = tmp_path / "historical.json"
    old.write_bytes(b"accepted historical bytes")
    with pytest.raises(ValueError):
        JinaPairedCache(tmp_path / "new", {}, [rows()[0], rows()[0]])
    with JinaPairedCache(tmp_path / "new", {}, rows()) as c:
        c.append(matrix(2))
    assert old.read_bytes() == b"accepted historical bytes"
