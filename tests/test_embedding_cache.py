"""Crash recovery and provenance checks on small synthetic canonical vectors."""

import json

import numpy as np
import pytest

from embedding_diagnostics.embedding_cache import EmbeddingCache, load_cache


def rows():
    return [{"image_id": f"{i:03}", "split": "train" if i < 2 else "val",
             "weather": 0, "scene": 2, "timeofday": 1} for i in range(4)]


def vectors(n=2):
    matrix = np.zeros((n, 768), dtype=np.float32)
    matrix[:, 0] = 1
    return matrix


def provenance():
    return {"model_revision": "a" * 40, "dtype": "float32", "dimension": 768,
            "processor": {"rescale": 1 / 255}, "batch_size": 1}


def test_resume_reload_manifest_and_geometry(tmp_path):
    from embedding_diagnostics.diagnostics.metrics import collapse_metrics

    with EmbeddingCache(tmp_path, provenance(), rows()) as cache:
        cache.append(vectors(), ["a" * 64, "b" * 64])
        assert cache.completed == 2
    with pytest.raises(ValueError, match="incomplete"):
        load_cache(tmp_path)
    with EmbeddingCache(tmp_path, provenance(), rows()) as cache:
        assert cache.completed == 2
        matrix = vectors()
        matrix[:, :2] = [0, 1]
        cache.append(matrix, ["c" * 64, "d" * 64])
    matrix, manifest = load_cache(tmp_path)
    assert matrix.shape == (4, 768) and matrix.dtype == np.float32
    assert manifest["selected_rows"] == rows()
    assert manifest["created_at"] and manifest["provenance"] == provenance()
    assert manifest["chunks"][1]["image_sha256"] == ["c" * 64, "d" * 64]
    assert collapse_metrics(matrix)["total_variance"] == pytest.approx(2 / 3)
    assert not any("image_bytes" in key for key in manifest)


@pytest.mark.parametrize("key,value", [
    ("model_revision", "b" * 40), ("dtype", "float16"), ("dimension", 512),
    ("processor", {"rescale": 1}), ("batch_size", 2),
])
def test_resume_rejects_incompatible_provenance(tmp_path, key, value):
    with EmbeddingCache(tmp_path, provenance(), rows()):
        pass
    changed = provenance() | {key: value}
    with pytest.raises(ValueError, match="provenance"):
        EmbeddingCache(tmp_path, changed, rows())


def test_resume_rejects_different_order_or_labels(tmp_path):
    with EmbeddingCache(tmp_path, provenance(), rows()):
        pass
    with pytest.raises(ValueError, match="selection"):
        EmbeddingCache(tmp_path, provenance(), rows()[::-1])
    changed = rows()
    changed[0]["weather"] = 1
    with pytest.raises(ValueError, match="selection"):
        EmbeddingCache(tmp_path, provenance(), changed)


@pytest.mark.parametrize("matrix", [
    np.ones((2, 512), dtype=np.float32), np.full((2, 768), np.nan, dtype=np.float32),
    np.zeros((2, 768), dtype=np.float32), np.ones((2, 768), dtype=np.float16),
    np.ones((2, 768), dtype=np.float32),  # unit norm required at canonical boundary
])
def test_invalid_chunk_never_advances_completion(tmp_path, matrix):
    with EmbeddingCache(tmp_path, provenance(), rows()) as cache:
        with pytest.raises(ValueError):
            cache.append(matrix, ["a" * 64, "b" * 64])
        assert cache.completed == 0


def test_corrupted_chunk_rejected_on_reload_and_resume(tmp_path):
    with EmbeddingCache(tmp_path, provenance(), rows()) as cache:
        cache.append(vectors(4), ["a" * 64] * 4)
    chunk = next(tmp_path.glob("*.npz"))
    chunk.write_bytes(b"corrupt")
    for call in (lambda: load_cache(tmp_path),
                 lambda: EmbeddingCache(tmp_path, provenance(), rows())):
        with pytest.raises(ValueError, match="checksum"):
            call()


def test_orphan_chunk_is_ignored_after_manifest_commit_failure(tmp_path, monkeypatch):
    with EmbeddingCache(tmp_path, provenance(), rows()) as cache:
        def fail(*args):
            raise OSError("crash")
        monkeypatch.setattr(cache, "_commit", fail)
        with pytest.raises(OSError, match="crash"):
            cache.append(vectors(), ["a" * 64] * 2)
    with EmbeddingCache(tmp_path, provenance(), rows()) as cache:
        assert cache.completed == 0
        cache.append(vectors(4), ["a" * 64] * 4)
    assert load_cache(tmp_path)[0].shape == (4, 768)


def test_concurrent_writer_is_refused(tmp_path):
    with EmbeddingCache(tmp_path, provenance(), rows()):
        with pytest.raises(ValueError, match="locked"):
            EmbeddingCache(tmp_path, provenance(), rows())


def test_manifest_chunk_order_is_validated(tmp_path):
    with EmbeddingCache(tmp_path, provenance(), rows()) as cache:
        cache.append(vectors(4), ["a" * 64] * 4)
    path = tmp_path / "manifest.json"
    payload = json.loads(path.read_text())
    payload["chunks"][0]["start"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="order"):
        load_cache(tmp_path)
