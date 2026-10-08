"""Role identity, row alignment and crash-safe reuse without real inference."""

import numpy as np
import pytest

from embedding_diagnostics.paired_cache import PairedCache, load_paired_cache


def rows(role="query"):
    return [{"id": i, "parent_image_id": 10, "role": role,
             "input_sha256": "a"*64} for i in [3, 1]]


def matrix(n):
    x = np.zeros((n, 768), dtype=np.float32)
    x[:, 0] = 1
    return x


def test_partial_resume_preserves_order_and_completed_prefix(tmp_path):
    with PairedCache(tmp_path, {"role": "query"}, rows()) as cache:
        cache.append(matrix(1))
    with pytest.raises(ValueError, match="incomplete"):
        load_paired_cache(tmp_path)
    with PairedCache(tmp_path, {"role": "query"}, rows()) as cache:
        assert cache.completed == 1
        cache.append(matrix(1))
    x, manifest = load_paired_cache(tmp_path)
    assert x.shape == (2, 768) and manifest["rows"] == rows()
    with PairedCache(tmp_path, {"role": "query"}, rows()) as cache:
        assert cache.completed == 2


def test_role_selection_and_provenance_mismatch_rejected(tmp_path):
    with PairedCache(tmp_path, {"role": "query"}, rows()):
        pass
    for provenance, selection in [({"role": "document"}, rows("document")),
                                  ({"role": "query"}, list(reversed(rows()))),
                                  ({"role": "query", "revision": "other"}, rows())]:
        with pytest.raises(ValueError, match="mismatch"):
            PairedCache(tmp_path, provenance, selection)


def test_bad_vectors_and_checksum_tampering_fail(tmp_path):
    with PairedCache(tmp_path, {"role": "query"}, rows()) as cache:
        for x in [np.zeros((1,768),np.float32), np.full((1,768),np.nan,np.float32),
                  np.ones((1,128),np.float32)]:
            with pytest.raises(ValueError):
                cache.append(x)
        cache.append(matrix(2))
    chunk = next(tmp_path.glob('chunk-*.npz'))
    chunk.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_paired_cache(tmp_path)


def test_row_metadata_validated(tmp_path):
    bad = rows()
    bad[0]["input_sha256"] = "not a hash"
    with pytest.raises(ValueError):
        PairedCache(tmp_path, {}, bad)
