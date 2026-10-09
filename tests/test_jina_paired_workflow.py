import importlib.util
from pathlib import Path

import numpy as np

from embedding_diagnostics.jina_paired_cache import JinaPairedCache
from embedding_diagnostics.jina_protocol import cache_provenance


def runner():
    spec = importlib.util.spec_from_file_location(
        "jina_runner", Path("scripts/run_jina_paired_replication.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_extraction_has_zero_model_calls(tmp_path, monkeypatch):
    m = runner()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    freeze = {"repositories": {"jinaai/jina-clip-v2": "frozen"}}
    (tmp_path / "freeze.json").write_text("{}")
    rows = {
        "image": [{"id": 1, "parent_image_id": 1, "role": "image", "input_sha256": "a" * 64}],
        "caption": [
            {
                "id": i,
                "parent_image_id": 1,
                "role": "caption",
                "input_sha256": "a" * 64,
                "rendered": "plain",
                "token_ids": [0, 2],
            }
            for i in range(5)
        ],
    }
    m.atomic_json(tmp_path / "input_manifest.json", rows)
    m.atomic_json(
        tmp_path / "smoke_metrics.json",
        {
            "resource_gate_passed": True,
            "roles": {"image": {"completed": 16}, "caption": {"completed": 80}},
            "contract_passed": True,
            "checks": {
                r: {
                    "equivalent": True,
                    "prefixes": {str(d): {"equivalent": True} for d in (256, 128)},
                }
                for r in ("image", "caption")
            },
        },
    )
    for r in rows:
        with JinaPairedCache(
            tmp_path / r / "embedding-cache",
            cache_provenance(freeze, r, tmp_path),
            m.cache_rows(rows[r]),
        ) as c:
            c.append(np.ones((len(rows[r]), 1024), dtype=np.float32) / 32)
    monkeypatch.setattr(m, "checked_freeze", lambda *a, **kw: freeze)

    def forbidden(*a, **kw):
        raise AssertionError("completed resume loaded model")

    monkeypatch.setattr(m, "JinaCLIPv2", forbidden)
    m.extract()
    report = m.read(tmp_path / "extraction_metrics.json")
    assert report["new_inferences"] == report["forward_calls"] == 0
    assert not report["model_loaded"]


def test_completed_endpoint_resume_does_no_ranking_or_interval(tmp_path, monkeypatch):
    m = runner()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "checked_freeze", lambda *a, **kw: {"protected_sha256": {}})
    (tmp_path / "freeze.json").write_text("{}")
    binding = {"freeze_sha256": m.digest(tmp_path / "freeze.json"), "files_sha256": {}}
    m.atomic_json(tmp_path / "cache_binding.json", binding)
    m.atomic_json(tmp_path / "input_manifest.json", {"image": [{"id": 1}]})
    monkeypatch.setattr(m, "require_committed", lambda *a: None)
    monkeypatch.setattr(m, "completed_records", lambda *a: True)
    monkeypatch.setattr(m, "finish_intervals", lambda *a: 0)
    m.evaluate()
    report = m.read(tmp_path / "endpoint_resume.json")
    assert report["endpoint_calls"] == report["interval_calls"] == report["reference_rankings"] == 0


def test_saved_failed_smoke_cannot_be_promoted_by_resume(tmp_path, monkeypatch):
    import pytest

    m = runner()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "checked_freeze", lambda *a, **k: {})
    (tmp_path / "freeze.json").write_text("{}")
    m.atomic_json(tmp_path / "input_manifest.json", {"image": [], "caption": []})
    m.atomic_json(
        tmp_path / "smoke_checks.json",
        {
            "freeze_sha256": m.digest(tmp_path / "freeze.json"),
            "checks": {
                "image": {
                    "equivalent": False,
                    "prefixes": {"256": {"equivalent": True}, "128": {"equivalent": True}},
                }
            },
        },
    )
    with pytest.raises(ValueError, match="failed"):
        m.extract(smoke=True)


def test_recorded_numerical_failure_stops_resume_before_model(tmp_path, monkeypatch):
    import pytest

    m = runner()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "checked_freeze", lambda *a, **kw: {})
    m.atomic_json(tmp_path / "numerical_blocker.json", {"error": "activation float16"})
    with pytest.raises(ValueError, match="blocker"):
        m.extract(smoke=True)


def test_interrupted_interval_finalization_recovers_and_rejects_corruption(tmp_path):
    import pytest

    m = runner()
    records = [
        {
            "condition": n,
            "retrieval": {d: {"per_parent": {"1": {"hit@10": v}}} for d in ("t2i", "i2t")},
        }
        for n, v in [("native_1024", 1), ("mrl_256", 0), ("mrl_128", 0)]
    ]
    assert m.finish_intervals(tmp_path, records, [1]) == 4
    assert m.finish_intervals(tmp_path, records, [1]) == 0
    (tmp_path / "paired_intervals.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        m.finish_intervals(tmp_path, records, [1])
