"""Publication checks: corrected endpoints must never fall back to defective probes."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def builder():
    path = ROOT / "viz" / "build_site.py"
    assert path.exists(), "The corrected project-page builder is missing"
    spec = importlib.util.spec_from_file_location("build_site", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_site_uses_corrected_phase_b_probes():
    data = builder().collect(ROOT / "experiments")
    control = data["conditions"]["none_nostopgrad"]
    assert control["probe_accuracy_unscaled_timeofday"][0] == pytest.approx(0.91325)
    assert control["original_probe_timeofday"][0] < 0.5
    assert data["seeds"] == [0, 1, 2, 3, 4]
    assert data["step"] == 4000
    assert len(data["conditions"]) == 7
    assert control["probe_converged_unscaled_timeofday"][0] == 0


def test_missing_reprobe_blocks_publication(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        builder().collect(tmp_path)


def test_build_publishes_data_and_static_evidence(tmp_path):
    builder().build(ROOT / "experiments", tmp_path)
    data = json.loads((tmp_path / "data.json").read_text())
    page = (tmp_path / "index.html").read_text()
    assert data["conditions"]["none_nostopgrad"]["probe_accuracy_unscaled_timeofday"][0] > 0.9
    assert "91.37" in page  # Corrected evidence remains readable without JavaScript.
    assert "48.32" in page
    assert "Not corrected" in page
    assert "convergence" in page.lower()
    assert (tmp_path / "assets" / "story.js").exists()
    assert (tmp_path / "assets" / "style.css").exists()


def test_corrected_record_with_wrong_seed_is_rejected(tmp_path):
    module = builder()
    for seed in module.SEEDS:
        for condition in module.CONDITIONS:
            source = ROOT / "experiments" / f"phaseB_s{seed}" / condition
            dest = tmp_path / f"phaseB_s{seed}" / condition
            dest.mkdir(parents=True)
            for name in ("metrics.jsonl", "metrics_reprobed.jsonl", "config.json"):
                (dest / name).write_text((source / name).read_text())
    target = tmp_path / "phaseB_s0" / "none_nostopgrad" / "metrics_reprobed.jsonl"
    record = json.loads(target.read_text())
    record["seed"] = 4
    target.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="seed"):
        module.collect(tmp_path)
