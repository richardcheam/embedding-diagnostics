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
    assert data["seeds"] == [0, 1, 2, 3, 4]
    assert data["step"] == 4000
    assert len(data["conditions"]) == 7
    assert control["probe_converged_unscaled_timeofday"][0] == 0


def test_missing_reprobe_blocks_publication(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        builder().collect(tmp_path)


def test_table_extrema_keep_displayed_ties_and_avoid_uniform_verdicts():
    module = builder()
    columns = list(zip(["91.37%", "1.00"], ["92.71%", "1.00"], ["92.71%", "1.00"]))
    low = module.highlighted_cells(["91.37%", "1.00"], columns)
    high = module.highlighted_cells(["92.71%", "1.00"], columns)
    assert low.count('class="metric-extreme metric-low"') == 1
    assert high.count('class="metric-extreme metric-high"') == 1
    assert "<td>1.00</td>" in high  # An equal column has no high/low annotation.


def test_build_publishes_data_and_static_evidence(tmp_path):
    builder().build(ROOT / "experiments", tmp_path)
    data = json.loads((tmp_path / "data.json").read_text())
    page = (tmp_path / "index.html").read_text()
    assert data["conditions"]["none_nostopgrad"]["probe_accuracy_unscaled_timeofday"][0] > 0.9
    assert "91.37" in page  # Corrected evidence remains readable without JavaScript.
    assert "48.32" not in page
    assert 'id="audit"' not in page
    assert "Classification remains unresolved" in page
    assert "convergence" in page.lower()
    assert len(data["phase_a_geometry"]["conditions"]) == 7
    assert "__CIFAR_TABLE__" not in page
    assert 'class="cifar-geometry"' in page
    assert (tmp_path / "assets" / "story.js").exists()
    assert (tmp_path / "assets" / "style.css").exists()


def test_phase_a_publishes_geometry_without_defective_probes():
    module = builder()
    data = module.collect_phase_a_geometry(ROOT / "experiments")
    for columns in data["conditions"].values():
        assert set(columns) == set(module.GEOMETRY)
        assert all(len(values) == 5 for values in columns.values())


def test_phase_a_nonfinal_endpoint_blocks_publication(tmp_path):
    directory = tmp_path / "phaseA_s0" / "ema_stopgrad"
    directory.mkdir(parents=True)
    source = ROOT / "experiments" / "phaseA_s0" / "ema_stopgrad"
    (directory / "config.json").write_text((source / "config.json").read_text())
    (directory / "metrics.jsonl").write_text(json.dumps({"step": 200}))
    with pytest.raises(ValueError, match="final matched Phase-A"):
        builder().collect_phase_a_geometry(tmp_path)


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
