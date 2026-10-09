"""Publication checks: corrected endpoints must never fall back to defective probes."""

import importlib.util
import json
from html.parser import HTMLParser
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
    assert "convergence" in page.lower()
    assert "phase_a_geometry" not in data
    assert "CIFAR" not in page
    assert "What the table shows" in page
    assert (tmp_path / "assets" / "story.js").exists()
    assert (tmp_path / "assets" / "style.css").exists()


def test_public_report_builds_with_only_bdd_records(tmp_path):
    module = builder()
    records = tmp_path / "records"
    for seed in module.SEEDS:
        for condition in module.CONDITIONS:
            source = ROOT / "experiments" / f"phaseB_s{seed}" / condition
            dest = records / f"phaseB_s{seed}" / condition
            dest.mkdir(parents=True)
            for name in ("metrics.jsonl", "metrics_reprobed.jsonl", "config.json"):
                (dest / name).write_text((source / name).read_text())
    module.build(records, tmp_path / "site")
    page = (tmp_path / "site" / "index.html").read_text()
    assert "CIFAR" not in page
    assert "Phase A" not in page and "Phase B" not in page
    assert "35 completed runs" in page


def test_pipeline_figure_is_static_readable(tmp_path):
    module = builder()
    module.build(ROOT / "experiments", tmp_path)
    page = (tmp_path / "index.html").read_text()
    assert 'class="architecture" id="architecture"' in page
    assert '<summary>Technical detail: shared architecture' not in page
    assert 'class="pipeline" id="pipeline"' in page
    for condition in module.CONDITIONS:
        assert f'value="{condition}"' in page
    assert "One optimizer step" in page
    assert "encodes no measured quantity" in page


def test_contents_links_have_static_targets(tmp_path):
    class Targets(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = set()
            self.anchors = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if "id" in attrs:
                self.ids.add(attrs["id"])
            if tag == "a" and attrs.get("href", "").startswith("#"):
                self.anchors.append(attrs["href"][1:])

    builder().build(ROOT / "experiments", tmp_path)
    page = (tmp_path / "index.html").read_text()
    parser = Targets()
    parser.feed(page)
    assert set(parser.anchors) <= parser.ids
    assert 'aria-controls="contents-links"' in page
    for target in ("overview", "architecture", "bdd-results", "uncertainty", "rank"):
        assert f'href="#{target}"' in page


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


def test_uncertainty_plots_preserve_paired_endpoints_without_javascript(tmp_path):
    class IntervalBars(HTMLParser):
        def __init__(self):
            super().__init__()
            self.bars = []
            self.zero_axes = 0

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if attrs.get("class") == "interval-bar":
                self.bars.append(attrs)
            if attrs.get("class") == "interval-zero":
                self.zero_axes += 1

    module = builder()
    data = module.collect(ROOT / "experiments")
    module.build(ROOT / "experiments", tmp_path)
    page = (tmp_path / "index.html").read_text()
    parser = IntervalBars()
    parser.feed(page)
    expected = [
        item for item in data["intervals"]
        if item["metric"].startswith(("probe_accuracy_unscaled_", "retrieval_p10_"))
    ]
    # Each attribute has two metrics and wide/compact versions of each plot.
    assert len(parser.bars) == 2 * len(expected)
    assert parser.zero_axes == 12
    for bar in parser.bars:
        item = next(
            item for item in expected
            if item["a"] == bar["data-a"] and item["b"] == bar["data-b"]
            and item["metric"] == bar["data-metric"]
        )
        assert float(bar["data-low"]) == pytest.approx(100 * item["low"])
        assert float(bar["data-high"]) == pytest.approx(100 * item["high"])
        assert float(bar["data-mean"]) == pytest.approx(100 * item["mean"])
    assert "How uncertain are these differences?" in page
    assert 'id="interval-table"' not in page


def test_synthesis_exports_qualified_sample_and_recorded_coco_values():
    module = builder()
    lock, _, records = module.synthesis_sources()
    data = module.synthesis_data(records)
    native = next(p for p in data['bdd'] if p['condition'] == 'c1_pristine')
    assert sum(v['val'] for v in native['attributes']['weather']['support'].values()) == 983
    assert sum(v['train'] for v in native['attributes']['weather']['support'].values()) == 2000
    assert data['coco'][0]['retrieval']['i2t']['hit@1'] == .843
    assert data['coco'][2]['retrieval']['i2t']['set_recall@10'] == .689
    assert lock['source_commit'].startswith('e902aa0')
    assert '68.90' in module.report_tables(data)


def test_modified_accepted_evidence_blocks_synthesis(tmp_path, monkeypatch):
    module = builder()
    (tmp_path / 'viz').mkdir()
    evidence = tmp_path / 'accepted.json'
    evidence.write_text('changed')
    (tmp_path / 'viz/evidence-lock.json').write_text(json.dumps({
        'files': {'accepted.json': '0' * 64}}))
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='checksum mismatch'):
        module.synthesis_sources()


def test_generated_report_table_matches_checked_in_artifact():
    module = builder()
    _, _, records = module.synthesis_sources()
    generated = module.report_tables(module.synthesis_data(records))
    assert (ROOT / 'report/sections/final_tables.tex').read_text() == generated
    for name, table in module.report_evidence_tables(module.synthesis_data(records)).items():
        assert (ROOT / f'report/sections/final_{name}.tex').read_text() == table


def test_all_packaged_links_and_fragments_resolve(tmp_path):
    from urllib.parse import unquote, urlsplit

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.targets = set()
            self.links = []

        def handle_starttag(self, tag, attrs):
            values = dict(attrs)
            if 'id' in values:
                assert values['id'] not in self.targets, 'duplicate HTML id'
                self.targets.add(values['id'])
            for key in ('href', 'src'):
                if key in values:
                    self.links.append(values[key])

    builder().build(ROOT / 'experiments', tmp_path)
    documents = {}
    for path in tmp_path.rglob('*.html'):
        parser = Links()
        parser.feed(path.read_text())
        documents[path] = parser
    for path, document in documents.items():
        for link in document.links:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                continue  # Network links are checked separately, not claimed here.
            target = path.parent / unquote(parsed.path) if parsed.path else path
            target = target.resolve()
            assert target.is_file(), (path, link)
            if parsed.fragment:
                assert unquote(parsed.fragment) in documents[target].targets, (path, link)
    page = (tmp_path / 'index.html').read_text()
    assert '__C1_TABLE__' not in page
    assert 'second-encoder replication with Jina-CLIP-v2 could not be completed' in page


def test_generated_latex_alignment_rows_have_valid_terminators():
    module = builder()
    _, _, records = module.synthesis_sources()
    data = module.synthesis_data(records)
    tables = [module.report_tables(data), *module.report_evidence_tables(data).values()]
    for table in tables:
        for line in table.splitlines():
            if ' & ' in line:
                assert line.endswith(chr(92) * 2), line


def test_extension_figures_use_accepted_endpoints_and_bootstrap_intervals(tmp_path):
    """Changing a plotted endpoint or interval must break the publication contract."""
    class Marks(HTMLParser):
        def __init__(self):
            super().__init__()
            self.values = []
            self.intervals = []

        def handle_starttag(self, tag, attrs):
            values = dict(attrs)
            if 'data-source-metric' in values:
                self.values.append(values)
            if values.get('class') == 'paired-ci':
                self.intervals.append(values)

    module = builder()
    module.build(ROOT / 'experiments', tmp_path)
    page = (tmp_path / 'index.html').read_text()
    parser = Marks()
    parser.feed(page)
    assert parser.values, 'Accepted extension endpoints need static measured figures'
    _, _, records = module.synthesis_sources()
    data = module.synthesis_data(records)
    lookup = {p['condition']: p for p in data['bdd'] + data['coco']}
    for mark in parser.values:
        value = lookup[mark['data-source-condition']]
        for key in mark['data-source-metric'].split('.'):
            value = value[key]
        assert float(mark['data-source-value']) == pytest.approx(value)
    accepted = json.loads((ROOT / 'experiments/phaseC_paired_replication/paired_intervals.json')
                          .read_text())
    assert len(parser.intervals) == 4
    for mark in parser.intervals:
        record = accepted[mark['data-condition']][mark['data-direction']]
        for key in ('delta', 'lower', 'upper'):
            assert float(mark[f'data-{key}']) == pytest.approx(100 * record[key])
    assert 'conditional on the fixed gallery' in page


def test_integrated_story_defines_methods_before_showing_results(tmp_path):
    module = builder()
    module.build(ROOT / 'experiments', tmp_path)
    page = (tmp_path / 'index.html').read_text()
    positions = [page.index(f'id="{target}"') for target in (
        'motivation', 'experiment', 'data', 'evidence', 'geometry', 'compression',
        'numerics', 'paired', 'qualification', 'bibliography', 'appendix')]
    assert positions == sorted(positions)
    for label in ('[ours]', '[established]', '[interpretation]', '[replication]', '—'):
        assert label not in page
    assert 'id="ref-mrl"' in page and 'id="ref-coco"' in page


def test_static_training_figures_preserve_five_seed_means(tmp_path):
    class TrainingMarks(HTMLParser):
        def __init__(self):
            super().__init__()
            self.marks = []

        def handle_starttag(self, tag, attrs):
            values = dict(attrs)
            if 'data-training-metric' in values:
                self.marks.append(values)

    module = builder()
    data = module.collect(ROOT / 'experiments')
    module.build(ROOT / 'experiments', tmp_path)
    parser = TrainingMarks()
    parser.feed((tmp_path / 'index.html').read_text())
    assert len(parser.marks) == 6
    metrics = {'total_variance', 'probe_accuracy_unscaled_timeofday', 'retrieval_p10_timeofday'}
    assert {p['data-training-metric'] for p in parser.marks} == metrics
    for mark in parser.marks:
        values = data['conditions'][mark['data-training-condition']][mark['data-training-metric']]
        assert float(mark['data-training-value']) == pytest.approx(sum(values)/5)
