from jepa_lens.report_html import build_html

RUNS = {
    "ema_stopgrad": [
        {"step": 0, "probe_accuracy": 0.1, "effective_rank": 8.0, "projection": [[0.0, 0.0]]},
        {"step": 500, "probe_accuracy": 0.4, "effective_rank": 7.0, "projection": [[1.0, 1.0]]},
    ],
    "sigreg_nostopgrad": [
        {"step": 0, "probe_accuracy": 0.1, "effective_rank": 8.0, "projection": [[0.0, 0.0]]},
        {"step": 500, "probe_accuracy": 0.2, "effective_rank": 2.0, "projection": [[0.1, 0.1]]},
    ],
}


def test_output_is_self_contained_html():
    html = build_html(RUNS, title="test")
    assert html.startswith("<!doctype html>")
    assert "</html>" in html
    # No external requests: the page must work offline.
    assert "http://" not in html
    assert 'src="https://' not in html


def test_run_data_is_embedded():
    html = build_html(RUNS, title="test")
    assert "sigreg_nostopgrad" in html
    assert "probe_accuracy" in html


def test_claim_labels_are_rendered():
    """Labels must ship with the figure so it cannot circulate stripped of caveat."""
    html = build_html(RUNS, title="test")
    assert "replication" in html.lower()


def test_handles_empty_runs():
    html = build_html({}, title="empty")
    assert "<!doctype html>" in html
