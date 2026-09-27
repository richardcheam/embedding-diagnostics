import sys
from pathlib import Path

from embedding_diagnostics.report_html import build_html

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz"))

from build_report import build_scorecard  # noqa: E402

RUNS = {
    "ema_stopgrad": [
        {"step": 0, "probe_accuracy": 0.1, "rankme": 8.0, "projection": [[0.0, 0.0]]},
        {"step": 500, "probe_accuracy": 0.4, "rankme": 7.0, "projection": [[1.0, 1.0]]},
    ],
    "sigreg_nostopgrad": [
        {"step": 0, "probe_accuracy": 0.1, "rankme": 8.0, "projection": [[0.0, 0.0]]},
        {"step": 500, "probe_accuracy": 0.2, "rankme": 2.0, "projection": [[0.1, 0.1]]},
    ],
}


def make_finals(key: str, dead: list[float], weak: list[float], healthy: list[float]):
    """Final-checkpoint records for the three reference conditions, one per seed."""
    return {
        "none_nostopgrad": [{"step": 1, key: v} for v in dead],
        "none_stopgrad": [{"step": 1, key: v} for v in weak],
        "ema_stopgrad": [{"step": 1, key: v} for v in healthy],
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


def test_provenance_is_rendered():
    """The page must state which runs its numbers came from, since the curves
    and the scorecard deliberately come from different seed sets."""
    html = build_html(RUNS, title="t", provenance="Curves: phaseA_s0, one seed.")
    assert "phaseA_s0, one seed" in html


def test_claim_labels_are_rendered():
    """Labels must ship with the figure so it cannot circulate stripped of caveat."""
    html = build_html(RUNS, title="test")
    lowered = html.lower()
    assert "established" in lowered
    assert "ours" in lowered


def test_handles_empty_runs():
    html = build_html({}, title="empty")
    assert "<!doctype html>" in html


def test_scorecard_rows_are_embedded():
    rows = [
        {
            "name": "total variance",
            "collapsed": "0.0002",
            "weakest": "21.57",
            "healthy": "87.75",
            "verdict": "separates them",
            "status": "correct",
        }
    ]
    page = build_html(RUNS, title="t", scorecard=rows)
    assert "total variance" in page
    assert "0.0002" in page


def test_scorecard_defaults_to_empty_rather_than_inventing_numbers():
    """Curves are one seed and the scorecard is across seeds. If the scorecard
    were derived from the curves the page would state a false basis."""
    page = build_html(RUNS, title="t")
    assert "const SCORECARD = [];" in page


def test_embedded_data_cannot_close_the_script_tag():
    """A condition name containing </script> must not break the page.

    json.dumps does not escape it, so interpolating raw output into a
    <script> block would end the script early and silently corrupt the report.
    """
    hostile = {"</script><b>x</b>": [{"step": 0, "probe_accuracy": 0.1}]}
    page = build_html(hostile, title="t")
    assert "</script><b>" not in page
    assert page.count("</script>") == 1


def test_title_is_html_escaped():
    page = build_html({}, title="<img src=x onerror=alert(1)>")
    assert "<img src=x" not in page


def test_provenance_is_html_escaped():
    page = build_html({}, title="t", provenance="<img src=x onerror=alert(1)>")
    assert "<img src=x" not in page


# --- scorecard verdicts -------------------------------------------------


def test_a_metric_that_separates_them_resolvably_is_correct():
    rows = build_scorecard(
        make_finals("total_variance", dead=[0.0002] * 5, weak=[20.0, 21.0, 22.0, 20.5, 21.5],
                    healthy=[87.0] * 5)
    )
    row = next(r for r in rows if r["name"] == "total variance")
    assert row["status"] == "correct"


def test_a_metric_pointing_the_wrong_way_is_blind():
    """The standardized probe's actual Phase-A behaviour: it rates the dead
    encoder ABOVE one that is merely weak."""
    rows = build_scorecard(
        make_finals("probe_accuracy", dead=[0.413] * 5, weak=[0.347] * 5, healthy=[0.541] * 5)
    )
    row = next(r for r in rows if r["name"] == "standardized linear probe")
    assert row["status"] == "blind"


def test_a_right_direction_margin_inside_noise_is_unresolved():
    """Retrieval's actual behaviour: right way by 0.002, which its own
    across-seed spread swallows. That is a different failure from pointing the
    wrong way and must not be scored the same."""
    rows = build_scorecard(
        make_finals(
            "retrieval_p10",
            dead=[0.180, 0.185, 0.183, 0.186, 0.184],
            weak=[0.170, 0.200, 0.180, 0.195, 0.188],
            healthy=[0.358] * 5,
        )
    )
    row = next(r for r in rows if r["name"] == "retrieval P@10 (cosine)")
    assert row["status"] == "unresolved"


def test_lower_is_healthier_metrics_are_scored_in_their_own_direction():
    """Cosine near 1 is the collapsed end. A metric whose healthy direction is
    DOWN must not be scored as if higher were better."""
    rows = build_scorecard(
        make_finals(
            "mean_pairwise_cosine",
            dead=[1.0] * 5,
            weak=[0.30, 0.32, 0.31, 0.29, 0.33],
            healthy=[0.20] * 5,
        )
    )
    row = next(r for r in rows if r["name"] == "mean pairwise cosine")
    assert row["status"] == "correct"


def test_rows_are_ordered_correct_then_unresolved_then_blind():
    rows = build_scorecard(
        {
            "none_nostopgrad": [
                {"probe_accuracy": 0.413, "total_variance": 0.0002} for _ in range(5)
            ],
            "none_stopgrad": [
                {"probe_accuracy": 0.347, "total_variance": 20.0 + i} for i in range(5)
            ],
            "ema_stopgrad": [
                {"probe_accuracy": 0.541, "total_variance": 87.0} for _ in range(5)
            ],
        }
    )
    order = {"correct": 0, "unresolved": 1, "blind": 2}
    statuses = [r["status"] for r in rows]
    assert statuses == sorted(statuses, key=lambda s: order[s])


def test_scorecard_is_empty_without_the_three_reference_conditions():
    assert build_scorecard({"ema_stopgrad": [{"total_variance": 1.0}]}) == []


def test_metrics_missing_from_the_logs_are_skipped_not_faked():
    rows = build_scorecard(make_finals("total_variance", [0.1] * 5, [1.0] * 5, [2.0] * 5))
    assert {r["name"] for r in rows} == {"total variance"}
