"""Build the case study from final geometry logs and corrected Phase-B probes.

Standard-library only: publishing must not download CUDA or require saved encoders.
Paired Student-t intervals follow embedding_diagnostics.stats (five paired seeds,
df=4). The authoritative contrasts and quantiles are read as literals from that
module, without importing the training package's numerical dependencies.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import shutil
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [0, 1, 2, 3, 4]
CONDITIONS = [
    "ema_stopgrad", "none_stopgrad", "sigreg_stopgrad", "sigreg_nostopgrad",
    "proj_sigreg_stopgrad", "proj_sigreg_nostopgrad", "none_nostopgrad",
]
ATTRIBUTES = ["weather", "scene", "timeofday"]
GEOMETRY = ["total_variance", "mean_pairwise_cosine", "rankme", "participation_ratio"]
LABELS = {
    "ema_stopgrad": "EMA reference",
    "none_stopgrad": "Stop-gradient only",
    "none_nostopgrad": "Contracted control",
    "sigreg_stopgrad": "SIGReg + stop-gradient",
    "sigreg_nostopgrad": "SIGReg",
    "proj_sigreg_stopgrad": "Projector + SIGReg + stop-gradient",
    "proj_sigreg_nostopgrad": "Projector + SIGReg",
}


def read_records(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Empty record file: {path}")
    return records


def protocol() -> tuple[list, dict]:
    tree = ast.parse((ROOT / "src/embedding_diagnostics/stats.py").read_text())
    literals = {
        node.target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in {"CONTRASTS", "_T_TABLE"}
    }
    return literals["CONTRASTS"], literals["_T_TABLE"]


def collect(experiments: Path) -> dict:
    """Require all 35 final endpoints and independently matched corrected records.

    No fallback to the original probes. Convergence and underfit flags travel
    with the scores, including corrected records that reached their iteration cap.
    """
    conditions = {}
    for condition in CONDITIONS:
        columns: dict[str, list] = {}
        for seed in SEEDS:
            directory = experiments / f"phaseB_s{seed}" / condition
            config = json.loads((directory / "config.json").read_text())
            original = max(read_records(directory / "metrics.jsonl"), key=lambda r: r["step"])
            corrected_records = read_records(directory / "metrics_reprobed.jsonl")
            if len(corrected_records) != 1:
                raise ValueError(f"Expected one corrected endpoint: {directory}")
            corrected = corrected_records[0]
            if config["seed"] != seed or corrected["seed"] != seed:
                raise ValueError(f"Mismatched seed: {directory}")
            if config["data"]["dataset"] != "bdd100k" or original["step"] != 4000:
                raise ValueError(f"Not a final Phase-B endpoint: {directory}")
            if corrected["condition"] != condition or corrected["tag"] != f"phaseB_s{seed}":
                raise ValueError(f"Mismatched corrected identity: {directory}")
            if corrected["eval_split_seed"] != config["data"]["eval_split_seed"]:
                raise ValueError(f"Mismatched evaluation split: {directory}")
            # reprobe uses population variance; metrics uses sample covariance.
            # Correct that known convention before checking numerical consistency.
            n = config["data"]["probe_test_samples"]
            if not math.isclose(
                original["total_variance"], corrected["total_variance"] * n / (n - 1),
                rel_tol=1e-4,
            ):
                raise ValueError(f"Inconsistent remeasured variance: {directory}")
            values = {key: original[key] for key in GEOMETRY}
            for attribute in ATTRIBUTES:
                for prefix in ("retrieval_p10", "retrieval_chance"):
                    key = f"{prefix}_{attribute}"
                    values[key] = original[key]
                for prefix in (
                    "probe_accuracy_unscaled", "probe_balanced_accuracy_unscaled",
                    "probe_majority", "probe_converged_unscaled", "probe_underfit_train_unscaled",
                    "probe_selected_C_unscaled", "probe_n_iter_unscaled",
                ):
                    key = f"{prefix}_{attribute}"
                    values[key] = corrected[key]
            for key, value in values.items():
                if not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError(f"Invalid {key}: {directory}")
                columns.setdefault(key, []).append(value)
        conditions[condition] = columns

    contrasts, quantiles = protocol()
    intervals = []
    endpoints = GEOMETRY + [
        f"{prefix}_{attribute}"
        for attribute in ATTRIBUTES
        for prefix in (
            "probe_accuracy_unscaled", "probe_balanced_accuracy_unscaled", "retrieval_p10"
        )
    ]
    for a, b, name in contrasts:
        for key in endpoints:
            differences = [x - y for x, y in zip(conditions[a][key], conditions[b][key])]
            mean = statistics.mean(differences)
            half = quantiles[4] * statistics.stdev(differences) / math.sqrt(5)
            intervals.append({
                "a": a, "b": b, "name": name, "metric": key,
                "mean": mean, "low": mean - half, "high": mean + half,
                "differences": differences,
            })
    return {"seeds": SEEDS, "step": 4000, "conditions": conditions, "intervals": intervals}


def mean(data: dict, condition: str, key: str) -> float:
    return statistics.mean(data["conditions"][condition][key])


def highlighted_cells(values: list[str], columns: list[tuple[str, ...]]) -> str:
    """Mark displayed column extrema, including ties; do not assign quality verdicts."""
    cells = []
    for value, column in zip(values, columns):
        number = float(value.rstrip("%"))
        numbers = [float(item.rstrip("%")) for item in column]
        role = ""
        if min(numbers) != max(numbers):
            role = "high" if number == max(numbers) else "low" if number == min(numbers) else ""
        if role:
            label = "High" if role == "high" else "Low"
            content = (
                f'<span class="metric-extreme metric-{role}">{value}'
                f'<span class="extreme-label">{label}</span></span>'
            )
        else:
            content = value
        cells.append(f"<td>{content}</td>")
    return "".join(cells)


def evidence_table(data: dict) -> str:
    rows = []
    symbols = {
        "ema_stopgrad": ("reference", "●"),
        "none_stopgrad": ("comparison", "◆"),
        "none_nostopgrad": ("control", "■"),
    }
    values_by_condition = {
        condition: [
            f"{mean(data, condition, 'total_variance'):.3g}",
            f"{mean(data, condition, 'rankme'):.2f}",
            f"{100 * mean(data, condition, 'probe_accuracy_unscaled_timeofday'):.2f}%",
            f"{100 * mean(data, condition, 'retrieval_p10_timeofday'):.2f}%",
        ]
        for condition in CONDITIONS
    }
    columns = list(zip(*values_by_condition.values()))
    for condition, values in values_by_condition.items():
        symbol = ""
        if condition in symbols:
            role, marker = symbols[condition]
            symbol = (
                f'<span class="condition-symbol condition-{role}" '
                f'aria-hidden="true">{marker}</span>'
            )
        rows.append(
            f'<tr><th scope="row">{symbol}{html.escape(LABELS[condition])}</th>'
            + highlighted_cells(values, columns) + "</tr>"
        )
    return "\n".join(rows)



def interval_svg(items: list[dict], limit: float, compact: bool) -> str:
    """Plot recorded paired differences and t intervals; colour carries no verdict."""
    width = 360 if compact else 1000
    left, right = (22, 338) if compact else (345, 968)
    top, spacing = (96, 105) if compact else (60, 85)
    bottom = top + (len(items) - 1) * spacing + 25

    def x(value: float) -> float:
        return left + (100 * value + limit) / (2 * limit) * (right - left)

    description_id = (
        f'interval-description-{items[0]["metric"]}-'
        f'{"compact" if compact else "wide"}'
    )
    description = " ".join(
        f'{item["name"]}. A: {LABELS[item["a"]]}; B: {LABELS[item["b"]]}. '
        f'Mean A minus B: {100 * item["mean"]:+.2f} percentage points. '
        f'95% interval: {100 * item["low"]:+.2f} to {100 * item["high"]:+.2f} '
        'percentage points.'
        for item in items
    )
    parts = [
        f'<svg class="interval-{"compact" if compact else "wide"}" '
        f'viewBox="0 0 {width} {bottom + 82}" role="img" '
        f'aria-describedby="{description_id}" '
        'aria-label="Five paired comparisons: seed points, mean diamonds, '
        'and 95 percent intervals. Scores are A minus B in percentage points.">',
        f'<desc id="{description_id}">{html.escape(description)}</desc>',
    ]
    for index in range(7):
        tick = (index - 3) * limit / 3
        position = x(tick / 100)
        anchor = "start" if index == 0 else "end" if index == 6 else "middle"
        css = "interval-zero" if index == 3 else "interval-grid"
        label = "0" if index == 3 else f"{tick:+g}"
        parts.extend([
            f'<line class="{css}" x1="{position}" x2="{position}" '
            f'y1="{top - 15}" y2="{bottom}"/>',
            f'<text class="interval-tick" x="{position}" y="{bottom + 28}" '
            f'text-anchor="{anchor}">{label}</text>',
        ])
    for index, item in enumerate(items):
        y = top + index * spacing
        label_x = left if compact else 12
        label_y = y - 63 if compact else y - 20
        title = html.escape(item["name"].replace(" effect,", ","))
        a, b = html.escape(LABELS[item["a"]]), html.escape(LABELS[item["b"]])
        parts.extend([
            f'<text class="interval-label" x="{label_x}" y="{label_y}">{title}</text>',
            f'<text class="interval-arm" x="{label_x}" y="{label_y + 19}">A: {a}</text>',
            f'<text class="interval-arm" x="{label_x}" y="{label_y + 36}">B: {b}</text>',
            f'<line class="interval-bar" data-a="{item["a"]}" data-b="{item["b"]}" '
            f'data-metric="{item["metric"]}" data-mean="{100 * item["mean"]}" '
            f'data-low="{100 * item["low"]}" data-high="{100 * item["high"]}" '
            f'x1="{x(item["low"])}" x2="{x(item["high"])}" y1="{y}" y2="{y}"/>',
        ])
        for endpoint in ("low", "high"):
            parts.append(
                f'<line class="interval-cap" x1="{x(item[endpoint])}" '
                f'x2="{x(item[endpoint])}" y1="{y - 7}" y2="{y + 7}"/>'
            )
        for seed, value in enumerate(item["differences"]):
            parts.append(
                f'<circle class="interval-seed" cx="{x(value)}" '
                f'cy="{y + (seed - 2) * 4}" r="3">'
                f'<title>Seed {seed}: {100 * value:+.3f} percentage points</title></circle>'
            )
        px = x(item["mean"])
        parts.append(
            f'<path class="interval-mean" d="M{px},{y - 6} l6,6 l-6,6 l-6,-6 Z">'
            f'<title>{a} minus {b}: mean {100 * item["mean"]:+.2f}; '
            f'95% interval [{100 * item["low"]:.2f}, {100 * item["high"]:.2f}] '
            'percentage points</title></path>'
        )
    parts.extend([
        f'<text class="interval-direction" x="{left}" y="{bottom + 61}">Lower score for A</text>',
        f'<text class="interval-direction" x="{right}" y="{bottom + 61}" '
        'text-anchor="end">Higher score for A</text>',
        '</svg>',
    ])
    return "".join(parts)


def uncertainty_figures(data: dict) -> str:
    """Generate no-JavaScript paired plots, with fixed axes across attributes."""
    metrics = {
        "probe_accuracy_unscaled": "Classification accuracy",
        "retrieval_p10": "Neighbour agreement / P@10",
    }
    limits = {}
    for prefix in metrics:
        items = [i for i in data["intervals"] if i["metric"].startswith(prefix + "_")]
        extent = max(
            abs(value) * 100 for item in items
            for value in [item["low"], item["high"], *item["differences"]]
        )
        base = 10 ** math.floor(math.log10(max(extent / 3, 0.01)))
        step = next(base * m for m in (1, 2, 5, 10) if base * m >= extent / 3)
        limits[prefix] = 3 * step
    groups = []
    for attribute in ATTRIBUTES:
        hidden = "" if attribute == "timeofday" else " hidden"
        figures = []
        for prefix, title in metrics.items():
            items = [i for i in data["intervals"] if i["metric"] == f"{prefix}_{attribute}"]
            figures.append(
                f'<figure class="uncertainty-figure"><h3>{title}</h3>'
                + interval_svg(items, limits[prefix], False)
                + interval_svg(items, limits[prefix], True)
                + '<figcaption>Difference in percentage points. Dots: five paired seeds. '
                'Diamond: mean difference. Horizontal line: paired Student-t 95% interval. '
                'Zero: equal scores. Axis limits stay fixed across attributes for this metric.'
                '</figcaption></figure>'
            )
        groups.append(
            f'<div class="uncertainty-group" data-attribute="{attribute}"{hidden}>'
            + "".join(figures) + '</div>'
        )
    return "".join(groups)


def build(experiments: Path, out: Path) -> None:
    data = collect(experiments)
    control = "none_nostopgrad"
    warnings = sum(
        value == 0
        for columns in data["conditions"].values()
        for attribute in ATTRIBUTES
        for value in columns[f"probe_converged_unscaled_{attribute}"]
    )
    probe_key = "probe_accuracy_unscaled_timeofday"
    retrieval_key = "retrieval_p10_timeofday"
    probe_gap = 100 * (mean(data, "ema_stopgrad", probe_key) - mean(data, control, probe_key))
    retrieval_gap = 100 * (
        mean(data, "ema_stopgrad", retrieval_key) - mean(data, control, retrieval_key)
    )
    projector_with_stop = 100 * (
        mean(data, "proj_sigreg_stopgrad", retrieval_key)
        - mean(data, "sigreg_stopgrad", retrieval_key)
    )
    projector_without_stop = 100 * (
        mean(data, "sigreg_nostopgrad", retrieval_key)
        - mean(data, "proj_sigreg_nostopgrad", retrieval_key)
    )
    replacements = {
        "__NEW__": f"{100 * mean(data, control, 'probe_accuracy_unscaled_timeofday'):.2f}",
        "__VARIANCE__": f"{mean(data, control, 'total_variance'):.7f}",
        "__REFERENCE_VARIANCE__": f"{mean(data, 'ema_stopgrad', 'total_variance'):.2f}",
        "__RANKME__": f"{mean(data, control, 'rankme'):.2f}",
        "__WARNINGS__": str(warnings),
        "__SIGREG_VARIANCE__": f"{mean(data, 'sigreg_stopgrad', 'total_variance'):.3g}",
        "__SIGREG_RANKME__": f"{mean(data, 'sigreg_stopgrad', 'rankme'):.2f}",
        "__PROBE_GAP__": f"{probe_gap:.2f}",
        "__RETRIEVAL_GAP__": f"{retrieval_gap:.2f}",
        "__PROJECTOR_WITH_STOP__": f"{projector_with_stop:.2f}",
        "__PROJECTOR_WITHOUT_STOP__": f"{projector_without_stop:.2f}",
        "__TABLE__": evidence_table(data),
        "__UNCERTAINTY_PLOTS__": uncertainty_figures(data),
        "__REFERENCE_PROBE__": (
            f"{100 * mean(data, 'ema_stopgrad', 'probe_accuracy_unscaled_timeofday'):.2f}"
        ),
        "__CONTROL_RETRIEVAL__": (
            f"{100 * mean(data, control, 'retrieval_p10_timeofday'):.2f}"
        ),
        "__REFERENCE_RETRIEVAL__": (
            f"{100 * mean(data, 'ema_stopgrad', 'retrieval_p10_timeofday'):.2f}"
        ),
    }
    page = (ROOT / "viz/site/index.html").read_text()
    for key, value in replacements.items():
        page = page.replace(key, value)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "viz/site/assets", out / "assets", dirs_exist_ok=True)
    (out / "index.html").write_text(page)
    (out / "data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (out / ".nojekyll").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", type=Path, default=ROOT / "experiments")
    parser.add_argument("--out", type=Path, default=ROOT / "viz/dist")
    args = parser.parse_args()
    build(args.experiments_dir, args.out)
    print(
        "Built BDD100K project page with final geometry and corrected probes: "
        f"{args.out / 'index.html'}"
    )
