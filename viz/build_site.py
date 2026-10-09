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
import importlib.util
import json
import math
import re
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

_figure_spec = importlib.util.spec_from_file_location("site_figures", ROOT / "viz/figures.py")
figures = importlib.util.module_from_spec(_figure_spec)
_figure_spec.loader.exec_module(figures)


def public_table(markup):
    """Readable display names; research table generation retains its original contract."""
    names = {
        'scale_contraction_0.99': 'Shrink magnitude (.99)',
        'mean_injection_0.99': 'Common offset (.99)',
        'isotropic_noise_0.99': 'Severe noise (.99)',
        'rank_truncation_0.99': 'Eight directions (.99)',
        'pristine': 'Unchanged embeddings',
        'native_fp16_gallery': 'FP16 gallery', 'native_fp16_both': 'FP16 queries + gallery',
        'native_int8_gallery': 'INT8 gallery', 'native_int8_both': 'INT8 queries + gallery',
        'native_fp32_arithmetic': 'FP32 arithmetic', 'native_ref': 'Native reference',
        'mean99_fp16_gallery': 'Offset + FP16 gallery', 'mean99_ref': 'Common-offset reference',
        'native_768': 'Native 768d', 'mrl_512': 'MRL 512d', 'pca_512': 'PCA 512d',
        'mrl_256': 'MRL 256d', 'pca_256': 'PCA 256d',
        'mrl_128': 'MRL 128d', 'pca_128': 'PCA 128d',
        'severe C1 interventions': 'severe interventions',
        'Qualified BDD': 'BDD driving-scene subset',
    }
    for old, new in names.items():
        markup = markup.replace(old, new)
    rows = re.findall(r'<tr><th scope="row">(.*?)</th>((?:<td>.*?</td>)+)</tr>', markup)
    values = [re.findall(r'<td>(.*?)</td>', row[1]) for row in rows]
    columns = list(zip(*values))
    for (label, cells), numbers in zip(rows, values):
        markup = markup.replace(
            f'<th scope="row">{label}</th>{cells}',
            f'<th scope="row">{label}</th>' + highlighted_cells(numbers, columns))
    return markup


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
                + '<figcaption>[ours] Difference in percentage points. Dots: five paired seeds. '
                'Diamond: mean difference. Horizontal line: paired Student-t 95% interval. '
                'Zero: equal scores. Axis limits stay fixed across attributes for this metric.'
                '</figcaption></figure>'
            )
        groups.append(
            f'<div class="uncertainty-group" data-attribute="{attribute}"{hidden}>'
            + "".join(figures) + '</div>'
        )
    return "".join(groups)



def synthesis_sources():
    """Validate accepted records; display recorded values without evaluating vectors."""
    import hashlib

    lock = json.loads((ROOT / "viz/evidence-lock.json").read_text())
    sources = {}
    for name, expected in lock["files"].items():
        raw = (ROOT / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Accepted evidence checksum mismatch: {name}")
        sources[name] = raw
    records = {}
    for campaign in ("phaseC_source_sensitivity", "phaseC_paired_replication"):
        source = json.loads(sources[f"experiments/{campaign}/results.json"])
        values = []
        for record in source["records"]:
            raw = json.dumps(record["payload"], sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode()
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError(f"Invalid endpoint record: {campaign}")
            values.append(record["payload"])
        records[campaign] = values
    return lock, sources, records


def synthesis_data(records):
    """Compact display export; no embeddings, neighbour recomputation or fitted models."""
    bdd = []
    for p in records["phaseC_source_sensitivity"]:
        item = {"condition": p["condition"], "campaign": p["campaign"]}
        if p["campaign"] == "C3":
            item["summary"] = p["summary"]
            item["attributes"] = {
                a: {k: v[k] for k in ("p10", "delta", "identity_changed_p10_unchanged")}
                for a, v in p["attributes"].items()
            }
        else:
            item["geometry"] = p["geometry"]["val"]
            item["attributes"] = {
                a: {"accuracy": v["probe_standardized"]["accuracy"],
                    "balanced_accuracy": v["probe_standardized"]["balanced_accuracy"],
                    "p10": v["retrieval_p10"], "support": v["class_support"],
                    "converged": v["probe_standardized"]["converged"]}
                for a, v in p["attributes"].items()
            }
        bdd.append(item)
    coco = []
    for p in records["phaseC_paired_replication"]:
        coco.append({"condition": p["condition"], "dimension": p["dimension"],
                     "geometry": p["geometry"],
                     "category_p10": p["category"]["macro_precision@10"],
                     "retrieval": {a: {k: v[k] for k in
                                       ("hit@1", "hit@10", "set_recall@10", "overlap@10")}
                                   for a, v in p["retrieval"].items()}})
    return {"bdd": bdd, "coco": coco}


def display_table(headers, rows, caption):
    """Static tables remain the accessible equivalent of every figure."""
    from html import escape

    head = "".join(f'<th scope="col">{escape(h)}</th>' for h in headers)
    body = "".join('<tr>' + ''.join(
        f'<th scope="row">{escape(str(v))}</th>' if i == 0
        else f'<td>{escape(str(v))}</td>' for i, v in enumerate(row)) + '</tr>'
        for row in rows)
    return (f'<div class="table-scroll" tabindex="0" role="region" '
            f'aria-label="{escape(caption)}"><table class="results">'
            f'<caption>[ours] {escape(caption)}</caption><thead><tr>{head}</tr>'
            f'</thead><tbody>{body}</tbody></table></div>')


def synthesis_tables(data):
    def pct(x):
        return f"{100*x:.2f}"
    attrs = ("weather", "scene", "timeofday")
    c1 = [p for p in data["bdd"] if p["condition"] in (
        "c1_pristine", "c1_scale_contraction_0.99", "c1_mean_injection_0.99",
        "c1_isotropic_noise_0.99", "c1_rank_truncation_0.99")]
    geometry = display_table(
        ["Condition", "Variance", "Cosine", "RankMe", "PR", "Weather BA %", "Weather P@10 %"],
        [[p["condition"].removeprefix("c1_"),
          f'{p["geometry"]["total_variance"]:.5g}',
          f'{p["geometry"]["mean_pairwise_cosine"]:.6f}',
          f'{p["geometry"]["rankme"]:.2f}',
          f'{p["geometry"]["participation_ratio"]:.2f}',
          pct(p["attributes"]["weather"]["balanced_accuracy"]),
          pct(p["attributes"]["weather"]["p10"])] for p in c1],
        "Qualified BDD: pristine and severe C1 interventions; 983 validation rows")
    c2 = [p for p in data["bdd"] if p["campaign"] == "C2"]
    compression = display_table(
        ["Representation", "Weather BA %", "Scene BA %", "Time BA %",
         "Weather P@10 %", "Scene P@10 %", "Time P@10 %"],
        [[p["condition"].removeprefix("c2_")]
         + [pct(p["attributes"][a]["balanced_accuracy"]) for a in attrs]
         + [pct(p["attributes"][a]["p10"]) for a in attrs] for p in c2],
        "Qualified BDD: standardized probe and attribute retrieval, 983 validation rows")
    c3 = [p for p in data["bdd"] if p["campaign"] == "C3"]
    numerics = display_table(
        ["Condition", "Max score error", "Changed queries / 983", "Overlap@10",
         "Δ weather P@10 pp", "Δ scene pp", "Δ time pp"],
        [[p["condition"].removeprefix("c3_"), f'{p["summary"]["max_score_error"]:.3g}',
          p["summary"]["identity_changed_queries"], f'{p["summary"]["mean_overlap"]:.5f}']
         + [f'{100*p["attributes"][a]["delta"]:+.3f}' for a in attrs] for p in c3],
        "Qualified BDD: storage and arithmetic conditions; deltas use each condition's reference")
    paired = display_table(
        ["Representation", "T→I Hit@1 %", "T→I Hit@10 %", "I→T Hit@1 %",
         "I→T Hit@10 %", "I→T caption recall@10 %", "Category P@10 %"],
        [[p["condition"], pct(p["retrieval"]["t2i"]["hit@1"]),
          pct(p["retrieval"]["t2i"]["hit@10"]), pct(p["retrieval"]["i2t"]["hit@1"]),
          pct(p["retrieval"]["i2t"]["hit@10"]), pct(p["retrieval"]["i2t"]["set_recall@10"]),
          pct(p["category_p10"])] for p in data["coco"]],
        "COCO: 1,000 image groups and five captions per image; separate text roles by direction")
    return geometry, compression, numerics, paired


def paired_figure(data):
    """Plot recorded paired-retrieval values; axes never encode a quality verdict."""
    values = [("T→I first positive at rank 1", "t2i", "hit@1"),
              ("T→I at least one positive in top 10", "t2i", "hit@10"),
              ("I→T first positive at rank 1", "i2t", "hit@1"),
              ("I→T at least one positive in top 10", "i2t", "hit@10"),
              ("I→T fraction of five captions in top 10", "i2t", "set_recall@10")]
    parts = ['<svg viewBox="0 0 720 320" role="img" aria-labelledby="paired-title paired-desc">',
             '<title id="paired-title">Positive hits, first-rank recovery and caption'
             ' coverage</title>',
             '<desc id="paired-desc">Recorded COCO values for native 768, MRL 256 and MRL 128. '
             'Exact values are in the adjacent table. Horizontal axis ranges from zero'
             ' to 100 percent.</desc>']
    for tick in (0, 25, 50, 75, 100):
        x = 330 + 3.4*tick
        parts.append(f'<path d="M{x} 35 V275" stroke="var(--grid)"/>'
                     f'<text x="{x}" y="300" text-anchor="middle">{tick}%</text>')
    for i, (label, direction, metric) in enumerate(values):
        y = 60 + i*47
        parts.append(f'<text x="0" y="{y+4}">{label}</text>')
        for j, p in enumerate(data["coco"]):
            value = p["retrieval"][direction][metric]
            parts.append(f'<circle cx="{330+340*value}" cy="{y+(j-1)*9}" r="4" '
                         f'fill="var(--{["finding", "focus", "control"][j]})">'
                         f'<title>{p["condition"]}: {100*value:.2f}%</title></circle>')
    return ''.join(parts) + '</svg>'


def export_evidence(out, lock, sources):
    """Package readable evidence locally so review never depends on unpublished commits."""
    from html import escape

    directory = out / "evidence"
    directory.mkdir(exist_ok=True)
    entries = []
    for name, raw in sources.items():
        if not name.endswith('.md'):
            continue
        slug = name.replace('/', '--') + '.html'
        text = escape(raw.decode())
        (directory / slug).write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{escape(name)}</title><link rel="stylesheet" href="../assets/style.css">'
            f'<body><main class="evidence-document"><a href="../index.html">Back to synthesis</a>'
            f'<h1>Evidence record</h1><p>{escape(name)}</p><p>SHA-256: '
            f'{lock["files"][name]}</p><pre>{text}</pre></main></body></html>')
        entries.append(f'<li><a href="{slug}">{escape(name)}</a></li>')
    (directory / 'index.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Evidence index</title><link rel="stylesheet" href="../assets/style.css">'
        '<body><main class="evidence-document"><a href="../index.html">Back to synthesis</a>'
        '<h1>Detailed evidence</h1><p>Verbatim committed records; historical'
        ' analysis is preserved. '
        'The current BDD tables use the exclusion sensitivity.</p><ul>'
        + ''.join(entries) + '</ul><a href="../evidence-lock.json">Source checksums</a>'
        '</main></body></html>')
    (out / 'evidence-lock.json').write_text(json.dumps(lock, indent=2) + '\n')


def report_tables(data):
    """LaTeX table generated from the same validated display data as the page."""
    lines = [r'\begin{table}[ht]', r'\centering\small', r'\resizebox{\textwidth}{!}{%',
             r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Representation & T$\to$I Hit@1 & T$\to$I Hit@10 & I$\to$T Hit@1 & I$\to$T'
              r' Hit@10 & Caption recall@10 \\',
             r'\midrule']
    for p in data['coco']:
        values = [p['retrieval']['t2i']['hit@1'], p['retrieval']['t2i']['hit@10'],
                  p['retrieval']['i2t']['hit@1'], p['retrieval']['i2t']['hit@10'],
                  p['retrieval']['i2t']['set_recall@10']]
        lines.append(p['condition'].replace('_', r'\_') + ' & '
                     + ' & '.join(f'{100*x:.2f}' for x in values) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}}',
                  r'\caption{\ours{} Recorded COCO endpoints (percent). One thousand image'
                   ' groups, five captions each. Role-specific text inputs differ by direction.}',
                  r'\end{table}'])
    return '\n'.join(lines) + '\n'


def report_evidence_tables(data):
    """Translate the static BDD evidence tables; share cells and precision with HTML."""
    from html.parser import HTMLParser

    class Table(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows = []
            self.caption = ''
            self.target = None

        def handle_starttag(self, tag, attrs):
            if tag == 'tr':
                self.rows.append([])
            if tag in ('td', 'th', 'caption'):
                self.target = tag
                if tag != 'caption':
                    self.rows[-1].append('')

        def handle_endtag(self, tag):
            if tag in ('td', 'th', 'caption'):
                self.target = None

        def handle_data(self, value):
            if self.target == 'caption':
                self.caption += value
            elif self.target:
                self.rows[-1][-1] += value

    def tex(value):
        return (value.replace('_', r'\_').replace('%', r'\%')
                .replace('Δ', r'$\Delta$').replace('[ours]', r'\ours{}'))

    output = {}
    for name, markup in zip(('geometry', 'compression', 'numerics'), synthesis_tables(data)[:3]):
        table = Table()
        table.feed(markup)
        lines = [r'\begin{table}[ht]', r'\centering\small',
                 r'\resizebox{\textwidth}{!}{%',
                 r'\begin{tabular}{l' + 'r' * (len(table.rows[0])-1) + '}', r'\toprule']
        for i, row in enumerate(table.rows):
            lines.append(' & '.join(tex(cell) for cell in row) + r' \\')
            if i == 0:
                lines.append(r'\midrule')
        lines.extend([r'\bottomrule', r'\end{tabular}}',
                      r'\caption{' + tex(table.caption) + '}', r'\end{table}'])
        output[name] = '\n'.join(lines) + '\n'
    return output

def build(experiments: Path, out: Path) -> None:
    data = collect(experiments)
    lock, sources, records = synthesis_sources()
    for name, raw in sources.items():
        if name.startswith("experiments/phaseB_s"):
            path = experiments / Path(name).relative_to("experiments")
            if path.read_bytes() != raw:
                raise ValueError(f"Accepted training evidence checksum mismatch: {path}")
    synthesis = synthesis_data(records)
    tables = synthesis_tables(synthesis)
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
        "__C1_TABLE__": public_table(tables[0]), "__C2_TABLE__": public_table(tables[1]),
        "__C3_TABLE__": public_table(tables[2]), "__COCO_TABLE__": public_table(tables[3]),
        "__COCO_FIGURE__": paired_figure(synthesis),
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
    replacements.update(figures.extension_figures(synthesis, json.loads(
        sources['experiments/phaseC_paired_replication/paired_intervals.json'])))
    replacements['__TRAINING_FIGURES__'] = figures.training_figures(data)
    page = (ROOT / "viz/site/index.html").read_text()
    for key, value in replacements.items():
        page = page.replace(key, value)
    # Public-page naming exception: labels remain in the research artifacts.
    for label in ('[ours] ', '[established] ', '[interpretation] ', '[replication] '):
        page = page.replace(label, '')
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "viz/site/assets", out / "assets", dirs_exist_ok=True)
    (out / "index.html").write_text(page)
    (out / "data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (out / "synthesis.json").write_text(json.dumps(synthesis, indent=2, allow_nan=False) + "\n")
    export_evidence(out, lock, sources)
    (out / ".nojekyll").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", type=Path, default=ROOT / "experiments")
    parser.add_argument("--out", type=Path, default=ROOT / "viz/dist")
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    build(args.experiments_dir, args.out)
    if args.report_out:
        _, _, records = synthesis_sources()
        data = synthesis_data(records)
        args.report_out.write_text(report_tables(data))
        for name, text in report_evidence_tables(data).items():
            (args.report_out.parent / f"final_{name}.tex").write_text(text)
    print(
        "Built final diagnostic-study project page with final geometry and corrected probes: "
        f"{args.out / 'index.html'}"
    )
