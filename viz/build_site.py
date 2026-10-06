"""Build the public case study from final Phase-B logs and corrected probe records.

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
            values["original_probe_timeofday"] = original["probe_accuracy_unscaled_timeofday"]
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


def evidence_table(data: dict) -> str:
    rows = []
    for condition in CONDITIONS:
        values = [
            f"{mean(data, condition, 'total_variance'):.3g}",
            f"{mean(data, condition, 'rankme'):.2f}",
            f"{100 * mean(data, condition, 'probe_accuracy_unscaled_timeofday'):.2f}%",
            f"{100 * mean(data, condition, 'retrieval_p10_timeofday'):.2f}%",
        ]
        rows.append(
            f'<tr><th scope="row">{html.escape(LABELS[condition])}</th>'
            + "".join(f"<td>{value}</td>" for value in values) + "</tr>"
        )
    return "\n".join(rows)


def build(experiments: Path, out: Path) -> None:
    data = collect(experiments)
    control = "none_nostopgrad"
    warnings = sum(
        value == 0
        for columns in data["conditions"].values()
        for attribute in ATTRIBUTES
        for value in columns[f"probe_converged_unscaled_{attribute}"]
    )
    replacements = {
        "__OLD__": f"{100 * mean(data, control, 'original_probe_timeofday'):.2f}",
        "__NEW__": f"{100 * mean(data, control, 'probe_accuracy_unscaled_timeofday'):.2f}",
        "__VARIANCE__": f"{mean(data, control, 'total_variance'):.7f}",
        "__RANKME__": f"{mean(data, control, 'rankme'):.2f}",
        "__WARNINGS__": str(warnings),
        "__TABLE__": evidence_table(data),
    }
    for label, key in (("OLD", "original_probe_timeofday"),
                       ("NEW", "probe_accuracy_unscaled_timeofday")):
        height = 240 * mean(data, control, key)
        replacements[f"__{label}_HEIGHT__"] = f"{height:.3f}"
        replacements[f"__{label}_Y__"] = f"{295 - height:.3f}"
        replacements[f"__{label}_LABEL_Y__"] = f"{279 - height:.3f}"
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
    print(f"Built corrected Phase-B project page: {args.out / 'index.html'}")
