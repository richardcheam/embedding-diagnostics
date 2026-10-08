"""Render post-audit comparisons from checksummed records; no vector endpoints."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from embedding_diagnostics.bdd100k import ATTRIBUTE_VOCAB
from embedding_diagnostics.embedding_cache import atomic_json
from embedding_diagnostics.phase_c_numerics import resume_results
from embedding_diagnostics.source_sensitivity import digest

ROOT = Path("experiments/phaseC_source_sensitivity")
ATTRIBUTES = tuple(ATTRIBUTE_VOCAB)


def read(path):
    return json.loads(Path(path).read_text())


def table(headers, rows):
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(str(x) for x in row) + " |" for row in rows],
    ]


def value(x):
    return "unscorable" if x is None else f"{x:.6f}"


def comparison(old, new, old_base, new_base):
    return {
        "historical": old,
        "reduced": new,
        "absolute_change": new - old,
        "historical_effect": old - old_base,
        "reduced_effect": new - new_base,
        "effect_change": (new - new_base) - (old - old_base),
    }


def endpoints(record):
    output = {}
    for attr in ATTRIBUTES:
        s = record["attributes"][attr]
        if record.get("campaign") == "C3":
            scores = {
                "p10": s["p10"],
                "macro_p10": s["macro_p10_support10"],
                "chance_sum_p_squared": s["chance_sum_p_squared"],
                "chance_self_excluded": s["chance_self_excluded"],
            }
        else:
            scores = {
                "p10": s["retrieval_p10"],
                "macro_p10": s["retrieval_macro_p10"],
                "chance_sum_p_squared": s["retrieval_chance"],
                "majority_floor": s["probe_standardized"]["majority"],
                "balanced_constant_floor": s.get(
                    "balanced_constant_class_floor", 1 / len(s["eligible_probe_classes"])
                ),
            }
            for variant in ("probe_standardized", "probe_unscaled"):
                if variant in s:
                    for key in ("accuracy", "balanced_accuracy", "macro_f1"):
                        scores[f"{variant}_{key}"] = s[variant][key]
        output[attr] = scores
    return output


def build_comparisons(records, historical):
    """Absolute and baseline-relative changes; input records never mutated."""
    by_name = {r["condition"]: r for r in records}
    output = {}
    for name, record in by_name.items():
        baseline = (
            "c1_pristine"
            if record["campaign"] == "C1"
            else "c2_native_768"
            if record["campaign"] == "C2"
            else "c3_mean99_ref"
            if record["source_condition"].startswith("mean99")
            else "c3_native_ref"
        )
        old, new = endpoints(historical[name]), endpoints(record)
        old_base, new_base = endpoints(historical[baseline]), endpoints(by_name[baseline])
        output[name] = {
            "baseline": baseline,
            "attributes": {
                a: {
                    k: comparison(old[a][k], new[a][k], old_base[a][k], new_base[a][k])
                    for k in new[a]
                }
                for a in ATTRIBUTES
            },
        }
        if record["campaign"] == "C3":
            output[name]["query_counts"] = {
                "historical": len(historical[name]["query_ids"]),
                "reduced": len(record["query_ids"]),
            }
            output[name]["numerical_summary"] = {
                k: {
                    "historical": historical[name]["summary"][k],
                    "reduced": v,
                    "absolute_change": v - historical[name]["summary"][k]
                    if v is not None and historical[name]["summary"][k] is not None
                    else None,
                }
                for k, v in record["summary"].items()
            }
        if "geometry" in record:
            output[name]["validation_geometry"] = {
                k: comparison(
                    historical[name]["geometry"]["val"][k],
                    val,
                    historical[baseline]["geometry"]["val"][k],
                    by_name[baseline]["geometry"]["val"][k],
                )
                for k, val in record["geometry"]["val"].items()
            }
    return output


def historical_records():
    source = Path("experiments/phaseC_c1_main")
    out = {"c1_pristine": read(source / "pristine_metrics.json") | {"campaign": "C1"}}
    for r in read(source / "stress_results.json")["records"]:
        out[f"c1_{r['transform']}_{r['severity']:g}"] = r | {"campaign": "C1"}
    for r in read("experiments/phaseC_c2/results.json")["records"]:
        out["c2_" + r["representation"]] = r | {"campaign": "C2"}
    for e in read("experiments/phaseC_c3/results.json")["records"]:
        r = copy.deepcopy(e["payload"])
        r["campaign"], r["source_condition"] = "C3", r["condition"]
        out["c3_" + r["condition"]] = r
    return out


def render():
    freeze = read(ROOT / "freeze.json")
    state = read(ROOT / "results.json")
    if digest(ROOT / "freeze.json") != state["identity"]["freeze_sha256"]:
        raise ValueError("report freeze identity mismatch")
    state = resume_results(ROOT / "results.json", state["identity"])
    records = [e["payload"] for e in state["records"]]
    if [r["condition"] for r in records] != [
        d["condition"] for d in freeze["condition_definitions"]
    ]:
        raise ValueError("report requires all frozen records in order")
    historical = historical_records()
    effects = build_comparisons(records, historical)
    by_name = {r["condition"]: r for r in records}
    matched = {}
    for d in (512, 256, 128):
        m, p = f"c2_mrl_{d}", f"c2_pca_{d}"
        matched[str(d)] = {
            a: {
                k: comparison(
                    endpoints(historical[m])[a][k],
                    endpoints(by_name[m])[a][k],
                    endpoints(historical[p])[a][k],
                    endpoints(by_name[p])[a][k],
                )
                for k in endpoints(by_name[m])[a]
            }
            for a in ATTRIBUTES
        }
    atomic_json(
        ROOT / "comparisons.json",
        {
            "claim_label": "[ours]",
            "identity": state["identity"],
            "conditions": effects,
            "MRL_minus_PCA": matched,
        },
    )
    header = [
        "# Post-audit source exclusion sensitivity",
        "",
        "[ours] Exact 17-row exclusion; unchanged 2,000 training vectors, 983 validation",
        "queries/gallery rows. This is a post-audit sensitivity, not independent replication.",
        "All tables are generated from frozen checksummed results. Per-class zero supports",
        "are unscorable; historical balanced-probe eligibility is unchanged.",
        "",
    ]
    lines = header + ["## Pristine endpoints", ""]
    base = by_name["c1_pristine"]
    rows = []
    for a in ATTRIBUTES:
        s = base["attributes"][a]
        p = s["probe_standardized"]
        rows.append(
            [
                a,
                value(p["accuracy"]),
                value(p["balanced_accuracy"]),
                value(s["retrieval_p10"]),
                value(s["retrieval_macro_p10"]),
                value(s["retrieval_chance"]),
                value(s["validation_majority_floor"]),
            ]
        )
    lines += table(
        [
            "Attribute",
            "Accuracy",
            "Balanced accuracy",
            "P@10",
            "Macro P@10",
            "sum(p²) floor",
            "Majority floor",
        ],
        rows,
    )
    lines += ["", "## All retained class supports", ""]
    rows = []
    for a in ATTRIBUTES:
        for name, s in base["attributes"][a]["class_support"].items():
            rows.append(
                [
                    a,
                    name,
                    s["train"],
                    historical["c1_pristine"]["support"]["val"][a][name],
                    s["val"],
                    s["balanced_probe_eligible"],
                    value(base["attributes"][a]["retrieval_per_class_all"][name]["p10"]),
                ]
            )
    lines += table(
        [
            "Attribute",
            "Class",
            "Train",
            "Historical val",
            "Retained val",
            "Fixed BA eligible",
            "P@10",
        ],
        rows,
    )
    lines += [
        "",
        "## Severe C1 conditions",
        "",
        "[ours] Values are reduced-sample condition-minus-pristine effects in percentage",
        "points; every severity and absolute/effect change is in comparisons.md/json.",
        "",
    ]
    rows = []
    for r in records:
        if r["campaign"] == "C1" and r.get("severity") == 0.99:
            for a in ATTRIBUTES:
                e = effects[r["condition"]]["attributes"][a]
                rows.append(
                    [
                        r["transform"],
                        a,
                        f"{100 * e['probe_standardized_balanced_accuracy']['reduced_effect']:+.4f}",
                        f"{100 * e['p10']['reduced_effect']:+.4f}",
                        f"{100 * e['probe_standardized_balanced_accuracy']['effect_change']:+.4f}",
                        f"{100 * e['p10']['effect_change']:+.4f}",
                    ]
                )
    lines += table(
        [
            "Transform (.99)",
            "Attribute",
            "BA effect pp",
            "P@10 effect pp",
            "BA effect change pp",
            "P@10 effect change pp",
        ],
        rows,
    )
    lines += ["", "## C2 reduced representations", ""]
    rows = []
    for r in records:
        if r["campaign"] == "C2":
            rows.append(
                [r["representation"]]
                + [
                    value(r["attributes"][a]["probe_standardized"]["balanced_accuracy"])
                    for a in ATTRIBUTES
                ]
                + [value(r["attributes"][a]["retrieval_p10"]) for a in ATTRIBUTES]
            )
    lines += table(
        [
            "Representation",
            "Weather BA",
            "Scene BA",
            "Time BA",
            "Weather P@10",
            "Scene P@10",
            "Time P@10",
        ],
        rows,
    )
    lines += ["", "## C3 error, identity and attribute effects", ""]
    rows = []
    for r in records:
        if r["campaign"] == "C3":
            s = r["summary"]
            rows.append(
                [
                    r["source_condition"],
                    f"{s['max_score_error']:.3e}",
                    f"{s['mean_overlap']:.6f}",
                    s["identity_changed_queries"],
                    s["prospective_sufficient_queries"],
                ]
                + [f"{100 * r['attributes'][a]['delta']:+.5f}" for a in ATTRIBUTES]
            )
    lines += table(
        [
            "Condition",
            "Max score error",
            "Overlap",
            "Changed / 983",
            "Prospective covered",
            "Weather delta pp",
            "Scene delta pp",
            "Time delta pp",
        ],
        rows,
    )
    verification = read(ROOT / "verification.json")
    lines += [
        "",
        "## Execution",
        "",
        "[ours] `" + json.dumps(verification, sort_keys=True) + "`.",
        "",
        "[interpretation] See interpretation.md for qualified findings. Absolute/effect",
        "changes are not equivalence tests. The source qualification remains conditional",
        "on enriched-fragment provenance; instance semantics and near-duplicates are untested.",
        "",
    ]
    (ROOT / "results.md").write_text("\n".join(lines))

    lines = header + ["## Absolute and effect changes", "", "[ours] Percentage points.", ""]
    rows = []
    for name, e in effects.items():
        for a, scores in e["attributes"].items():
            for endpoint, s in scores.items():
                rows.append(
                    [
                        name,
                        a,
                        endpoint,
                        value(s["historical"]),
                        value(s["reduced"]),
                        f"{100 * s['absolute_change']:+.4f}",
                        f"{100 * s['historical_effect']:+.4f}",
                        f"{100 * s['reduced_effect']:+.4f}",
                        f"{100 * s['effect_change']:+.4f}",
                    ]
                )
    lines += table(
        [
            "Condition",
            "Attribute",
            "Endpoint",
            "Historical",
            "Reduced",
            "Absolute Δ pp",
            "Old effect pp",
            "New effect pp",
            "Effect change pp",
        ],
        rows,
    )
    lines += ["", "## Validation geometry", "", "[ours] Raw units; every metric also in JSON.", ""]
    rows = []
    for name, e in effects.items():
        if "validation_geometry" in e:
            for k in ("total_variance", "mean_pairwise_cosine", "rankme", "participation_ratio"):
                s = e["validation_geometry"][k]
                rows.append(
                    [
                        name,
                        k,
                        value(s["historical"]),
                        value(s["reduced"]),
                        value(s["absolute_change"]),
                        value(s["effect_change"]),
                    ]
                )
    lines += table(
        ["Condition", "Metric", "Historical", "Reduced", "Absolute change", "Effect change"], rows
    )
    lines += [
        "",
        "## C3 numerical summary changes",
        "",
        "[ours] Counts retain their respective 1000/983 denominators; fixed-query identity",
        "comparisons are in gallery_diagnostic.md. Bounds remain empirical checks.",
        "",
    ]
    numeric_rows = []
    for name, e in effects.items():
        if "numerical_summary" in e:
            for metric, x in e["numerical_summary"].items():
                numeric_rows.append(
                    [
                        name,
                        metric,
                        value(x["historical"]),
                        value(x["reduced"]),
                        value(x["absolute_change"]),
                    ]
                )
    lines += table(
        ["Condition", "Metric", "Historical", "Reduced", "Absolute change"], numeric_rows
    )
    lines += ["", "## Matched-D MRL minus PCA", ""]
    rows = []
    for d, attrs in matched.items():
        for a, scores in attrs.items():
            for endpoint, s in scores.items():
                rows.append(
                    [
                        d,
                        a,
                        endpoint,
                        f"{100 * s['historical_effect']:+.4f}",
                        f"{100 * s['reduced_effect']:+.4f}",
                        f"{100 * s['effect_change']:+.4f}",
                    ]
                )
    lines += table(
        [
            "D",
            "Attribute",
            "Endpoint",
            "Old difference pp",
            "New difference pp",
            "Difference change pp",
        ],
        rows,
    )
    (ROOT / "comparisons.md").write_text("\n".join(lines) + "\n")

    lines = header + [
        "## Final refit diagnostics",
        "",
        "[ours] Original train-only selection diagnostics/C are reused; final models",
        "are refitted. Full inner candidate records remain in results.json.",
        "",
    ]
    rows = []
    for r in records:
        if r["campaign"] != "C3":
            for a in ATTRIBUTES:
                for variant in ("probe_standardized", "probe_unscaled"):
                    if variant in r["attributes"][a]:
                        p = r["attributes"][a][variant]
                        rows.append(
                            [
                                r["condition"],
                                a,
                                variant,
                                p["selected_C"],
                                p["grid_boundary"],
                                p["converged"],
                                p["n_iter"],
                                p["max_iter_reached"],
                                value(p["train_accuracy"]),
                                value(p["train_balanced_accuracy"]),
                                value(p["accuracy"]),
                                value(p["balanced_accuracy"]),
                                value(p["refit_provenance"]["train_accuracy_difference"]),
                            ]
                        )
    lines += table(
        [
            "Condition",
            "Attribute",
            "Probe",
            "C",
            "Boundary",
            "Converged",
            "Iterations",
            "max_iter",
            "Train acc",
            "Train BA",
            "Val acc",
            "Val BA",
            "Train acc Δ",
        ],
        rows,
    )
    (ROOT / "fit_diagnostics.md").write_text("\n".join(lines) + "\n")

    lines = header + [
        "## Fixed retained queries: query versus gallery removal",
        "",
        "[ours] Same 983 queries in middle/final columns. Effects use the relevant",
        "reference under each gallery. C3 old-gallery values reuse accepted per-query",
        "records, rather than recomputing old endpoints. Conditional chance is the",
        "retained-query mean (matching gallery labels minus self)/(gallery N minus 1).",
        "",
    ]
    rows = []
    for r in records:
        old = historical[r["condition"]]
        for a in ATTRIBUTES:
            if r["campaign"] == "C3":
                keep_ids = set(r["query_ids"])
                indices = [i for i, name in enumerate(old["query_ids"]) if name in keep_ids]
                old_all = old["attributes"][a]["p10"]
                only = float(np.mean(np.array(old["attributes"][a]["per_query"]["p10"])[indices]))
                new = r["attributes"][a]["p10"]
            else:
                s = r["attributes"][a]["gallery_decomposition"]
                old_all, only = s["historical_p10"], s["retained_queries_historical_gallery_p10"]
                new = r["attributes"][a]["retrieval_p10"]
            rows.append(
                [
                    r["condition"],
                    a,
                    value(old_all),
                    value(only),
                    value(new),
                    f"{100 * (only - old_all):+.5f}",
                    f"{100 * (new - only):+.5f}",
                ]
            )
    lines += table(
        [
            "Condition",
            "Attribute",
            "Historical 1000/1000",
            "Retained 983 / old 1000",
            "Reduced 983/983",
            "Query removal pp",
            "Gallery removal pp",
        ],
        rows,
    )
    lines += ["", "## Conditional retained-query chance", ""]
    sample_rows = read("experiments/phaseC_c1_main/sample_manifest.json")["selected_rows"]
    old_rows = [r for r in sample_rows if r["split"] == "val"]
    retained = set(read(ROOT / "sample_manifest.json")["retained_validation_ids"])
    chance_rows = []
    for a in ATTRIBUTES:
        old_labels = np.array([r[a] for r in old_rows])
        query_labels = np.array([r[a] for r in old_rows if r["image_id"] in retained])
        old_chance = np.mean(
            [(np.sum(old_labels == label) - 1) / (len(old_labels) - 1) for label in query_labels]
        )
        new_chance = np.mean(
            [
                (np.sum(query_labels == label) - 1) / (len(query_labels) - 1)
                for label in query_labels
            ]
        )
        chance_rows.append([a, value(old_chance), value(new_chance)])
    lines += table(
        ["Attribute", "983 queries / 1000 gallery", "983 queries / 983 gallery"], chance_rows
    )
    lines += ["", "## C3 identity effects with retained queries under both galleries", ""]
    rows = []
    for r in records:
        if r["campaign"] == "C3":
            old = historical[r["condition"]]
            ids = set(r["query_ids"])
            indices = [i for i, name in enumerate(old["query_ids"]) if name in ids]
            overlap = np.array(old["per_query"]["overlap"])[indices]
            rows.append(
                [
                    r["source_condition"],
                    float(overlap.mean()),
                    int(np.sum(overlap < 1)),
                    r["summary"]["mean_overlap"],
                    r["summary"]["identity_changed_queries"],
                ]
            )
    lines += table(
        [
            "Condition",
            "Old gallery overlap",
            "Old gallery changed / 983",
            "Reduced gallery overlap",
            "Reduced gallery changed / 983",
        ],
        rows,
    )
    (ROOT / "gallery_diagnostic.md").write_text("\n".join(lines) + "\n")

    lines = header + [
        "## C3 rebuilt margins and empirical sufficient checks",
        "",
        "[established] Strict gamma > 2 epsilon is a sufficient set-order condition;",
        "failure to satisfy it does not imply instability. Prospective reconstruction",
        "bounds are separate from retrospective score errors; neither is rigorous",
        "floating-point certification. Arithmetic-only prospective checks are n/a.",
        "",
    ]
    rows = []
    for r in records:
        if r["campaign"] == "C3":
            for b in r["bin_analysis"]:
                rows.append(
                    [
                        r["source_condition"],
                        b["bin"],
                        b["count"],
                        f"{b['minimum']:.3e} / {b['maximum']:.3e}",
                        value(b["mean_overlap"]),
                        b["changed_queries"],
                        b["prospective_sufficient"],
                        b["retrospective_sufficient"],
                    ]
                )
    lines += table(
        [
            "Condition",
            "Bin",
            "N",
            "Margin min/max",
            "Overlap",
            "Changed",
            "Prospective covered",
            "Retrospective covered",
        ],
        rows,
    )
    lines += ["", "## Attribute turnover, zero supports and floors", ""]
    rows = []
    for r in records:
        if r["campaign"] == "C3":
            for a in ATTRIBUTES:
                s = r["attributes"][a]
                rows.append(
                    [
                        r["source_condition"],
                        a,
                        s["gained_queries"],
                        s["lost_queries"],
                        s["unchanged_queries"],
                        s["identity_changed_p10_unchanged"],
                        s["identity_changed_label_counts_unchanged"],
                        value(s["p10"]),
                        value(s["macro_p10_support10"]),
                        value(s["chance_sum_p_squared"]),
                        value(s["chance_self_excluded"]),
                    ]
                )
    lines += table(
        [
            "Condition",
            "Attribute",
            "Gain",
            "Loss",
            "Unchanged",
            "Changed set/same P@10",
            "Changed set/same label counts",
            "P@10",
            "Macro",
            "sum(p²)",
            "Self-excluded chance",
        ],
        rows,
    )
    lines += ["", "## Numerical errors and exceptions", ""]
    rows = []
    for r in records:
        if r["campaign"] == "C3":
            s, c = r["summary"], r["clipping"]
            rows.append(
                [
                    r["source_condition"],
                    f"{s['mean_max_score_error']:.3e}",
                    f"{s['max_arithmetic_residual']:.3e}",
                    c["clipped_coordinates"],
                    c["clipped_rows"],
                    c["outside_calibration_coordinates"],
                    r["boundary_tie_count"],
                    s["prospective_empirical_violations"],
                    s["retrospective_sufficient_queries"],
                ]
            )
    lines += table(
        [
            "Condition",
            "Mean max error",
            "Arithmetic residual",
            "Clipped coords",
            "Clipped rows",
            "Outside range",
            "Ties",
            "Prospective exceptions",
            "Retrospective covered",
        ],
        rows,
    )
    (ROOT / "margin_analysis.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    render()
