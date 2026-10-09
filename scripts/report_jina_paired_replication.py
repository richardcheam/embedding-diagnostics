"""Render frozen paired records; no encoder, image decoding or endpoint fitting."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from embedding_diagnostics.embedding_cache import atomic_json
from embedding_diagnostics.phase_c_numerics import resume_results
from embedding_diagnostics.source_sensitivity import digest, verify_hashes

ROOT = Path("experiments/phaseC_paired_jina_repair")


def floors(images, captions, positives):
    return {
        "t2i": {f"hit@{k}": k / images for k in (1, 5, 10)},
        "i2t": {
            f"hit@{k}": 1 - math.comb(captions - positives, k) / math.comb(captions, k)
            for k in (1, 5, 10)
        }
        | {"set_recall@10": 10 / captions},
    }


def changes(reference, condition):
    a, b = np.asarray(reference), np.asarray(condition)
    if a.shape != b.shape:
        raise ValueError("query alignment mismatch")
    return {
        "gain": int(np.sum(b > a)),
        "loss": int(np.sum(b < a)),
        "unchanged": int(np.sum(b == a)),
    }


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(str(v) for v in row) + " |" for row in rows],
        ]
    )


def report():
    identity = {
        "freeze_sha256": digest(ROOT / "freeze.json"),
        "cache_binding_sha256": digest(ROOT / "cache_binding.json"),
    }
    state = resume_results(ROOT / "results.json", identity)
    records = [r["payload"] for r in state["records"]]
    if [r["condition"] for r in records] != ["native_1024", "mrl_256", "mrl_128"]:
        raise ValueError("incomplete three-representation result")
    completion = json.loads((ROOT / "interval_completion.json").read_text())
    expected = (
        __import__("hashlib").sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    )
    if completion["identity"] != {"records_sha256": expected}:
        raise ValueError("interval source mismatch")
    verify_hashes(completion["files_sha256"])
    intervals = json.loads((ROOT / "paired_intervals.json").read_text())
    lines = [
        "# Frozen COCO paired-data results",
        "",
        "[ours] Measurements on 1,000 original-release COCO2017 validation image groups, "
        "five captions each. Second encoder on the exact previously examined evaluation sample. "
        "Both directions use the same plain-caption cache; task=None, trained default LoRA.",
        "",
        "## Recorded paired retrieval",
        "",
    ]
    rows = []
    for r in records:
        for d in ("t2i", "i2t"):
            p = r["retrieval"][d]
            rows.append(
                [
                    r["condition"],
                    d,
                    p["query_count"],
                    p["gallery_count"],
                    *[f"{100 * p[k]:.3f}" for k in ("hit@1", "hit@5", "hit@10", "set_recall@10")],
                    f"{p['overlap@10']:.5f}",
                    p["changed_queries"],
                    p["ties_at_k"],
                ]
            )
    lines += [
        table(
            [
                "representation",
                "direction",
                "queries",
                "gallery",
                "Hit@1 %",
                "Hit@5 %",
                "Hit@10 %",
                "set recall@10 %",
                "overlap@10",
                "changed identities",
                "boundary ties@1/5/10",
            ],
            rows,
        ),
        "",
        "[ours] Primary endpoint is T2I Hit@10. I2T Hit@10 and "
        "positive-set recall@10 are different multiple-positive endpoints.",
        "",
        "## Paired deltas and query changes",
        "",
    ]
    rows = []
    summaries = {}
    for r in records[1:]:
        summaries[r["condition"]] = {}
        for d in ("t2i", "i2t"):
            p = r["retrieval"][d]
            n = records[0]["retrieval"][d]
            ci = intervals[r["condition"]][d]
            change = changes(n["per_query"]["hit@10"], p["per_query"]["hit@10"])
            summaries[r["condition"]][d] = change
            rows.append(
                [
                    r["condition"],
                    d,
                    f"{100 * ci['delta']:+.3f}",
                    f"[{100 * ci['lower']:+.3f}, {100 * ci['upper']:+.3f}]",
                    *change.values(),
                    f"{100 * (p['set_recall@10'] - n['set_recall@10']):+.3f}",
                ]
            )
    lines += [
        table(
            [
                "representation",
                "direction",
                "Δ Hit@10 pp",
                "conditional 95% interval",
                "gained queries",
                "lost queries",
                "unchanged",
                "Δ set recall@10 pp",
            ],
            rows,
        ),
        "",
        "[ours] Intervals resample 1,000 parent-image units, carrying five captions "
        "and paired conditions together, conditional on the fixed gallery. "
        "Marginal percentile intervals; no equivalence test or encoder uncertainty.",
        "",
        "## Coarse category image-only comparator",
        "",
    ]
    rows = []
    n = records[0]["category"]["macro_precision@10"]
    for r in records:
        c = r["category"]
        rows.append(
            [
                r["condition"],
                len(c["eligible"]),
                f"{100 * c['macro_precision@10']:.3f}",
                f"{100 * (c['macro_precision@10'] - n):+.3f}",
                f"{100 * c['macro_chance']:.3f}",
            ]
        )
    lines += [
        table(
            [
                "representation",
                "eligible categories",
                "macro P@10 %",
                "Δ native pp",
                "macro chance %",
            ],
            rows,
        ),
        "",
        "[ours] Eligibility was frozen at >=10 member images. Each category queries "
        "its members against 999 other images; multi-category membership is preserved. "
        "This is coarse image-only category utility, not paired-caption relevance.",
        "",
    ]
    support = []
    names = {
        str(c["id"]): c["name"]
        for c in json.loads(
            Path("experiments/phaseC_paired_replication/acquisition.json").read_text()
        )["categories"]
    }
    for key, c in records[0]["category"]["categories"].items():
        support.append(
            [
                names[key],
                c["support"],
                key in map(str, records[0]["category"]["eligible"]),
                "unscorable" if c["chance"] is None else f"{100 * c['chance']:.3f}",
                *[
                    "unscorable"
                    if r["category"]["categories"][key]["precision"] is None
                    else f"{100 * r['category']['categories'][key]['precision']:.3f}"
                    for r in records
                ],
            ]
        )
    lines += [
        table(
            ["category", "support", "macro eligible", "chance %", "native %", "256 %", "128 %"],
            support,
        ),
        "",
        "## Marginal geometry (descriptive)",
        "",
    ]
    rows = []
    for r in records:
        for role, g in r["geometry"].items():
            rows.append(
                [
                    r["condition"],
                    role,
                    *[
                        f"{g[k]:.6f}"
                        for k in (
                            "total_variance",
                            "mean_pairwise_cosine",
                            "rankme",
                            "participation_ratio",
                            "rankme_fraction",
                            "participation_ratio_fraction",
                        )
                    ],
                ]
            )
    lines += [
        table(
            [
                "representation",
                "role",
                "variance",
                "mean cosine",
                "RankMe",
                "participation ratio",
                "RankMe/D",
                "PR/D",
            ],
            rows,
        ),
        "",
        "[interpretation] Separate marginal geometry is not paired alignment. "
        "Three dimensions do not validate a general utility predictor.",
        "",
        "## Native relevance permutation and random-order floors",
        "",
    ]
    baseline = floors(1000, 5000, 5)
    rows = []
    for d in ("t2i", "i2t"):
        p = records[0]["pairing_control"][d]
        rows.append(
            [
                d,
                *[f"{100 * p[k]:.3f}" for k in ("hit@1", "hit@5", "hit@10")],
                f"{100 * p['set_recall@10']:.3f}",
                f"{100 * baseline[d]['hit@10']:.3f}",
            ]
        )
    lines += [
        table(
            [
                "direction",
                "scrambled Hit@1 %",
                "Hit@5 %",
                "Hit@10 %",
                "set recall@10 %",
                "random-order Hit@10 %",
            ],
            rows,
        ),
        "",
        "[ours] One fixed one-parent circular shift changes recorded relevance only, "
        "preserving vectors, rankings and geometry. Floors are combinatorial "
        "gallery/positive-count references, not realistic irrelevant-caption guarantees.",
        "",
    ]
    oldroot = Path("experiments/phaseC_paired_replication")
    oldidentity = {
        "freeze_sha256": digest(oldroot / "freeze.json"),
        "cache_binding_sha256": digest(oldroot / "cache_binding.json"),
    }
    previous = [
        r["payload"] for r in resume_results(oldroot / "results.json", oldidentity)["records"]
    ]
    cross = []
    for j, g in zip(records[1:], previous[1:], strict=True):
        for d in ("t2i", "i2t"):
            for endpoint in ("hit@1", "hit@5", "hit@10", "set_recall@10"):
                jd = 100 * (j["retrieval"][d][endpoint] - records[0]["retrieval"][d][endpoint])
                gd = 100 * (g["retrieval"][d][endpoint] - previous[0]["retrieval"][d][endpoint])
                cross.append(
                    [
                        j["condition"],
                        d,
                        endpoint,
                        f"{jd:+.3f}",
                        f"{gd:+.3f}",
                    ]
                )
    lines += [
        "## Descriptive within-encoder response comparison",
        "",
        table(["prefix", "direction", "endpoint", "Jina Δ native pp", "Gemma Δ native pp"], cross),
        "",
        "[interpretation] Jina compression ratios are 4×/8×; Gemma ratios are 3×/6×. "
        "Input resolution, trained representations, text roles and normalization differ. "
        "These paired responses cannot isolate architectural or training causes. "
        "The same examined sample does not provide independent evaluation sampling.",
        "",
    ]
    atomic_json(
        ROOT / "report_summary.json",
        {
            "random_order_floors": baseline,
            "query_changes": summaries,
            "result_sha256": digest(ROOT / "results.json"),
        },
    )
    (ROOT / "results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    report()
