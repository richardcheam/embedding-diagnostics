"""Freeze/run post-audit sensitivity only; all historical artifacts are read-only."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import resource
import runpy
import socket
import subprocess
import time
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from embedding_diagnostics import phase_c_compression as compression
from embedding_diagnostics import phase_c_main as main
from embedding_diagnostics.bdd100k import ATTRIBUTE_VOCAB
from embedding_diagnostics.diagnostics.probe import ProbeConfig
from embedding_diagnostics.embedding_cache import atomic_json
from embedding_diagnostics.phase_c_numerics import (
    CONDITIONS,
    TrainInt8,
    arithmetic_audit,
    evaluate_condition,
    payload_hash,
    run_conditions,
)
from embedding_diagnostics.phase_c_pilot import fitted_transform
from embedding_diagnostics.source_sensitivity import (
    add_support_context,
    array_digest,
    digest,
    exclude_rows,
    gallery_scores,
    rebind_fit_history,
    refit_probe,
    verify_hashes,
)

ROOT = Path("experiments/phaseC_source_sensitivity")
ACCEPTED = "4343421"
SOURCE = Path("experiments/phaseC_c1_main")
AUDIT = Path("experiments/phaseC_source_audit")
PCA_PATH = Path("experiments/phaseC_c2/transform-cache/train_pca.npz")
C3_RUNNER = runpy.run_path("scripts/run_c3.py")
VERSIONS = C3_RUNNER["VERSIONS"]
FILES = sorted(
    {str(p) for p in Path("src/embedding_diagnostics").rglob("*.py")}
    | {
        "scripts/run_source_sensitivity.py",
        "scripts/report_source_sensitivity.py",
        "scripts/run_c1_main.py",
        "scripts/run_c2.py",
        "scripts/run_c3.py",
        "tests/test_source_sensitivity.py",
        "docs/source-sensitivity-protocol.md",
        "docs/source-sensitivity-plan.md",
        "docs/source-provenance-audit.md",
        "docs/c1-main-protocol.md",
        "docs/c2-protocol.md",
        "docs/c3-protocol.md",
        "pyproject.toml",
        "uv.lock",
    }
)


def read(path):
    return json.loads(Path(path).read_text())


def committed(path, commit="HEAD"):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"])


def historical_hashes():
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", ACCEPTED, "experiments"], text=True
    ).splitlines()
    hashes = {}
    for p in paths:
        if Path(p).read_bytes() != committed(p, ACCEPTED):
            raise ValueError(f"accepted historical artifact changed: {p}")
        hashes[p] = digest(p)
    return hashes


def source():
    matrix, sample, train, val, rows = C3_RUNNER["source_cache"]()
    flagged = read(AUDIT / "flagged_rows.json")["rows"]
    ids = [r["image_id"] for r in flagged]
    if len(ids) != 17 or any(r["evidence_classification"] != 4 for r in flagged):
        raise ValueError("expected exact 17 audited content-confirmed exclusions")
    a, b, reduced, keep = exclude_rows(matrix, sample, ids)
    if (len(a), len(b)) != (2000, 983) or a.tobytes() != train.tobytes():
        raise ValueError("fixed sensitivity split or training bytes changed")
    ledger = read(AUDIT / "selected_row_ledger.json")
    if [r["image_id"] for r in ledger] != [r["image_id"] for r in sample["selected_rows"]]:
        raise ValueError("audit ledger alignment mismatch")
    retained_ids = {r["image_id"] for r in reduced["selected_rows"]}
    if any(r["source_fragment"] not in (0, 1) for r in ledger if r["image_id"] in retained_ids):
        raise ValueError("retained source fragment mismatch")
    return matrix, sample, train, val, rows, reduced, keep, ids


def pca_parameters():
    with np.load(PCA_PATH, allow_pickle=False) as saved:
        params = {name: saved[name].copy() for name in saved.files}
    expected = read("experiments/phaseC_c2/results.json")["identity"]["PCA_parameter_sha256"]
    if {name: array_digest(x) for name, x in params.items()} != expected:
        raise ValueError("accepted training PCA parameter mismatch")
    return compression.TrainPCA(
        params["mean"], params["components"], params["explained_variance"], 2000
    )


def definitions():
    stress = read(SOURCE / "stress_results.json")["records"]
    c2 = read("experiments/phaseC_c2/results.json")["records"]
    return (
        [
            {
                "condition": "c1_pristine",
                "campaign": "C1",
                "original": read(SOURCE / "pristine_metrics.json"),
            }
        ]
        + [
            {
                "condition": f"c1_{r['transform']}_{r['severity']:g}",
                "campaign": "C1",
                "transform": r["transform"],
                "severity": r["severity"],
                "seed": 0,
                "original": r,
            }
            for r in stress
        ]
        + [
            {
                "condition": "c2_" + r["representation"],
                "campaign": "C2",
                "representation": r["representation"],
                "original": r,
            }
            for r in c2
        ]
        + [
            {"condition": "c3_" + name, "campaign": "C3", "source_condition": name}
            for name in CONDITIONS
        ]
    )


def vectors(definition, train, val, pca):
    if "transform" in definition:
        return fitted_transform(train, val, definition["transform"], definition["severity"], seed=0)
    if definition["campaign"] == "C2":
        method, dimension = definition["representation"].split("_")
        dimension = int(dimension)
        operation = pca.transform if method == "pca" else compression.matryoshka
        return operation(train, dimension), operation(val, dimension)
    if definition.get("source_condition", "").startswith("mean99"):
        return fitted_transform(train, val, "mean_injection", 0.99, seed=0)
    return train, val


def prepare_freeze():
    if (ROOT / "freeze.json").exists():
        raise ValueError("freeze exists; never overwrite")
    for p in FILES:
        if Path(p).read_bytes() != committed(p):
            raise ValueError(f"commit tested implementation/protocol first: {p}")
    history = historical_hashes()
    matrix, sample, train, val, rows, reduced, keep, ids = source()
    pca = pca_parameters()
    with threadpool_limits(limits=4):
        transformations = {}
        for d in definitions():
            a, b = vectors(d, train, val, pca)
            transformations[d["condition"]] = {
                "train_sha256": array_digest(a),
                "original_val_sha256": array_digest(b),
                "retained_val_sha256": array_digest(b[keep]),
                "dtype": str(b.dtype),
                "dimension": b.shape[1],
            }
        quantizer = TrainInt8.fit(train)
        c3 = read("experiments/phaseC_c3/freeze.json")
        if quantizer.scales.tolist() != c3["quantizer_scales"]:
            raise ValueError("historical calibration mismatch")
        runtime = C3_RUNNER["backend_identity"]()
        arithmetic = arithmetic_audit()
    retained = {
        "claim_label": "[ours]",
        "status": "FROZEN POST-AUDIT SENSITIVITY",
        "exclusion_ids": ids,
        "selected_rows": reduced["selected_rows"],
        "retained_validation_ids": [r["image_id"] for r, k in zip(rows, keep, strict=True) if k],
        "support": reduced["support"],
        "eligible_probe_class_codes": sample["eligible_probe_class_codes"],
        "eligible_probe_class_names": sample["eligible_probe_class_names"],
        "eligibility_rule": sample["eligibility_rule"],
        "source_fragments": {"train": 0, "val": 1},
        "qualification": "fragment provenance verified; not independently authenticated "
        "against original BDD release; near-duplicates untested",
    }
    atomic_json(ROOT / "sample_manifest.json", retained)
    atomic_json(
        ROOT / "freeze.json",
        {
            "claim_label": "[ours]",
            "created_at": datetime.now(UTC).isoformat(),
            "accepted_source_commit": subprocess.check_output(
                ["git", "rev-parse", ACCEPTED], text=True
            ).strip(),
            "implementation_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "files_sha256": {p: digest(p) for p in FILES},
            "historical_sha256": history,
            "canonical_matrix_sha256": array_digest(matrix),
            "train_matrix_sha256": array_digest(train),
            "canonical_cache_manifest_sha256": digest(SOURCE / "embedding-cache/manifest.json"),
            "canonical_sample_sha256": digest(SOURCE / "sample_manifest.json"),
            "sensitivity_sample_sha256": digest(ROOT / "sample_manifest.json"),
            "exclusion_ids": ids,
            "transformations": transformations,
            "condition_definitions": [
                {k: v for k, v in d.items() if k != "original"} for d in definitions()
            ],
            "PCA_file_sha256": digest(PCA_PATH),
            "PCA_parameter_sha256": read("experiments/phaseC_c2/results.json")["identity"][
                "PCA_parameter_sha256"
            ],
            "PCA_training_rows": 2000,
            "quantizer_scales": quantizer.scales.tolist(),
            "quantizer_scales_sha256": array_digest(quantizer.scales),
            "probe_configs": {
                "standardized": asdict(ProbeConfig()),
                "unscaled": asdict(main.RAW_CONFIG),
            },
            "probe_selection": "reuse historical train-only selected C and candidate diagnostics; "
            "refit final LBFGS/scaler on unchanged training inputs",
            "class_eligibility": sample["eligible_probe_class_codes"],
            "arithmetic_audit": arithmetic,
            "runtime_backend": runtime,
            "runtime_versions": {v: importlib.metadata.version(v) for v in VERSIONS},
            "threads": 4,
            "k": 10,
            "train_count": 2000,
            "validation_count": 983,
            "c1_c2_retrieval": "inherited FP64 norm/dot, argpartition, self excluded",
            "c3_rules": {
                k: c3[k] for k in ("conditions", "mean_stress", "ordering", "margin_bins")
            },
            "supplement": "fixed 983 retained queries against historical versus reduced gallery; "
            "C1/C2 rectangular retrieval, C3 historical per-query records",
            "status": "post-audit sensitivity; not independent replication",
        },
    )
    print("Prepared sample/freeze without scientific endpoints; commit both before running.")


def checked_freeze():
    for p in (ROOT / "freeze.json", ROOT / "sample_manifest.json"):
        if p.read_bytes() != committed(p):
            raise ValueError(f"freeze/sample must be committed unchanged: {p}")
    freeze = read(ROOT / "freeze.json")
    for section in ("files_sha256", "historical_sha256"):
        verify_hashes(freeze[section])
    verify_hashes(
        {
            str(ROOT / "sample_manifest.json"): freeze["sensitivity_sample_sha256"],
            str(PCA_PATH): freeze["PCA_file_sha256"],
            str(SOURCE / "embedding-cache/manifest.json"): freeze[
                "canonical_cache_manifest_sha256"
            ],
        }
    )
    for name, version in freeze["runtime_versions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("frozen runtime changed")
    return freeze


def forbidden(*args, **kwargs):
    raise RuntimeError("sensitivity forbids network, image decoding and model loading")


def run():
    started = time.perf_counter()
    freeze = checked_freeze()
    with ExitStack() as guards:
        guards.enter_context(patch.object(socket.socket, "connect", forbidden))
        guards.enter_context(patch.object(socket, "create_connection", forbidden))
        guards.enter_context(patch("PIL.Image.open", forbidden))
        guards.enter_context(
            patch(
                "embedding_diagnostics.models.embeddinggemma2.EmbeddingGemma2.__init__", forbidden
            )
        )
        guards.enter_context(threadpool_limits(limits=4))
        if (
            C3_RUNNER["backend_identity"]() != freeze["runtime_backend"]
            or arithmetic_audit() != freeze["arithmetic_audit"]
        ):
            raise ValueError("frozen backend/arithmetic changed")
        matrix, sample, train, val, rows, reduced, keep, ids = source()
        if array_digest(matrix) != freeze["canonical_matrix_sha256"]:
            raise ValueError("canonical matrix changed")
        pca, quantizer = pca_parameters(), TrainInt8.fit(train)
        if array_digest(quantizer.scales) != freeze["quantizer_scales_sha256"]:
            raise ValueError("calibration changed")
        identity = {
            "freeze_sha256": digest(ROOT / "freeze.json"),
            "implementation_commit": freeze["implementation_commit"],
        }
        by_name = {d["condition"]: d for d in definitions()}
        probe_cache = {}
        final_fits = 0

        def evaluate(name):
            nonlocal final_fits
            tick = time.perf_counter()
            d = by_name[name]
            a, b = vectors(d, train, val, pca)
            observed = {
                "train_sha256": array_digest(a),
                "original_val_sha256": array_digest(b),
                "retained_val_sha256": array_digest(b[keep]),
                "dtype": str(b.dtype),
                "dimension": b.shape[1],
            }
            if observed != freeze["transformations"][name]:
                raise ValueError("frozen transformation changed")
            if d["campaign"] == "C3":
                record = evaluate_condition(
                    d["source_condition"],
                    b[keep],
                    b[keep],
                    [r for r, k in zip(rows, keep, strict=True) if k],
                    quantizer,
                    k=10,
                )
                record["source_condition"] = record["condition"]
            else:
                original = d["original"]

                def reused_probe(x, y, z, labels, **options):
                    nonlocal final_fits
                    attr = next(
                        attr
                        for attr in ATTRIBUTE_VOCAB
                        if np.array_equal(
                            y, [r[attr] for r in sample["selected_rows"] if r["split"] == "train"]
                        )
                    )
                    variant = (
                        "probe_standardized"
                        if options.get("standardize", True)
                        else "probe_unscaled"
                    )
                    prior = original["attributes"][attr][variant]
                    key = (
                        array_digest(np.asarray(x, dtype=np.float64)),
                        array_digest(y),
                        array_digest(np.asarray(z, dtype=np.float64)),
                        array_digest(labels),
                        variant,
                        payload_hash(prior["selection_fits"]),
                        prior["selected_C"],
                        payload_hash(asdict(options.get("config", ProbeConfig()))),
                        tuple(options.get("eligible_classes", ())),
                        len(ATTRIBUTE_VOCAB[attr]),
                    )
                    reused = key in probe_cache
                    if key not in probe_cache:
                        probe_cache[key] = refit_probe(
                            x,
                            y,
                            z,
                            labels,
                            prior,
                            class_count=len(ATTRIBUTE_VOCAB[attr]),
                            **options,
                        )
                        final_fits += 1
                    return rebind_fit_history(probe_cache[key], prior, reused=reused)

                module = main if d["campaign"] == "C1" else compression
                evaluator = (
                    main.evaluate_main
                    if d["campaign"] == "C1"
                    else compression.evaluate_representation
                )
                with patch.object(module, "linear_probe_scores", reused_probe):
                    record = evaluator(a, b[keep], reduced)
                record["historical_gallery_retained_queries"] = gallery_scores(
                    b, rows, np.flatnonzero(keep)
                )
                reduced_rows = [r for r, k in zip(rows, keep, strict=True) if k]
                detail = gallery_scores(b[keep], reduced_rows, np.arange(int(keep.sum())))
                for attr, vocab in ATTRIBUTE_VOCAB.items():
                    labels = np.array([r[attr] for r in reduced_rows])
                    values = np.array(detail["attributes"][attr]["per_query_p10"])
                    if abs(values.mean() - record["attributes"][attr]["retrieval_p10"]) > 1e-12:
                        raise ValueError("supplement/accepted retrieval arithmetic disagreement")
                    record["attributes"][attr]["retrieval_per_class_all"] = {
                        name: {
                            "support": int(np.sum(labels == code)),
                            "p10": float(values[labels == code].mean())
                            if np.any(labels == code)
                            else None,
                            "macro_eligible": int(np.sum(labels == code)) >= 10,
                        }
                        for code, name in enumerate(vocab)
                    }
                del record["historical_gallery_retained_queries"]["neighbour_ids"]
                for attr in ATTRIBUTE_VOCAB:
                    old = original["attributes"][attr]["retrieval_p10"]
                    only = record["historical_gallery_retained_queries"]["attributes"][attr]["p10"]
                    new = record["attributes"][attr]["retrieval_p10"]
                    record["attributes"][attr]["gallery_decomposition"] = {
                        "historical_p10": old,
                        "retained_queries_historical_gallery_p10": only,
                        "query_removal_delta": only - old,
                        "gallery_removal_delta": new - only,
                    }
            add_support_context(record, reduced)
            record.update(
                condition=name, campaign=d["campaign"], wall_seconds=time.perf_counter() - tick
            )
            for key in ("transform", "severity", "representation"):
                if key in d:
                    record[key] = d[key]
            print(f"{name}: {record['wall_seconds']:.2f}s", flush=True)
            return record

        conditions = tuple(d["condition"] for d in freeze["condition_definitions"])
        state, calls = run_conditions(ROOT / "results.json", identity, conditions, evaluate)
        verify_hashes(freeze["historical_sha256"])
        final_matrix = C3_RUNNER["source_cache"]()[0]
        if array_digest(final_matrix) != freeze["canonical_matrix_sha256"]:
            raise ValueError("canonical cache changed during execution")
        atomic_json(
            ROOT / ("verification.json" if calls else "resume_verification.json"),
            {
                "claim_label": "[ours]",
                "identity": identity,
                "endpoint_calls": calls,
                "final_probe_fits_this_invocation": final_fits,
                "conditions": len(state["records"]),
                "wall_seconds": time.perf_counter() - started,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "historical_files_preserved": len(freeze["historical_sha256"]),
                "canonical_cache_preserved": True,
                "training_bytes_unchanged": True,
                "train_count": 2000,
                "validation_count": 983,
                "excluded_validation_count": 17,
                "guards": "network/image opening/encoder construction blocked",
                "protocol_deviations": [],
                "pilot_not_rerun": True,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    prepare_freeze() if args.freeze else run()
