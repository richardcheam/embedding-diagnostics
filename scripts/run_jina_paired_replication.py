"""Pinned Jina replication; shared captions and unchanged accepted evaluator."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import resource
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from threadpoolctl import threadpool_limits

from embedding_diagnostics.jina_compression import jina_matryoshka
from embedding_diagnostics.jina_paired_cache import JinaPairedCache, load_jina_cache
from embedding_diagnostics.jina_protocol import (
    ACCEPTED,
    CONDITIONS,
    ROOT,
    SwapMonitor,
    atomic_json,
    cache_provenance,
    checked_freeze,
    digest,
    logical_views,
    read,
    resource_gate,
    verify_hashes,
)
from embedding_diagnostics.models.jinaclipv2 import JinaCLIPv2

SOURCE = Path("/mnt/hdd/data/datasets/COCO-original-2017")


def require_committed(path):
    if subprocess.check_output(["git", "show", f"HEAD:{path}"]) != path.read_bytes():
        raise ValueError("binding must be committed unchanged")


def cache_rows(rows):
    return [{k: r[k] for k in ("id", "parent_image_id", "role", "input_sha256")} for r in rows]


def validate_cache_identity(manifest, rows, provenance):
    if manifest["rows"] != rows or manifest["provenance"] != provenance:
        raise ValueError("canonical row alignment/inference provenance mismatch")


def finish_intervals(root, records, image_ids):
    from embedding_diagnostics.paired_evaluation import paired_interval

    identity = {
        "records_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    }
    path = root / "interval_completion.json"
    if path.exists():
        completion = read(path)
        if completion["identity"] != identity:
            raise ValueError("interval completion source mismatch")
        verify_hashes(completion["files_sha256"])
        return 0
    intervals = {}
    for record in records[1:]:
        intervals[record["condition"]] = {}
        for direction in ("t2i", "i2t"):
            a = records[0]["retrieval"][direction]["per_parent"]
            b = record["retrieval"][direction]["per_parent"]
            intervals[record["condition"]][direction] = paired_interval(
                [a[str(i)]["hit@10"] for i in image_ids], [b[str(i)]["hit@10"] for i in image_ids]
            )
    atomic_json(root / "paired_intervals.json", intervals)
    atomic_json(
        path,
        {
            "identity": identity,
            "files_sha256": {
                str(root / "paired_intervals.json"): digest(root / "paired_intervals.json")
            },
        },
    )
    return 4


def completed_records(root, identity):
    """Validate completed checksums/identity before any reference endpoint work."""
    path = root / "results.json"
    from embedding_diagnostics.phase_c_numerics import resume_results

    state = resume_results(path, identity)
    names = [r["payload"]["condition"] for r in state["records"]]
    if names != list(CONDITIONS[: len(names)]):
        raise ValueError("resumed condition order mismatch")
    return names == list(CONDITIONS)


def validate_saved_checks(checks):
    for check in checks.values():
        if (
            not check.get("equivalent")
            or set(check.get("prefixes", {})) != {"256", "128"}
            or not all(p.get("equivalent") for p in check["prefixes"].values())
        ):
            raise ValueError("saved failed smoke cannot authorize resume")


def extract(*, smoke=False):
    freeze = checked_freeze(ROOT, environment="extraction")
    for name in ("numerical_blocker.json", "resource_blocker.json", "compatibility_blocker.json"):
        if (ROOT / name).exists():
            raise ValueError("recorded blocker requires separately reviewed resolution")
    if not smoke:
        report = read(ROOT / "smoke_metrics.json")
        if not report.get("contract_passed") or not report.get("resource_gate_passed"):
            raise ValueError("passed smoke required before full extraction")
        validate_saved_checks(report.get("checks", {}))
        if set(report.get("checks", {})) != {"image", "caption"}:
            raise ValueError("complete numerical smoke checks required")
    inputs = read(ROOT / "input_manifest.json")
    targets = {r: (16 if r == "image" else 80) if smoke else len(v) for r, v in inputs.items()}
    started = time.perf_counter()
    model = None
    forward_calls = 0
    measures = {}
    monitor = SwapMonitor()
    windows = monitor.windows
    checks_path = ROOT / "smoke_checks.json"
    checks = {}
    if checks_path.exists():
        saved = read(checks_path)
        if saved["freeze_sha256"] != digest(ROOT / "freeze.json"):
            raise ValueError("smoke repeat freeze mismatch")
        checks = saved["checks"]
        validate_saved_checks(checks)
    torch.set_num_threads(4)

    def gate():
        now = time.perf_counter()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if not resource_gate(rss, windows):
            atomic_json(
                ROOT / "resource_blocker.json",
                {
                    "peak_rss_mib": rss,
                    "swap_windows": windows,
                    "forward_calls": forward_calls,
                    "elapsed_seconds": now - started,
                    "loading_seconds": loading_seconds,
                },
            )
            raise RuntimeError("frozen resource gate failed; no alternate loading authorized")
        return rss

    loading_seconds = 0
    with monitor, threadpool_limits(limits=4):
        for role in ("image", "caption"):
            phase = time.perf_counter()
            native_seconds = 0
            with JinaPairedCache(
                ROOT / role / "embedding-cache",
                cache_provenance(freeze, role, ROOT),
                cache_rows(inputs[role]),
            ) as cache:
                initial = cache.completed
                if smoke and cache.completed and role not in checks:
                    raise ValueError("persisted smoke entries without independent checks")
                for index in range(initial, targets[role]):
                    if model is None:
                        loading_start = time.perf_counter()
                        try:
                            model = JinaCLIPv2(
                                Path(freeze["model_roots"]["jinaai/jina-clip-v2"]),
                                freeze["model_roots"],
                            )
                        except Exception as exc:
                            atomic_json(
                                ROOT / "compatibility_blocker.json",
                                {
                                    "type": type(exc).__name__,
                                    "error": str(exc),
                                    "forward_calls": 0,
                                    "peak_rss_mib": resource.getrusage(
                                        resource.RUSAGE_SELF
                                    ).ru_maxrss
                                    / 1024,
                                    "elapsed_seconds": time.perf_counter() - started,
                                },
                            )
                            raise
                        loading_seconds = time.perf_counter() - loading_start
                        atomic_json(ROOT / "model_provenance.json", model.provenance)
                        gate()  # Loading peak must pass before any encoder forward.
                    row = inputs[role][index]
                    if role == "image":
                        source = read(ACCEPTED / "sample_manifest.json")["selected_rows"][index]
                        with zipfile.ZipFile(SOURCE / "val2017.zip") as archive:
                            raw = archive.read("val2017/" + source["file_name"])
                        if hashlib.sha256(raw).hexdigest() != row["input_sha256"]:
                            raise ValueError("source image byte continuity changed")
                        with Image.open(io.BytesIO(raw)) as image:
                            value = image.convert("RGB")
                        encode = model.encode_images
                    else:
                        value = row["rendered"]
                        if (
                            model.tokenizer(value, truncation=False)["input_ids"]
                            != row["token_ids"]
                        ):
                            raise ValueError("caption preprocessing changed")
                        encode = model.encode_captions
                    native_start = time.perf_counter()
                    x = encode([value])
                    native_seconds += time.perf_counter() - native_start
                    forward_calls += 1
                    gate()
                    if smoke and index == 0 and role not in checks:
                        repeat = encode([value])
                        forward_calls += 1
                        gate()
                        parity = {}
                        for d in (256, 128):
                            direct = encode([value], dimension=d)
                            forward_calls += 1
                            gate()
                            derived = jina_matryoshka(x, d)
                            parity[str(d)] = {
                                "maximum_difference": float(np.max(np.abs(direct - derived))),
                                "equivalent": bool(
                                    np.allclose(direct, derived, rtol=1e-5, atol=1e-6)
                                ),
                            }
                        checks[role] = {
                            "repeat_difference": float(np.max(np.abs(x - repeat))),
                            "equivalent": bool(np.allclose(x, repeat, rtol=1e-5, atol=1e-6)),
                            "prefixes": parity,
                        }
                        atomic_json(
                            checks_path,
                            {"freeze_sha256": digest(ROOT / "freeze.json"), "checks": checks},
                        )
                        if not checks[role]["equivalent"] or not all(
                            p["equivalent"] for p in parity.values()
                        ):
                            raise ValueError("frozen numerical contract failed")
                    if model.activation_dtypes != {"torch.float32"}:
                        atomic_json(
                            ROOT / "numerical_blocker.json",
                            {
                                "activation_dtypes": sorted(model.activation_dtypes),
                                "forward_calls": forward_calls,
                            },
                        )
                        raise ValueError("non-FP32 activation output before canonical commit")
                    cache.append(x)
                    if cache.completed % 16 == 0:
                        print(
                            f"{role} {cache.completed}/{targets[role]} RSS {gate():.1f}MiB",
                            flush=True,
                        )
                elapsed = time.perf_counter() - phase
                new = cache.completed - initial
                measures[role] = {
                    "completed": cache.completed,
                    "new_rows": new,
                    "seconds": elapsed,
                    "native_inference_seconds": native_seconds,
                    "seconds_per_new_row": native_seconds / new if new else None,
                    "images_or_captions_per_second": new / native_seconds
                    if native_seconds
                    else None,
                }
    if model is not None and model.activation_dtypes != {"torch.float32"}:
        raise ValueError(f"non-FP32 activation output: {model.activation_dtypes}")
    report = {
        "roles": measures,
        "new_inferences": sum(m["new_rows"] for m in measures.values()),
        "forward_calls": forward_calls,
        "model_loaded": model is not None,
        "wall_seconds": time.perf_counter() - started,
        "loading_seconds": loading_seconds,
        "peak_rss_mib": gate(),
        "swap_windows": windows,
        "resource_gate_passed": True,
        "contract_passed": all(r in checks for r in ("image", "caption")),
        "checks": checks,
        "activation_dtypes": sorted(model.activation_dtypes) if model else [],
        "scientific_endpoints": False,
    }
    if smoke:
        report["projected_extraction_seconds"] = sum(
            len(inputs[r]) * m["seconds_per_new_row"]
            for r, m in measures.items()
            if m["seconds_per_new_row"] is not None
        )
    name = "smoke" if smoke else "extraction"
    path = ROOT / f"{name}_metrics.json"
    if path.exists():
        path = ROOT / f"{name}_resume.json"
    atomic_json(path, report)
    verify_hashes(freeze.get("protected_sha256", {}))
    print(json.dumps(report), flush=True)


def bind_cache():
    freeze = checked_freeze(ROOT, environment="analysis")
    files, shapes = {}, {}
    for role in ("image", "caption"):
        root = ROOT / f"{role}/embedding-cache"
        x, manifest = load_jina_cache(root)
        validate_cache_identity(
            manifest,
            cache_rows(read(ROOT / "input_manifest.json")[role]),
            cache_provenance(freeze, role, ROOT),
        )
        files[str(root / "manifest.json")] = digest(root / "manifest.json")
        for c in manifest["chunks"]:
            files[str(root / c["file"])] = c["sha256"]
        shapes[role] = list(x.shape)
    binding = {
        "freeze_sha256": digest(ROOT / "freeze.json"),
        "files_sha256": files,
        "shapes": shapes,
        "status": "canonical cache binding before scientific endpoints",
    }
    path = ROOT / "cache_binding.json"
    if path.exists() and read(path) != binding:
        raise ValueError("existing canonical binding mismatch")
    atomic_json(path, binding)
    verify_hashes(freeze["protected_sha256"])


def evaluate():
    from embedding_diagnostics.paired_evaluation import category_scores, paired_scores, rankings
    from embedding_diagnostics.phase_c_compression import geometry_with_capacity
    from embedding_diagnostics.phase_c_numerics import resume_results, run_conditions

    freeze = checked_freeze(ROOT, environment="analysis")
    binding = read(ROOT / "cache_binding.json")
    if binding["freeze_sha256"] != digest(ROOT / "freeze.json"):
        raise ValueError("canonical binding/freeze mismatch")
    require_committed(ROOT / "cache_binding.json")
    verify_hashes(binding["files_sha256"])
    identity = {
        "freeze_sha256": digest(ROOT / "freeze.json"),
        "cache_binding_sha256": digest(ROOT / "cache_binding.json"),
    }
    if completed_records(ROOT, identity):
        state = resume_results(ROOT / "results.json", identity)
        interval_calls = finish_intervals(
            ROOT,
            [r["payload"] for r in state["records"]],
            [r["id"] for r in read(ROOT / "input_manifest.json")["image"]],
        )
        atomic_json(
            ROOT / "endpoint_resume.json",
            {
                "endpoint_calls": 0,
                "interval_calls": interval_calls,
                "reference_rankings": 0,
                "protected_files": len(freeze["protected_sha256"]),
                "cache_preserved": True,
            },
        )
        return
    source_inputs = read(ROOT / "input_manifest.json")
    image = load_jina_cache(ROOT / "image/embedding-cache")[0]
    caption = load_jina_cache(ROOT / "caption/embedding-cache")[0]
    matrices, inputs = logical_views(image, caption, source_inputs)
    sample = read(ACCEPTED / "sample_manifest.json")
    ids = {r: [v["id"] for v in inputs[r]] for r in matrices}
    parents = {r: [v["parent_image_id"] for v in inputs[r]] for r in matrices}
    native_top = {}
    with threadpool_limits(limits=4):
        for direction, q, g in [("t2i", "query", "image"), ("i2t", "image", "document")]:
            native_top[direction] = rankings(matrices[q], matrices[g], ids[q], ids[g])[0]
    started = time.perf_counter()

    def condition(name):
        dimension = int(name.split("_")[-1])
        derived = {r: jina_matryoshka(x, dimension) for r, x in matrices.items()}
        result = {"condition": name, "dimension": dimension, "geometry": {}, "retrieval": {}}
        for role, x in derived.items():
            result["geometry"][role] = geometry_with_capacity(x)
        for direction, q, g in [("t2i", "query", "image"), ("i2t", "image", "document")]:
            top, ties = rankings(derived[q], derived[g], ids[q], ids[g], tie_depths=(1, 5, 10))
            result["retrieval"][direction] = paired_scores(top, parents[q], parents[g])
            overlap = [
                len(set(a) & set(b)) / 10 for a, b in zip(top, native_top[direction], strict=True)
            ]
            result["retrieval"][direction].update(
                {
                    "overlap@10": float(np.mean(overlap)),
                    "changed_queries": sum(v < 1 for v in overlap),
                    "boundary_ties": ties["10"],
                    "ties_at_k": ties,
                    "top10_ids": np.asarray(ids[g])[top].tolist(),
                }
            )
            if name == "native_1024":
                rotate = dict(zip(ids["image"], ids["image"][1:] + ids["image"][:1], strict=True))
                scrambled = [rotate[v] for v in parents["query"]] if q == "query" else parents[q]
                scrambled_gallery = (
                    [rotate[v] for v in parents[g]] if g == "document" else (parents[g])
                )
                result.setdefault("pairing_control", {})[direction] = paired_scores(
                    top, scrambled, scrambled_gallery
                )
        top, ties = rankings(
            derived["image"], derived["image"], ids["image"], ids["image"], self_exclude=True
        )
        result["category"] = category_scores(
            top, [r["categories"] for r in sample["selected_rows"]], sample["category_ids"]
        )
        result["category"]["boundary_ties"] = ties
        return result

    with threadpool_limits(limits=4):
        state, calls = run_conditions(ROOT / "results.json", identity, CONDITIONS, condition)
    records = [r["payload"] for r in state["records"]]
    interval_calls = finish_intervals(ROOT, records, ids["image"])
    verify_hashes(freeze["protected_sha256"])
    verify_hashes(binding["files_sha256"])
    path = ROOT / ("verification.json" if calls else "endpoint_resume.json")
    atomic_json(
        path,
        {
            "endpoint_calls": calls,
            "interval_calls": interval_calls,
            "wall_seconds": time.perf_counter() - started,
            "protected_files": len(freeze["protected_sha256"]),
            "cache_preserved": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("smoke", "extract", "bind", "evaluate"))
    args = parser.parse_args()
    if args.stage in ("smoke", "extract"):
        extract(smoke=args.stage == "smoke")
    elif args.stage == "bind":
        bind_cache()
    else:
        evaluate()
