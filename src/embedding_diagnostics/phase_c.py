"""C0 extraction boundary: local images -> canonical cache -> existing geometry.

External pretrained vectors are not training conditions. This module performs
no fitting, post-hoc degradation, or interpretation of representation quality.
The geometry implementation is shared with A/B (participation ratio is the
covariance eigenvalue ratio variant, distinct from raw-matrix RankMe).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import resource
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from .diagnostics.metrics import collapse_metrics
from .embedding_cache import EmbeddingCache, load_cache


def code_provenance() -> dict:
    """Git identity plus source hashes, including uncommitted extraction code."""
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    sources = ["src/embedding_diagnostics/bdd100k.py", "src/embedding_diagnostics/bdd_lance.py",
               "src/embedding_diagnostics/models/embeddinggemma2.py",
               "src/embedding_diagnostics/embedding_cache.py",
               "src/embedding_diagnostics/phase_c.py",
               "src/embedding_diagnostics/diagnostics/metrics.py", "scripts/extract_phase_c.py",
               "pyproject.toml", "uv.lock"]
    return {"commit": commit, "source_sha256": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources
    }, "worktree_dirty": bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True,
    ).strip())}


def geometry_report(root: str | Path) -> dict:
    """Validate and reload the cache before applying unchanged A/B diagnostics."""
    matrix, manifest = load_cache(root)
    norms = np.linalg.norm(matrix, axis=1)
    return {"claim_label": "[ours]", "phase": "C0", "role": "integration smoke",
            "shape": list(matrix.shape), "all_finite": bool(np.isfinite(matrix).all()),
            "norm_range": [float(norms.min()), float(norms.max())],
            "image_ids": [r["image_id"] for r in manifest["selected_rows"]],
            "splits": {split: sum(r["split"] == split for r in manifest["selected_rows"])
                       for split in ("train", "val")},
            "geometry": collapse_metrics(matrix),
            "cache_bytes": sum(p.stat().st_size for p in Path(root).iterdir() if p.is_file()),
            "manifest": str(Path(root).resolve() / "manifest.json"),
            "provenance": manifest["provenance"]}


def extract(dataset, rows: list[dict], encoder, root: str | Path, *, batch_size: int = 1,
            chunk_size: int = 4, code_identity: dict | None = None,
            verify_repeat: bool = False, selection_procedure: str | None = None) -> dict:
    """Checkpoint complete chunks; optionally re-extract every selected image.

    Repeat tolerance is fixed before extraction: rtol=1e-5, atol=1e-6 on
    normalized FP32 outputs. Failed decodes/inference propagate with image IDs;
    examples are never silently discarded. No validation labels are fitted.
    """
    if batch_size < 1 or chunk_size < 1:
        raise ValueError("batch and chunk sizes must be positive")
    provenance = encoder.provenance | {
        "dataset": dataset.identity, "selection_procedure":
        selection_procedure or
        "first N fully-labelled lexicographic image IDs per split; global image_id ordering (C0)",
        "dataset_splits": sorted({r["split"] for r in rows}),
        "attribute_policy": "Phase B canonical vocab; unknown=-1; fully-labelled selection",
        "extraction_batch_size": batch_size, "chunk_size": chunk_size,
        "transformers_version": importlib.metadata.version("transformers"),
        "torch_version": torch.__version__, "lance_version": importlib.metadata.version("pylance"),
        "preprocessing_versions": {name: importlib.metadata.version(name) for name in
                                   ("pillow", "torchvision", "tokenizers", "numpy")},
        "code": code_identity if code_identity is not None else code_provenance(),
        "torch_cpu_threads": torch.get_num_threads(),
        "repeat_tolerance": {"rtol": 1e-5, "atol": 1e-6},
    }
    started = time.perf_counter()
    inference_seconds = 0.
    if encoder.provenance["device"] == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with EmbeddingCache(root, provenance, rows) as cache:
        initial = cache.completed
        # A local path/version can be reused after source replacement. Check
        # completed raw bytes before any new inference, even without repeat.
        completed_hashes = [h for c in cache.manifest["chunks"] for h in c["image_sha256"]]
        for start in range(0, initial, batch_size):
            stop = min(start + batch_size, initial)
            _, hashes = dataset.decode(rows[start:stop])
            if hashes != completed_hashes[start:stop]:
                raise ValueError(f"resume source image bytes changed at rows {start}:{stop}")
        while cache.completed < len(rows):
            selected = rows[cache.completed:cache.completed + chunk_size]
            arrays, hashes = [], []
            for start in range(0, len(selected), batch_size):
                batch = selected[start:start + batch_size]
                try:
                    images, digest = dataset.decode(batch)
                    tick = time.perf_counter()
                    arrays.append(encoder.encode(images))
                    inference_seconds += time.perf_counter() - tick
                    hashes.extend(digest)
                except Exception as error:
                    raise RuntimeError(f"extraction failed for {[r['image_id'] for r in batch]}") \
                        from error
            cache.append(np.concatenate(arrays), hashes)
            print(f"cached {cache.completed}/{len(rows)} images", flush=True)
    extraction_seconds = time.perf_counter() - started
    matrix, manifest = load_cache(root)
    maximum_error = None
    repeat_seconds = 0.
    if verify_repeat:
        tick = time.perf_counter()
        maximum_error = 0.
        expected_hashes = [h for c in manifest["chunks"] for h in c["image_sha256"]]
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            images, hashes = dataset.decode(batch)
            if hashes != expected_hashes[start:start + len(batch)]:
                raise ValueError("repeat image SHA256 mismatch")
            repeated = encoder.encode(images)
            original = matrix[start:start + len(batch)]
            if not np.allclose(original, repeated, rtol=1e-5, atol=1e-6):
                raise ValueError(f"repeat embeddings differ for {[r['image_id'] for r in batch]}")
            maximum_error = max(maximum_error, float(np.abs(original - repeated).max()))
        repeat_seconds = time.perf_counter() - tick
    report = geometry_report(root)
    report.update({"created_at": datetime.now(UTC).isoformat(),
                   "new_images": len(rows) - initial, "resumed_images": initial,
                   "extraction_seconds": extraction_seconds, "inference_seconds": inference_seconds,
                   "repeat_seconds": repeat_seconds,
                   "repeat_equivalent": True if verify_repeat else None,
                   "repeat_count": len(rows) if verify_repeat else 0,
                   "repeat_max_abs": maximum_error,
                   "peak_ram_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 2**20
                   if encoder.provenance["device"] == "cuda" else None,
                   "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / 2**20
                   if encoder.provenance["device"] == "cuda" else None})
    return report
