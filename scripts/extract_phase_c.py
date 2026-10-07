"""C0 local-only FP32 extraction; optionally benchmark CUDA in a separate process.

Run --device cpu first. A CUDA invocation uses the identical selected images
and must have its own cache directory; no device/dtype mixing on resume.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from embedding_diagnostics.bdd_lance import LanceBdd
from embedding_diagnostics.embedding_cache import atomic_json
from embedding_diagnostics.models.embeddinggemma2 import (
    MODEL_ID,
    MODEL_REVISION,
    EmbeddingGemma2,
    cuda_preflight,
)
from embedding_diagnostics.phase_c import code_provenance, extract, geometry_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--model", default=MODEL_ID, help="model ID or local snapshot directory")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--val-count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--verify-repeat", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    if args.diagnostics_only:
        report = geometry_report(args.cache)
    else:
        if args.dataset is None:
            parser.error("--dataset is required for extraction")
        if args.threads < 1 or args.batch_size < 1 or args.chunk_size < 1:
            parser.error("threads, batch size, and chunk size must be positive")
        if not 16 <= args.train_count + args.val_count <= 32:
            parser.error("C0 requires 16–32 images; freeze a separate C1 sampling protocol first")
        torch.set_num_threads(args.threads)
        started = time.perf_counter()
        dataset = LanceBdd(args.dataset)
        rows = sorted(dataset.select("train", args.train_count) +
                      dataset.select("val", args.val_count), key=lambda r: r["image_id"])
        cuda = cuda_preflight()
        print(json.dumps({"cuda_preflight": cuda}), flush=True)
        encoder = EmbeddingGemma2(args.model, revision=args.revision, device=args.device)
        load_seconds = time.perf_counter() - started
        report = extract(dataset, rows, encoder, args.cache, batch_size=args.batch_size,
                         chunk_size=args.chunk_size, verify_repeat=args.verify_repeat,
                         code_identity=code_provenance())
        report.update({"load_and_selection_seconds": load_seconds,
                       "total_wall_seconds": time.perf_counter() - started,
                       "cuda_preflight": cuda})
    atomic_json(args.result, report)
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
