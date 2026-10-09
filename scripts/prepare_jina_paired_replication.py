"""Jina input/runtime freeze; no encoder construction, forward, or endpoints."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from embedding_diagnostics.jina_protocol import (
    ACCEPTED,
    atomic_json,
    digest,
    read,
    validate_inputs,
    verify_hashes,
    versions,
)
from embedding_diagnostics.jina_protocol import (
    REPAIR_ROOT as ROOT,
)


def inputs():
    from embedding_diagnostics.models.jinaclipv2 import processor_tokenizer

    acquisition = read(ROOT / "acquisition.json")
    roots = acquisition["roots"]
    tokenizer, processor = processor_tokenizer(roots["jinaai/jina-clip-v2"], roots["resolved_clip"])
    sample = read(ACCEPTED / "sample_manifest.json")
    rows = {"image": [], "caption": []}
    for image in sample["selected_rows"]:
        rows["image"].append(
            {
                "id": image["image_id"],
                "parent_image_id": image["image_id"],
                "role": "image",
                "input_sha256": image["raw_sha256"],
            }
        )
        for caption in image["captions"]:
            text = caption["caption"]
            tokens = tokenizer(text, truncation=False)["input_ids"]
            if len(tokens) > 8194:
                raise ValueError("over-context caption; no truncation allowed")
            rows["caption"].append(
                {
                    "id": caption["id"],
                    "parent_image_id": image["image_id"],
                    "role": "caption",
                    "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "rendered": text,
                    "token_ids": tokens,
                }
            )
    validate_inputs(rows)
    atomic_json(ROOT / "input_manifest.json", rows)
    atomic_json(
        ROOT / "isolated_runtime.json",
        {
            "versions": versions(),
            "tokenizer_class": type(tokenizer).__name__,
            "processor": processor.to_dict(),
            "encoder_forwards": 0,
        },
    )


def freeze():
    if (ROOT / "freeze.json").exists():
        raise ValueError("existing freeze cannot be overwritten")
    if subprocess.check_output(["git", "status", "--porcelain"]).strip():
        raise ValueError("commit implementation/input preparation before freeze")
    old = read(ACCEPTED / "freeze.json")
    protected = dict(old["protected_sha256"])
    protected.update(read(ACCEPTED / "cache_binding.json")["files_sha256"])
    paths = subprocess.check_output(["git", "ls-files", "experiments"]).decode().splitlines()
    protected.update({p: digest(p) for p in paths if not p.startswith(str(ROOT) + "/")})
    failed = Path("experiments/phaseC_paired_jina_replication")
    protected.update({str(p): digest(p) for p in failed.glob("*/embedding-cache/manifest.json")})
    protected.update({p: digest(p) for p in ("pyproject.toml", "uv.lock")})
    verify_hashes(protected)
    acquisition = read(ROOT / "acquisition.json")
    model = {p: r["sha256"] for p, r in acquisition["files"].items()}
    resolved = Path(acquisition["roots"]["resolved_clip"])
    model.update({str(p): digest(p) for p in resolved.glob("*.py")})
    verify_hashes(model)
    code = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "src",
                "scripts",
                "tests",
                "tools/jina_extraction",
                "docs/second-encoder-replication-proposal.md",
                "docs/second-encoder-replication-plan.md",
                "docs/second-encoder-source-review.json",
                "docs/jina-tokenizer-repair-protocol.md",
            ]
        )
        .decode()
        .splitlines()
    )
    source = dict(old["source_sha256"])
    extra = [
        "acquisition.json",
        "resolution_patch.json",
        "checkpoint_inventory.json",
        "isolated_runtime.json",
        "input_manifest.json",
        "tokenizer_acquisition.json",
        "resolution_reproduction.json",
        "resolution_verification.json",
        "inventory_reconciliation.json",
    ]
    code_hashes = {p: digest(p) for p in code}
    code_hashes.update({str(ROOT / n): digest(ROOT / n) for n in extra})
    atomic_json(
        ROOT / "freeze.json",
        {
            "accepted_commit": "ebe17fb",
            "supersedes_attempt_commit": "56e9765",
            "supersedes_freeze_commit": "8557b78",
            "supersedes_freeze_sha256": digest(
                Path("experiments/phaseC_paired_jina_replication/freeze.json")
            ),
            "implementation_commit": subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode()
            .strip(),
            "repositories": {
                r: e["revision"]
                for r, e in read("docs/second-encoder-source-review.json")["repositories"].items()
            },
            "model_roots": acquisition["roots"],
            "code_sha256": code_hashes,
            "protected_sha256": protected,
            "model_sha256": model,
            "source_sha256": source,
            "runtime_versions": {
                "analysis": versions(),
                "extraction": read(ROOT / "isolated_runtime.json")["versions"],
            },
            "conditions": ["native_1024", "mrl_256", "mrl_128"],
            "sample_sha256": digest(ACCEPTED / "sample_manifest.json"),
            "input_sha256": digest(ROOT / "input_manifest.json"),
            "configuration": {
                "dtype": "float32",
                "device": "cpu",
                "batch_size": 1,
                "threads": 4,
                "load_mode": "official complete wrapper",
                "text_task": None,
                "default_lora": "retrieval.query",
                "image_size": 512,
                "dimensions": [1024, 256, 128],
                "canonical_normalization": "official FP32 L2",
                "prefix_normalization": "FP64 L2",
                "score": "FP64 normalize/dot; block64; numeric-ID ties",
                "interval": "5000 parent-unit PCG64 bootstrap, seed20261010, fixed gallery",
                "resource_gate": {
                    "rss_mib": 4608,
                    "swap_io_bytes": 256 * 2**20,
                    "consecutive_windows": 2,
                    "window_seconds": 60,
                },
                "smoke": {
                    "groups": 16,
                    "captions": 80,
                    "prefix_parity": "first image and caption at256/128",
                    "independent_repeat": "first image and caption",
                    "rtol": 1e-5,
                    "atol": 1e-6,
                    "scientific_endpoints": False,
                },
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("inputs", "freeze"))
    stage = parser.parse_args().stage
    if stage == "inputs":
        inputs()
    else:
        freeze()
