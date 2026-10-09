"""Jina-only provenance guards and role aliases; no scientific endpoint definitions."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

from .embedding_cache import atomic_json as atomic_json

ROOT = Path("experiments/phaseC_paired_jina_replication")
ACCEPTED = Path("experiments/phaseC_paired_replication")
CONDITIONS = ("native_1024", "mrl_256", "mrl_128")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_hashes(hashes):
    for path, sha in hashes.items():
        if not Path(path).is_file() or digest(path) != sha:
            raise ValueError(f"protected file changed: {path}")


def validate_inputs(inputs):
    images, captions = inputs["image"], inputs["caption"]
    ids = [r["id"] for r in images]
    if not ids or ids != sorted(set(ids)):
        raise ValueError("unique ordered image IDs required")
    if len(captions) != 5 * len(images) or len({r["id"] for r in captions}) != len(captions):
        raise ValueError("exact five original captions per image required")
    if [r["parent_image_id"] for r in captions] != [i for i in ids for _ in range(5)]:
        raise ValueError("caption parent ordering mismatch")
    for start in range(0, len(captions), 5):
        group = [r["id"] for r in captions[start : start + 5]]
        if group != sorted(group):
            raise ValueError("original caption annotation ordering mismatch")


def logical_views(image, caption, inputs):
    validate_inputs(inputs)
    if len(image) != len(inputs["image"]) or len(caption) != len(inputs["caption"]):
        raise ValueError("matrix/source row alignment mismatch")
    return (
        {"image": image, "query": caption, "document": caption},
        {"image": inputs["image"], "query": inputs["caption"], "document": inputs["caption"]},
    )


def resource_gate(rss_mib, windows):
    consecutive = 0
    for w in windows:
        if w["seconds"] < 60:
            continue
        consecutive = consecutive + 1 if w["swap_io_bytes"] >= 256 * 2**20 else 0
        if consecutive >= 2:
            return False
    return rss_mib <= 4608


def swap_counters():
    values = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    return (int(values["pswpin"]) + int(values["pswpout"])) * os.sysconf("SC_PAGE_SIZE")


class SwapMonitor:
    """Sample system-wide active swap through loading and inference, without endpoints."""

    def __init__(self, *, clock=None, counter=None):
        import threading
        import time

        self.clock = clock or time.perf_counter
        self.counter = counter or swap_counters
        self.last_clock = self.clock()
        self.last_swap = self.counter()
        self.windows = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        now = self.clock()
        if now - self.last_clock >= 60:
            current = self.counter()
            self.windows.append(
                {"seconds": now - self.last_clock, "swap_io_bytes": current - self.last_swap}
            )
            self.last_clock, self.last_swap = now, current

    def run(self):
        while not self.stop.wait(1):
            self.sample()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def versions():
    names = (
        "torch",
        "torchvision",
        "transformers",
        "numpy",
        "pillow",
        "huggingface_hub",
        "timm",
        "einops",
        "accelerate",
        "tokenizers",
        "safetensors",
        "threadpoolctl",
    )
    out = {}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = None
    return out


def checked_freeze(root=ROOT, *, environment):
    freeze = read(root / "freeze.json")
    for name in ("freeze.json", "input_manifest.json"):
        path = root / name
        if subprocess.check_output(["git", "show", f"HEAD:{path}"]) != path.read_bytes():
            raise ValueError("freeze must be committed unchanged before inference/endpoints")
    for key in ("code_sha256", "protected_sha256", "model_sha256", "source_sha256"):
        verify_hashes(freeze[key])
    if versions() != freeze["runtime_versions"][environment]:
        raise ValueError("frozen runtime environment changed")
    validate_inputs(read(root / "input_manifest.json"))
    return freeze


def cache_provenance(freeze, role, root=ROOT):
    return {
        "freeze_sha256": digest(root / "freeze.json"),
        "role": role,
        "native_dimension": 1024,
        "model_revision": freeze["repositories"]["jinaai/jina-clip-v2"],
        "dtype": "float32",
        "device": "cpu",
        "batch_size": 1,
        "threads": 4,
        "normalization": "official L2",
        "text_task": None,
        "default_lora": "retrieval.query",
    }


def resolve_sources(acquisition, destination):
    """Copy immutable author code and apply two resolution-only call amendments."""
    import difflib

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(acquisition["roots"]["jinaai/jina-clip-implementation"])
    amendments = {
        "modeling_clip.py": (
            "        revision=None,\n",
            "        revision=config.jina_text_config_revision,\n"
            "        code_revision=config.jina_text_code_revision,\n",
        ),
        "hf_model.py": (
            "                self.config = AutoConfig.from_pretrained(\n"
            "                    model_name_or_path,\n",
            "                self.config = AutoConfig.from_pretrained(\n"
            "                    model_name_or_path,\n"
            "                    revision=revision,\n",
        ),
    }
    records = {}
    for file in source.glob("*.py"):
        sha = acquisition["files"][str(file)]["sha256"]
        if digest(file) != sha:
            raise ValueError("original author source hash mismatch")
        original = file.read_text()
        updated = original
        if file.name in amendments:
            before, after = amendments[file.name]
            if original.count(before) != 1:
                raise ValueError("resolution patch source mismatch")
            updated = original.replace(before, after)
        out = destination / file.name
        out.write_text(updated)
        records[file.name] = {
            "original_sha256": sha,
            "resolved_sha256": digest(out),
            "diff": "".join(
                difflib.unified_diff(
                    original.splitlines(True),
                    updated.splitlines(True),
                    fromfile="upstream/" + file.name,
                    tofile="resolved/" + file.name,
                )
            ),
        }
    (destination / "__init__.py").write_text(
        '"""Pinned author wrapper; resolution-only patch."""\n'
    )
    return records
