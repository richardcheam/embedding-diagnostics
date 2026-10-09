"""Jina native 1024d canonical caches, following existing atomic manifest conventions.

No source image bytes or caption text are duplicated in vector chunks. Ordered
identity and input hashes remain in the manifest. Dedicated dimension boundary;
historical cache formats remain unchanged.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .embedding_cache import _checksum, atomic_json
from .jina_compression import validate_jina


def _rows(rows):
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("unique ordered source IDs required")
    if len({r["role"] for r in rows}) != 1:
        raise ValueError("one caption/image role per cache required")
    for r in rows:
        h = r["input_sha256"]
        if r["role"] not in ("image", "caption") or type(r["id"]) is not int:
            raise ValueError("invalid role/source ID")
        if (
            type(r["parent_image_id"]) is not int
            or len(h) != 64
            or any(c not in "0123456789abcdef" for c in h)
        ):
            raise ValueError("invalid parent/input hash")


def _read(root, manifest):
    if manifest.get("format_version") != "jina-paired-1":
        raise ValueError("unsupported paired cache version")
    _rows(manifest["rows"])
    arrays, completed = [], 0
    for chunk in manifest["chunks"]:
        start, stop = chunk["start"], chunk["stop"]
        filename = f"chunk-{start:08d}-{stop:08d}.npz"
        if start != completed or not start < stop <= len(manifest["rows"]):
            raise ValueError("chunk row alignment mismatch")
        if chunk["file"] != filename or _checksum(root / filename) != chunk["sha256"]:
            raise ValueError("chunk checksum/filename mismatch")
        with np.load(root / filename, allow_pickle=False) as stored:
            matrix = validate_jina(stored["embeddings"], stop - start)
            if not np.allclose(np.linalg.norm(matrix, axis=1), 1, rtol=1e-5, atol=1e-6):
                raise ValueError("unit normalized canonical vectors required")
            for key in ("id", "parent_image_id", "role", "input_sha256"):
                if stored[key].tolist() != [r[key] for r in manifest["rows"][start:stop]]:
                    raise ValueError("chunk source metadata alignment mismatch")
            arrays.append(matrix)
        completed = stop
    return arrays, completed


def load_jina_cache(root):
    """Verify every chunk and return a complete plain FP32 matrix."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    arrays, completed = _read(root, manifest)
    if completed != len(manifest["rows"]):
        raise ValueError("incomplete paired cache")
    return np.concatenate(arrays), manifest


class JinaPairedCache:
    """Exclusive ordered writer, commits each validated prefix atomically."""

    def __init__(self, root, provenance, rows):
        _rows(rows)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.path = self.root / "manifest.json"
            provenance = json.loads(json.dumps(provenance, allow_nan=False))
            if self.path.exists():
                self.manifest = json.loads(self.path.read_text())
                if self.manifest["provenance"] != provenance or self.manifest["rows"] != rows:
                    raise ValueError("resume provenance/selection mismatch")
                _, self.completed = _read(self.root, self.manifest)
            else:
                self.manifest = {
                    "format_version": "jina-paired-1",
                    "provenance": provenance,
                    "rows": rows,
                    "chunks": [],
                }
                self.completed = 0
                atomic_json(self.path, self.manifest)
        except BaseException:
            self.lock.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.lock.close()

    def append(self, embeddings):
        matrix = validate_jina(embeddings, len(embeddings))
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1, rtol=1e-5, atol=1e-6):
            raise ValueError("unit normalized canonical vectors required")
        start, stop = self.completed, self.completed + len(matrix)
        rows = self.manifest["rows"][start:stop]
        if len(rows) != len(matrix):
            raise ValueError("chunk exceeds frozen selection")
        name = f"chunk-{start:08d}-{stop:08d}.npz"
        with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                np.savez_compressed(
                    handle,
                    embeddings=matrix,
                    **{k: np.array([r[k] for r in rows]) for k in rows[0]},
                )
                handle.flush()
                os.fsync(handle.fileno())
                os.replace(temporary, self.root / name)
            finally:
                temporary.unlink(missing_ok=True)
        chunk = {"file": name, "start": start, "stop": stop, "sha256": _checksum(self.root / name)}
        updated = self.manifest | {"chunks": [*self.manifest["chunks"], chunk]}
        atomic_json(self.path, updated)
        self.manifest, self.completed = updated, stop
