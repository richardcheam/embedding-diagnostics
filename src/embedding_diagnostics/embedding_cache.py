"""Canonical external vectors, atomic chunks, and immutable extraction provenance.

No image bytes are stored. A manifest commits each complete chunk by checksum;
an interrupted uncommitted chunk can be recomputed without losing earlier work.
All writers are serialized by a local advisory lock (Linux extraction host).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .bdd100k import ATTRIBUTE_VOCAB
from .models.embeddinggemma2 import validate_embeddings


def atomic_json(path: Path, payload: dict) -> None:
    """Commit JSON by fsynced temporary file and same-directory rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _checksum(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _validate_rows(rows: list[dict]) -> None:
    if not rows or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("selection must contain unique image IDs")
    if [r["image_id"] for r in rows] != sorted(r["image_id"] for r in rows):
        raise ValueError("selection must be ordered by image_id")
    for row in rows:
        if set(row) != {"image_id", "split", *ATTRIBUTE_VOCAB}:
            raise ValueError("invalid selection metadata columns")
        if row["split"] not in ("train", "val"):
            raise ValueError("selection split must be train or val")
        for attribute, vocab in ATTRIBUTE_VOCAB.items():
            if not isinstance(row[attribute], int) or not -1 <= row[attribute] < len(vocab):
                raise ValueError(f"invalid {attribute} label")


def _read_chunks(root: Path, manifest: dict) -> tuple[list[np.ndarray], int]:
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported cache format")
    rows = manifest["selected_rows"]
    _validate_rows(rows)
    arrays, completed = [], 0
    for chunk in manifest["chunks"]:
        start, stop = chunk["start"], chunk["stop"]
        if start != completed or not start < stop <= len(rows):
            raise ValueError("invalid chunk order")
        filename = f"chunk-{start:08d}-{stop:08d}.npz"
        if chunk["file"] != filename:
            raise ValueError("invalid chunk filename")
        path = root / filename
        if not path.is_file() or _checksum(path) != chunk["sha256"]:
            raise ValueError(f"chunk checksum mismatch: {filename}")
        if len(chunk["image_sha256"]) != stop - start:
            raise ValueError("image hash count mismatch")
        with np.load(path, allow_pickle=False) as stored:
            matrix = validate_embeddings(stored["embeddings"], stop - start)
            if not np.allclose(np.linalg.norm(matrix, axis=1), 1, rtol=1e-5, atol=1e-6):
                raise ValueError("canonical embeddings must have unit norms")
            for column in ("image_id", "split", *ATTRIBUTE_VOCAB):
                if stored[column].tolist() != [r[column] for r in rows[start:stop]]:
                    raise ValueError(f"cached {column} does not match selected metadata")
            arrays.append(matrix)
        completed = stop
    return arrays, completed


def load_cache(root: str | Path) -> tuple[np.ndarray, dict]:
    """Reload a complete cache and validate vectors, checksums, and row alignment."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    arrays, completed = _read_chunks(root, manifest)
    if completed != len(manifest["selected_rows"]):
        raise ValueError(f"incomplete cache: {completed}/{len(manifest['selected_rows'])}")
    return np.concatenate(arrays), manifest


class EmbeddingCache:
    """Exclusive, resumable writer; use as a context manager."""

    def __init__(self, root: str | Path, provenance: dict, rows: list[dict]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / ".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.lock.close()
            raise ValueError("cache is locked by another writer") from error
        try:
            self.path = self.root / "manifest.json"
            # JSON canonicalization ensures reload equality (tuples become lists).
            provenance = json.loads(json.dumps(provenance, allow_nan=False))
            if self.path.exists():
                self.manifest = json.loads(self.path.read_text())
                if self.manifest["provenance"] != provenance:
                    raise ValueError("resume provenance mismatch")
                if self.manifest["selected_rows"] != rows:
                    raise ValueError("resume selection mismatch")
                _, self.completed = _read_chunks(self.root, self.manifest)
            else:
                _validate_rows(rows)
                self.manifest = {
                    "format_version": 1, "created_at": datetime.now(UTC).isoformat(),
                    "provenance": provenance, "selected_rows": rows, "chunks": [],
                }
                self.completed = 0
                self._commit(self.manifest)
        except BaseException:
            self.lock.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.lock.close()

    def _commit(self, manifest: dict) -> None:
        atomic_json(self.path, manifest)

    def append(self, embeddings: np.ndarray, image_sha256: list[str]) -> None:
        """Atomically append the next ordered rows; commit only validated batches."""
        matrix = validate_embeddings(embeddings, len(embeddings))
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1, rtol=1e-5, atol=1e-6):
            raise ValueError("canonical embeddings must have unit norms")
        if len(image_sha256) != len(matrix) or any(
            len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in image_sha256
        ):
            raise ValueError("one SHA256 required per input image")
        start, stop = self.completed, self.completed + len(matrix)
        rows = self.manifest["selected_rows"][start:stop]
        if len(rows) != len(matrix):
            raise ValueError("chunk exceeds selected image count")
        filename = f"chunk-{start:08d}-{stop:08d}.npz"
        path = self.root / filename
        with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                np.savez_compressed(handle, embeddings=matrix, **{
                    column: np.array([r[column] for r in rows])
                    for column in ("image_id", "split", *ATTRIBUTE_VOCAB)
                })
                handle.flush()
                os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        chunk = {"file": filename, "start": start, "stop": stop,
                 "sha256": _checksum(path), "image_sha256": image_sha256}
        updated = self.manifest | {"chunks": [*self.manifest["chunks"], chunk]}
        self._commit(updated)
        self.manifest, self.completed = updated, stop
