"""Local Lance backend for BDD100K (Yu et al., CVPR 2020).

Phase C uses raw image bytes only. Explicit column projection excludes the
redistribution's virtual embedding/UDF columns. Phase B's vocabulary and -1
policy are reused; the existing filesystem loader is unchanged.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import lance
from PIL import Image

from .bdd100k import ATTRIBUTE_VOCAB, _encode

METADATA_COLUMNS = ["image_id", "split", *ATTRIBUTE_VOCAB]


class LanceBdd:
    """Scan small metadata once; decode only the requested image batch."""

    def __init__(self, path: str | Path):
        path = Path(path).resolve()
        if not (path / "_versions").is_dir():
            path = path / "data" / "bdd100k.lance"
        self.dataset = lance.dataset(str(path))
        required = {*METADATA_COLUMNS, "image_bytes"}
        if not required <= set(self.dataset.schema.names):
            raise ValueError(f"missing Lance columns: {required - set(self.dataset.schema.names)}")
        self.identity = {"path": str(path), "source": "lance-format/BDD100K-enriched",
                         "lance_version": self.dataset.version,
                         "rows": self.dataset.count_rows()}
        self.rows = []
        self.positions = {}
        scanner = self.dataset.scanner(columns=METADATA_COLUMNS, batch_size=1024,
                                       batch_readahead=1, fragment_readahead=1,
                                       scan_in_order=True)
        for batch in scanner.to_batches():
            for row in batch.to_pylist():
                image_id = row["image_id"]
                if not isinstance(image_id, str) or not image_id:
                    raise ValueError("image_id must be a nonempty string")
                if image_id in self.positions:
                    raise ValueError(f"duplicate image_id {image_id}")
                self.positions[image_id] = len(self.rows)
                self.rows.append({"image_id": image_id, "split": row["split"], **_encode(row)})

    def select(self, split: str, count: int, *, fully_labelled: bool = True) -> list[dict]:
        """C0 procedure: first N lexicographic IDs in each physical split.

        This is an integration sample, not the eventual representative C1
        sample. Requesting more than available fails rather than shrinking N.
        """
        if split not in ("train", "val") or count < 1:
            raise ValueError("split must be train or val, count must be positive")
        rows = sorted(
            (r for r in self.rows if r["split"] == split and
             (not fully_labelled or all(r[a] >= 0 for a in ATTRIBUTE_VOCAB))),
            key=lambda r: r["image_id"],
        )
        if count > len(rows):
            raise ValueError(f"requested {count} {split} rows; only {len(rows)} available")
        return [dict(r) for r in rows[:count]]

    def decode(self, rows: list[dict]) -> tuple[list[Image.Image], list[str]]:
        """Read one batch's bytes by physical position, validating its metadata.

        A dataset version is fixed at construction. Raw-byte SHA256 values
        connect each completed cache chunk to the precise input images.
        """
        positions = []
        for row in rows:
            position = self.positions[row["image_id"]]
            if row != self.rows[position]:
                raise ValueError(f"selected metadata changed for {row['image_id']}")
            positions.append(position)
        batch = self.dataset.take(positions, columns=[*METADATA_COLUMNS, "image_bytes"])
        images, hashes = [], []
        for expected, actual in zip(rows, batch.to_pylist(), strict=True):
            if actual["image_id"] != expected["image_id"]:
                raise ValueError("Lance take returned a different image order")
            payload = actual["image_bytes"]
            hashes.append(hashlib.sha256(payload).hexdigest())
            with Image.open(io.BytesIO(payload)) as image:
                images.append(image.convert("RGB"))
        return images, hashes
