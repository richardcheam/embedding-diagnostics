"""Original COCO joins and image-group sampling, with no encoder dependencies.

Schema provenance: COCO Consortium's format-data documentation; human-caption
collection: Chen et al., arXiv:1504.00325v2 (2015). Tests validate joins and
selection wiring, not dataset authenticity or completeness of semantic relevance.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from collections import defaultdict

import numpy as np
from PIL import Image

SELECTION_PREFIX = "embedding-diagnostics:coco2017:20261009:"


def inventory_zip(path, rows: list[dict]) -> dict[int, dict]:
    """Read source members incrementally; check archive CRC, dimensions and hashes.

    RGB pixel identity is shape-bound and does not apply EXIF reorientation.
    Neither these checks nor hash continuity authenticate human-photographic origin.
    """
    result = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        expected = {f"val2017/{r['file_name']}" for r in rows}
        members = {n for n in names if not n.endswith("/")}
        if len(names) != len(set(names)) or members != expected:
            raise ValueError("archive members do not match original image joins")
        for row in rows:
            raw = archive.read(f"val2017/{row['file_name']}")
            with Image.open(io.BytesIO(raw)) as image:
                if image.size != (row["width"], row["height"]):
                    raise ValueError("decoded dimensions differ from annotations")
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            header = f"RGB:{row['width']}:{row['height']}:".encode()
            result[row["image_id"]] = {
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "pixel_sha256": hashlib.sha256(header + rgb.tobytes()).hexdigest(),
                "source_bytes": len(raw),
            }
    return result


def _index(rows, name):
    result = {}
    for row in rows:
        identifier = row["id"]
        if type(identifier) is not int or identifier in result:
            raise ValueError(f"invalid/duplicate {name} ID")
        result[identifier] = row
    return result


def join_source(captions: dict, instances: dict) -> list[dict]:
    """Validate original image/annotation joins without rewriting identifiers."""
    images = _index(captions["images"], "image")
    other = _index(instances["images"], "instance image")
    licenses = _index(captions["licenses"], "license")
    categories = _index(instances["categories"], "category")
    if images != other:
        raise ValueError("caption/instance image joins differ")
    by_image, by_category = defaultdict(list), defaultdict(set)
    for ann in _index(captions["annotations"], "caption").values():
        if ann["image_id"] not in images or not isinstance(ann["caption"], str):
            raise ValueError("invalid caption foreign key/text")
        if not ann["caption"].strip():
            raise ValueError("empty caption")
        by_image[ann["image_id"]].append(ann)
    for ann in _index(instances["annotations"], "instance").values():
        if ann["image_id"] not in images or ann["category_id"] not in categories:
            raise ValueError("invalid instance foreign key/category")
        by_category[ann["image_id"]].add(ann["category_id"])
    rows = []
    for identifier, image in sorted(images.items()):
        if image["license"] not in licenses or min(image["width"], image["height"]) <= 0:
            raise ValueError("invalid license/dimensions")
        if image["file_name"] != f"{identifier:012d}.jpg":
            raise ValueError("original validation filename/ID mismatch")
        rows.append({**image, "image_id": identifier, "split": "val2017",
                     "license_record": licenses[image["license"]],
                     "captions": sorted(by_image[identifier], key=lambda a: a["id"]),
                     "categories": sorted(by_category[identifier])})
    return rows


def photo_id(url: str) -> str | None:
    """Recover Flickr photo identity from photo pages or original static filenames."""
    match = re.search(r"/photos/[^/]+/(\d+)(?:/|$|\?)", url)
    if match is None:
        match = re.search(r"/(\d+)_[^/]+\.(?:jpg|png)(?:\?|$)", url)
    return match.group(1) if match else None


def group_sources(rows: list[dict], inventory: dict[int, dict]) -> list[dict]:
    """Transitive exact-content/photo groups, retaining minimum-ID representatives."""
    ids = [r["image_id"] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(inventory):
        raise ValueError("inventory row alignment mismatch")
    parent = {i: i for i in ids}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for row in rows:
        i = row["image_id"]
        keys = [("raw", inventory[i]["raw_sha256"]),
                ("pixel", inventory[i]["pixel_sha256"])]
        recovered = photo_id(row.get("flickr_url", ""))
        if recovered:
            keys.append(("photo", recovered))
        for key in keys:
            if key in seen:
                a, b = root(i), root(seen[key])
                parent[max(a, b)] = min(a, b)
            else:
                seen[key] = i
    groups = defaultdict(list)
    index = {r["image_id"]: r for r in rows}
    for i in ids:
        groups[root(i)].append(i)
    return [{**index[min(aliases)], **inventory[min(aliases)],
             "aliases": sorted(aliases)} for _, aliases in sorted(groups.items())]


def select_groups(groups: list[dict], *, count: int = 1000) -> list[dict]:
    """Fixed hash ranking, no replacement/quotas; five lowest annotation IDs."""
    eligible = [r for r in groups if len(r["captions"]) >= 5]
    if count <= 0 or len(eligible) < count:
        raise ValueError("insufficient eligible image groups")
    if len({r["image_id"] for r in groups}) != len(groups):
        raise ValueError("duplicate group representatives")

    def key(row):
        i = row["image_id"]
        return hashlib.sha256(f"{SELECTION_PREFIX}{i}".encode()).hexdigest(), i

    result = []
    for row in sorted(sorted(eligible, key=key)[:count], key=lambda r: r["image_id"]):
        captions = sorted(row["captions"], key=lambda a: a["id"])
        result.append({**row, "captions": captions[:5],
                       "unused_caption_ids": [a["id"] for a in captions[5:]]})
    return result
