"""BDD100K still images as a self-supervised corpus with scenario-attribute probes.

BDD100K (Yu et al., CVPR 2020) ships 100k dashcam images, each tagged with
three scenario attributes: weather, scene type, and time of day. Training here
never touches those labels — they exist purely as probe targets, which is
exactly the structure ADAS scenario mining has: unlabeled logs at scale, with
scenario categories (weather x road type x lighting) as the thing an embedding
must preserve to be useful.

Two on-disk layouts are auto-detected, because the official zips and common
redistributions disagree:

    official:   <root>/images/100k/<split>/*.jpg
                <root>/labels/bdd100k_labels_images_<split>.json
    per_image:  <root>/<split>/*.jpg, each with a sibling <name>.json

`data.root` may sit anywhere — it need not be inside the repository. The BDD
`test` split carries no scenario attributes and is unused.

Images are resized to a square (aspect distortion accepted and documented in
the Phase-2 spec: the alternative — a non-square encoder — is an architectural
change with no bearing on the research question). Attribute values outside the
canonical vocabularies below (including "undefined") are dropped from probe
splits only; SSL training uses every image regardless, because scenario mining
does not get to discard unlabeled miles either.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Canonical vocabularies from the BDD100K label spec. Codes are the index into
# these tuples, so they are stable across machines and runs by construction.
ATTRIBUTE_VOCAB: dict[str, tuple[str, ...]] = {
    "weather": ("clear", "foggy", "overcast", "partly cloudy", "rainy", "snowy"),
    "scene": ("city street", "gas stations", "highway", "parking lot", "residential", "tunnel"),
    "timeofday": ("dawn/dusk", "daytime", "night"),
}

# ImageNet statistics — conventional for natural-image SSL and good enough
# here; per-channel BDD statistics would shift every value by a constant the
# encoder immediately absorbs.
BDD_MEAN = (0.485, 0.456, 0.406)
BDD_STD = (0.229, 0.224, 0.225)


def _encode(attributes: dict) -> dict[str, int]:
    """Map raw attribute strings to vocabulary indices; -1 if unusable."""
    return {
        attribute: (
            ATTRIBUTE_VOCAB[attribute].index(attributes[attribute])
            if attributes.get(attribute) in ATTRIBUTE_VOCAB[attribute]
            else -1
        )
        for attribute in ATTRIBUTE_VOCAB
    }


def _extract_attributes(payload: Any) -> dict:
    """Pull the attributes block out of whichever BDD JSON shape this is.

    Releases differ: some put `attributes` at the top level of a per-image
    file, some wrap the image in a single-element list, and some nest the
    scene-level attributes inside `frames[0]`. Rather than guess a release,
    look in all three places and take the first that yields a known key.
    """
    candidates: list[Any] = []
    if isinstance(payload, list):
        candidates.extend(payload[:1])
    else:
        candidates.append(payload)
    for candidate in list(candidates):
        if isinstance(candidate, dict):
            frames = candidate.get("frames")
            if isinstance(frames, list) and frames:
                candidates.append(frames[0])
    for candidate in candidates:
        if isinstance(candidate, dict):
            attributes = candidate.get("attributes")
            if isinstance(attributes, dict) and any(k in attributes for k in ATTRIBUTE_VOCAB):
                return attributes
    return {}


def discover_split_dir(root: Path, split: str) -> tuple[Path, str]:
    """Locate a split's image directory and identify the on-disk layout.

    Two layouts are supported, because the official zips and several common
    redistributions disagree:

    - "official":  <root>/images/100k/<split>/*.jpg
                   <root>/labels/bdd100k_labels_images_<split>.json
    - "per_image": <root>/<split>/*.jpg  with a sibling <name>.json each

    Returns (image_dir, layout). Raises with the paths tried if neither exists,
    since a silent empty index would look like a training bug much later.
    """
    root = Path(root)
    official = root / "images" / "100k" / split
    if official.is_dir():
        return official, "official"
    flat = root / split
    if flat.is_dir():
        return flat, "per_image"
    raise FileNotFoundError(
        f"no BDD100K split {split!r} under {root}. Tried {official} and {flat}. "
        "Point data.root at the directory containing either images/100k/<split>/ "
        "or <split>/ directly."
    )


def load_index(
    root: Path, split: str, use_cache: bool = True
) -> list[tuple[Path, dict[str, int]]]:
    """(image path, {attribute: code}) per image; code -1 = attribute unusable.

    Attribute values outside the canonical vocabulary (e.g. "undefined") get
    -1 so probe construction can filter them while SSL still trains on the
    image — scenario mining does not get to discard unlabeled miles either.

    The per-image layout means one `open()` per image to build the index, which
    is minutes across 70k files and would otherwise be repeated by all 21 runs
    of a seeded matrix. The result is cached beside the data; delete
    `.embedding_diagnostics_index_<split>.json` to force a rebuild.
    """
    root = Path(root)
    image_dir, layout = discover_split_dir(root, split)

    cache_path = root / f".embedding_diagnostics_index_{split}.json"
    if use_cache and cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        return [(image_dir / name, codes) for name, codes in cached]

    index: list[tuple[Path, dict[str, int]]] = []
    if layout == "official":
        labels_path = root / "labels" / f"bdd100k_labels_images_{split}.json"
        for entry in json.loads(labels_path.read_text()):
            name = entry.get("name")
            if name:
                index.append((image_dir / name, _encode(_extract_attributes(entry))))
    else:
        for image_path in sorted(image_dir.glob("*.jpg")):
            sidecar = image_path.with_suffix(".json")
            attributes = {}
            if sidecar.is_file():
                try:
                    attributes = _extract_attributes(json.loads(sidecar.read_text()))
                except json.JSONDecodeError:
                    attributes = {}
            index.append((image_path, _encode(attributes)))

    if use_cache and index:
        cache_path.write_text(
            json.dumps([[path.name, codes] for path, codes in index])
        )
    return index


class Bdd100kImages(Dataset):
    """Images only — attributes are deliberately not returned.

    The SSL loop must not be able to see labels even by accident; probe labels
    travel separately through `build_bdd_dataloaders`.
    """

    def __init__(self, index: list[tuple[Path, dict[str, int]]], transform) -> None:
        self.index = index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, position: int):
        path, _ = self.index[position]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, 0  # dummy label, matching the CIFAR loader's contract


def _probe_split(
    index: list[tuple[Path, dict[str, int]]],
    count: int,
    image_size: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    """Deterministically subsample fully-labelled images and stack them."""
    usable = [(path, codes) for path, codes in index if all(c >= 0 for c in codes.values())]
    if not usable:
        raise ValueError("no fully-labelled images in this split")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(usable), size=min(count, len(usable)), replace=False)

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(BDD_MEAN, BDD_STD),
        ]
    )
    images, codes_list = [], []
    for position in chosen:
        path, codes = usable[int(position)]
        with Image.open(path) as image:
            images.append(eval_transform(image.convert("RGB")))
        codes_list.append(codes)

    labels = {
        attribute: np.array([codes[attribute] for codes in codes_list])
        for attribute in ATTRIBUTE_VOCAB
    }
    return torch.stack(images), labels


def build_bdd_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, tuple, tuple]:
    """Mirror of the CIFAR loader's contract, with dict-valued probe labels.

    Returns (ssl_loader, probe_train, probe_test); each probe split is
    (images_tensor, {attribute: label_array}). The trainer's `evaluate` probes
    and retrieves each attribute separately and logs the across-attribute means
    under the canonical keys.
    """
    data_config = config["data"]
    image_size = config["model"]["image_size"]
    root = Path(data_config["root"])

    train_index = load_index(root, "train")
    val_index = load_index(root, "val")

    ssl_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(BDD_MEAN, BDD_STD),
        ]
    )
    # Explicit generator for the same reason as the CIFAR path: without one the
    # shuffle and worker seeds come from the global RNG, which conditions
    # consume at different rates — a between-condition confound.
    loader_generator = torch.Generator().manual_seed(config["seed"])
    ssl_loader = DataLoader(
        Bdd100kImages(train_index, ssl_transform),
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        drop_last=True,
        persistent_workers=data_config["num_workers"] > 0,
        generator=loader_generator,
    )

    # eval_split_seed, not the run seed: the scored images must be identical
    # across conditions and training seeds.
    eval_seed = data_config.get("eval_split_seed", 0)
    probe_train = _probe_split(
        train_index, data_config["probe_train_samples"], image_size, eval_seed
    )
    probe_test = _probe_split(
        val_index, data_config["probe_test_samples"], image_size, eval_seed
    )
    return ssl_loader, probe_train, probe_test
