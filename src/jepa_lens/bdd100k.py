"""BDD100K still images as a self-supervised corpus with scenario-attribute probes.

BDD100K (Yu et al., CVPR 2020) ships 100k dashcam images, each tagged with
three scenario attributes: weather, scene type, and time of day. Training here
never touches those labels — they exist purely as probe targets, which is
exactly the structure ADAS scenario mining has: unlabeled logs at scale, with
scenario categories (weather x road type x lighting) as the thing an embedding
must preserve to be useful.

Expected layout under `data.root` (the official zips produce this):

    <root>/images/100k/train/*.jpg
    <root>/images/100k/val/*.jpg
    <root>/labels/bdd100k_labels_images_train.json
    <root>/labels/bdd100k_labels_images_val.json

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


def load_index(root: Path, split: str) -> list[tuple[Path, dict[str, int]]]:
    """(image path, {attribute: code}) per labelled image; code -1 = unusable.

    Reads the official per-split labels JSON. Entries whose image name is
    missing are skipped; attribute values outside the canonical vocabulary
    (e.g. "undefined") get code -1 so probe construction can filter them while
    SSL keeps the image.
    """
    root = Path(root)
    labels_path = root / "labels" / f"bdd100k_labels_images_{split}.json"
    entries = json.loads(labels_path.read_text())
    image_dir = root / "images" / "100k" / split

    index: list[tuple[Path, dict[str, int]]] = []
    for entry in entries:
        name = entry.get("name")
        if not name:
            continue
        attributes = entry.get("attributes", {})
        codes = {
            attribute: (
                ATTRIBUTE_VOCAB[attribute].index(attributes[attribute])
                if attributes.get(attribute) in ATTRIBUTE_VOCAB[attribute]
                else -1
            )
            for attribute in ATTRIBUTE_VOCAB
        }
        index.append((image_dir / name, codes))
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

    probe_train = _probe_split(
        train_index, data_config["probe_train_samples"], image_size, config["seed"]
    )
    probe_test = _probe_split(
        val_index, data_config["probe_test_samples"], image_size, config["seed"]
    )
    return ssl_loader, probe_train, probe_test
