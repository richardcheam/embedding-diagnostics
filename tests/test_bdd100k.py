"""BDD100K loader, retrieval, and multi-attribute evaluation — fixture-based.

No download happens here: a synthetic BDD-shaped tree (tiny JPEGs plus the
official JSON layout) exercises the full path, so CI stays offline and fast.
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from jepa_lens.bdd100k import ATTRIBUTE_VOCAB, build_bdd_dataloaders, load_index
from jepa_lens.diagnostics.retrieval import retrieval_precision_at_k


def make_bdd_tree(root, split, entries):
    """Write tiny JPEGs and a labels JSON in the official layout."""
    image_dir = root / "images" / "100k" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    payload = []
    rng = np.random.default_rng(0)
    for name, attributes in entries:
        pixels = rng.integers(0, 255, size=(24, 32, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(image_dir / name, format="JPEG")
        payload.append({"name": name, "attributes": attributes, "labels": []})
    (labels_dir / f"bdd100k_labels_images_{split}.json").write_text(json.dumps(payload))


CLEAN = {"weather": "clear", "scene": "highway", "timeofday": "daytime"}
RAINY = {"weather": "rainy", "scene": "city street", "timeofday": "night"}


@pytest.fixture
def bdd_root(tmp_path):
    train = [(f"t{i}.jpg", CLEAN if i % 2 == 0 else RAINY) for i in range(8)]
    train.append(("undef.jpg", {"weather": "undefined", "scene": "highway",
                                "timeofday": "daytime"}))
    make_bdd_tree(tmp_path, "train", train)
    make_bdd_tree(tmp_path, "val", [(f"v{i}.jpg", CLEAN if i % 2 else RAINY) for i in range(6)])
    return tmp_path


def test_load_index_codes_attributes_against_the_canonical_vocab(bdd_root):
    index = load_index(bdd_root, "train")
    assert len(index) == 9
    _, codes = index[0]
    assert codes["weather"] == ATTRIBUTE_VOCAB["weather"].index("clear")
    assert codes["scene"] == ATTRIBUTE_VOCAB["scene"].index("highway")


def test_out_of_vocab_values_code_to_minus_one_but_stay_in_the_ssl_index(bdd_root):
    """SSL keeps every image; only probe splits filter. Scenario mining does
    not get to discard unlabeled miles either."""
    index = load_index(bdd_root, "train")
    undef = [codes for path, codes in index if path.name == "undef.jpg"]
    assert undef and undef[0]["weather"] == -1


def bdd_config(root, seed=0):
    return {
        "seed": seed,
        "data": {
            "dataset": "bdd100k",
            "root": str(root),
            "batch_size": 4,
            "num_workers": 0,
            "probe_train_samples": 20,
            "probe_test_samples": 6,
        },
        "model": {"image_size": 32},
    }


def test_build_bdd_dataloaders_returns_the_standard_contract(bdd_root):
    ssl_loader, probe_train, probe_test = build_bdd_dataloaders(bdd_config(bdd_root))
    images, _ = next(iter(ssl_loader))
    assert images.shape == (4, 3, 32, 32)

    train_images, train_labels = probe_train
    assert train_images.shape[1:] == (3, 32, 32)
    assert set(train_labels) == set(ATTRIBUTE_VOCAB)
    # The undefined-weather image must not reach the probe split.
    assert all((train_labels[a] >= 0).all() for a in train_labels)
    # 9 train images minus the undefined one; the cap (20) is not binding.
    assert len(train_images) == 8


def test_dataset_dispatch_reaches_bdd(bdd_root):
    from jepa_lens.data import build_dataloaders

    ssl_loader, _, _ = build_dataloaders(bdd_config(bdd_root))
    images, _ = next(iter(ssl_loader))
    assert images.shape == (4, 3, 32, 32)


def test_unknown_dataset_is_rejected():
    from jepa_lens.data import build_dataloaders

    with pytest.raises(ValueError, match="unknown dataset"):
        build_dataloaders({"data": {"dataset": "imagenet"}})


def test_retrieval_is_perfect_on_separated_clusters():
    rng = np.random.default_rng(0)
    features = np.vstack([rng.normal(-9, 0.1, (30, 8)), rng.normal(9, 0.1, (30, 8))])
    labels = np.array([0] * 30 + [1] * 30)
    assert retrieval_precision_at_k(features, labels, k=10) == 1.0


def test_retrieval_is_near_chance_on_shuffled_labels():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(300, 8))
    labels = rng.integers(0, 3, 300)
    precision = retrieval_precision_at_k(features, labels, k=10)
    assert 0.2 < precision < 0.5  # chance = sum of squared class frequencies ~ 1/3


def test_retrieval_excludes_self_matches():
    """With k=1 and pair-clusters, the neighbour is the OTHER cluster member —
    a self-match would trivially score 1.0 even on unclustered data."""
    features = np.array([[1.0, 0.0], [1.0, 0.001], [0.0, 1.0], [0.0, 1.001]])
    labels = np.array([0, 0, 1, 1])
    assert retrieval_precision_at_k(features, labels, k=1) == 1.0
    mixed = np.array([0, 1, 0, 1])  # now every nearest neighbour disagrees
    assert retrieval_precision_at_k(features, mixed, k=1) == 0.0


def test_evaluate_handles_dict_labels_with_per_attribute_keys():
    """The trainer must probe each scenario attribute separately AND publish
    the across-attribute means under the canonical keys every tool reads."""
    from jepa_lens.training.trainer import Trainer
    from test_phase2_conditions import tiny_config

    trainer = Trainer(tiny_config("none_stopgrad"))
    rng = np.random.default_rng(0)
    images_train = torch.randn(12, 3, 32, 32)
    images_test = torch.randn(10, 3, 32, 32)
    labels_train = {"weather": rng.integers(0, 3, 12), "scene": rng.integers(0, 2, 12)}
    labels_test = {"weather": rng.integers(0, 3, 10), "scene": rng.integers(0, 2, 10)}

    record = trainer.evaluate((images_train, labels_train), (images_test, labels_test))
    for attribute in ("weather", "scene"):
        assert f"probe_accuracy_{attribute}" in record
        assert f"probe_accuracy_unscaled_{attribute}" in record
        assert f"retrieval_p10_{attribute}" in record
    expected = np.mean([record["probe_accuracy_weather"], record["probe_accuracy_scene"]])
    assert abs(record["probe_accuracy"] - expected) < 1e-12
    assert "retrieval_p10" in record


def test_evaluate_still_handles_plain_array_labels():
    from jepa_lens.training.trainer import Trainer
    from test_phase2_conditions import tiny_config

    trainer = Trainer(tiny_config("none_stopgrad"))
    rng = np.random.default_rng(0)
    record = trainer.evaluate(
        (torch.randn(12, 3, 32, 32), rng.integers(0, 3, 12)),
        (torch.randn(10, 3, 32, 32), rng.integers(0, 3, 10)),
    )
    assert "probe_accuracy" in record
    assert "retrieval_p10" in record
    assert "probe_accuracy_weather" not in record


def make_per_image_tree(root, split, entries):
    """The layout on the GPU box: <root>/<split>/name.jpg + name.json."""
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, payload in entries:
        pixels = rng.integers(0, 255, size=(24, 32, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(split_dir / f"{name}.jpg", format="JPEG")
        if payload is not None:
            (split_dir / f"{name}.json").write_text(json.dumps(payload))


def test_per_image_layout_is_detected_and_parsed(tmp_path):
    from jepa_lens.bdd100k import discover_split_dir

    make_per_image_tree(
        tmp_path,
        "train",
        [("a", {"name": "a.jpg", "attributes": CLEAN, "labels": []}),
         ("b", {"name": "b.jpg", "attributes": RAINY, "labels": []})],
    )
    image_dir, layout = discover_split_dir(tmp_path, "train")
    assert layout == "per_image"
    assert image_dir == tmp_path / "train"

    index = load_index(tmp_path, "train", use_cache=False)
    assert len(index) == 2
    codes = dict(index)[tmp_path / "train" / "a.jpg"]
    assert codes["weather"] == ATTRIBUTE_VOCAB["weather"].index("clear")
    assert codes["timeofday"] == ATTRIBUTE_VOCAB["timeofday"].index("daytime")


def test_official_layout_still_wins_when_both_could_match(tmp_path):
    from jepa_lens.bdd100k import discover_split_dir

    make_bdd_tree(tmp_path, "train", [("a.jpg", CLEAN)])
    (tmp_path / "train").mkdir(exist_ok=True)
    _, layout = discover_split_dir(tmp_path, "train")
    assert layout == "official"


def test_attributes_nested_under_frames_are_found(tmp_path):
    """Some releases put scene attributes inside frames[0]."""
    make_per_image_tree(
        tmp_path, "train", [("a", {"name": "a.jpg", "frames": [{"attributes": RAINY}]})]
    )
    codes = load_index(tmp_path, "train", use_cache=False)[0][1]
    assert codes["weather"] == ATTRIBUTE_VOCAB["weather"].index("rainy")


def test_list_wrapped_payload_is_found(tmp_path):
    make_per_image_tree(tmp_path, "train", [("a", [{"name": "a.jpg", "attributes": CLEAN}])])
    codes = load_index(tmp_path, "train", use_cache=False)[0][1]
    assert codes["scene"] == ATTRIBUTE_VOCAB["scene"].index("highway")


def test_image_without_a_sidecar_stays_in_the_index_uncoded(tmp_path):
    """SSL must still see it; only probe splits filter on code >= 0."""
    make_per_image_tree(tmp_path, "train", [("a", None)])
    codes = load_index(tmp_path, "train", use_cache=False)[0][1]
    assert all(value == -1 for value in codes.values())


def test_corrupt_sidecar_does_not_kill_the_index(tmp_path):
    split_dir = tmp_path / "train"
    split_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(split_dir / "a.jpg")
    (split_dir / "a.json").write_text("{not json")
    index = load_index(tmp_path, "train", use_cache=False)
    assert len(index) == 1 and index[0][1]["weather"] == -1


def test_index_is_cached_and_reused(tmp_path):
    make_per_image_tree(tmp_path, "train", [("a", {"attributes": CLEAN})])
    first = load_index(tmp_path, "train")
    cache = tmp_path / ".jepa_lens_index_train.json"
    assert cache.is_file()

    # Corrupt the sidecar; a cached read must not notice.
    (tmp_path / "train" / "a.json").write_text(json.dumps({"attributes": RAINY}))
    assert load_index(tmp_path, "train") == first
    assert load_index(tmp_path, "train", use_cache=False) != first


def test_missing_split_names_the_paths_it_tried(tmp_path):
    from jepa_lens.bdd100k import discover_split_dir

    with pytest.raises(FileNotFoundError, match="images/100k/train"):
        discover_split_dir(tmp_path, "train")


def test_dataloaders_work_end_to_end_on_the_per_image_layout(tmp_path):
    for split in ("train", "val"):
        make_per_image_tree(
            tmp_path,
            split,
            [(f"{split}{i}", {"attributes": CLEAN if i % 2 else RAINY}) for i in range(6)],
        )
    ssl_loader, probe_train, probe_test = build_bdd_dataloaders(bdd_config(tmp_path))
    images, _ = next(iter(ssl_loader))
    assert images.shape == (4, 3, 32, 32)
    assert set(probe_train[1]) == set(ATTRIBUTE_VOCAB)
    assert len(probe_test[0]) == 6
