"""Real tiny Lance fixtures: metadata selection never scans stored embeddings."""

import io

import lance
import pyarrow as pa
import pytest
from PIL import Image

from embedding_diagnostics.bdd_lance import LanceBdd


@pytest.fixture
def dataset(tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), "red").save(buffer, format="PNG")
    rows = [
        {"image_id": name, "split": split, "image_bytes": buffer.getvalue(),
         "weather": weather, "scene": "highway", "timeofday": "daytime",
         "embedding": [float("nan")]}
        for name, split, weather in [
            ("z", "train", "clear"), ("a", "train", "rainy"),
            ("u", "train", "undefined"), ("v", "val", "foggy"),
        ]
    ]
    path = tmp_path / "bdd.lance"
    lance.write_dataset(pa.Table.from_pylist(rows), path)
    return LanceBdd(path)


def test_split_selection_mapping_and_order(dataset):
    train = dataset.select("train", 2)
    val = dataset.select("val", 1)
    assert [r["image_id"] for r in train] == ["a", "z"]
    assert [r["image_id"] for r in val] == ["v"]
    assert train[0]["weather"] == 4
    assert val[0]["weather"] == 1
    assert all(r["scene"] == 2 and r["timeofday"] == 1 for r in train + val)
    assert dataset.select("train", 2) == train
    assert set(train[0]) == {"image_id", "split", "weather", "scene", "timeofday"}


def test_decode_reads_raw_bytes_in_selected_order(dataset):
    rows = dataset.select("train", 2)
    images, hashes = dataset.decode(rows)
    assert len(images) == len(hashes) == 2
    assert images[0].size == (8, 6)
    assert images[0].getpixel((0, 0)) == (255, 0, 0)
    assert len(hashes[0]) == 64


def test_unknown_attribute_is_retained_as_minus_one_when_requested(dataset):
    rows = dataset.select("train", 3, fully_labelled=False)
    assert rows[1]["image_id"] == "u" and rows[1]["weather"] == -1


def test_selection_refuses_shortfall_and_unknown_split(dataset):
    with pytest.raises(ValueError, match="available"):
        dataset.select("train", 3)
    with pytest.raises(ValueError, match="split"):
        dataset.select("test", 1)


def test_decode_rejects_changed_metadata(dataset):
    rows = dataset.select("train", 1)
    rows[0]["weather"] = 0
    with pytest.raises(ValueError, match="metadata"):
        dataset.decode(rows)


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "duplicate.lance"
    lance.write_dataset(pa.Table.from_pylist([
        {"image_id": "a", "split": split, "weather": "clear", "scene": "highway",
         "timeofday": "daytime", "image_bytes": b"unused"}
        for split in ("train", "val")
    ]), path)
    with pytest.raises(ValueError, match="duplicate"):
        LanceBdd(path)


def test_all_lance_reads_project_only_phase_c_columns(dataset, monkeypatch):
    """A read of a virtual stored embedding is rejected at the I/O boundary."""
    real = dataset.dataset

    class ProjectedDataset:
        schema = real.schema
        version = real.version

        def count_rows(self):
            return real.count_rows()

        def scanner(self, *, columns, **kwargs):
            assert set(columns) == {"image_id", "split", "weather", "scene", "timeofday"}
            return real.scanner(columns=columns, **kwargs)

        def take(self, positions, *, columns):
            assert set(columns) == {
                "image_id", "split", "weather", "scene", "timeofday", "image_bytes",
            }
            return real.take(positions, columns=columns)

    monkeypatch.setattr(lance, "dataset", lambda path: ProjectedDataset())
    reader = LanceBdd(dataset.identity["path"])
    assert len(reader.decode(reader.select("train", 2))[0]) == 2
