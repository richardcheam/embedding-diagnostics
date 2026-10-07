"""Offline vertical slice with a synthetic encoder; no large model in CI."""

import io

import lance
import numpy as np
import pyarrow as pa
import pytest
from PIL import Image

from embedding_diagnostics.phase_c import extract, geometry_report


class SyntheticEncoder:
    def __init__(self):
        self.calls = 0
        self.provenance = {"model_id": "synthetic-test-only", "model_revision": "a" * 40,
                           "dtype": "float32", "device": "cpu"}

    def encode(self, images):
        self.calls += 1
        matrix = np.zeros((len(images), 768), dtype=np.float32)
        for i, image in enumerate(images):
            matrix[i, image.getpixel((0, 0))[0]] = 1
        return matrix


def test_complete_vertical_slice_resume_and_repeat(tmp_path):
    from embedding_diagnostics.bdd_lance import LanceBdd

    rows = []
    for i in range(4):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (i, 0, 0)).save(buffer, format="PNG")
        rows.append({"image_id": str(i), "split": "train" if i < 2 else "val",
                     "weather": "clear", "scene": "highway", "timeofday": "daytime",
                     "image_bytes": buffer.getvalue(), "embedding": [float("nan")]})
    path = tmp_path / "fixture.lance"
    lance.write_dataset(pa.Table.from_pylist(rows), path)
    dataset = LanceBdd(path)
    selected = dataset.select("train", 2) + dataset.select("val", 2)
    encoder = SyntheticEncoder()
    root = tmp_path / "cache"
    report = extract(dataset, selected, encoder, root, batch_size=1, chunk_size=2,
                     code_identity={"commit": "fixture"}, verify_repeat=True)
    assert report["shape"] == [4, 768]
    assert report["all_finite"] and report["repeat_equivalent"]
    assert report["repeat_count"] == 4
    assert report["norm_range"] == [1., 1.]
    assert report["new_images"] == 4
    assert report["geometry"]["total_variance"] == pytest.approx(1.)
    calls = encoder.calls
    resumed = extract(dataset, selected, encoder, root, batch_size=1, chunk_size=2,
                      code_identity={"commit": "fixture"}, verify_repeat=False)
    assert resumed["new_images"] == 0 and encoder.calls == calls
    assert geometry_report(root)["geometry"] == report["geometry"]
    with pytest.raises(ValueError, match="provenance"):
        extract(dataset, selected, encoder, root, batch_size=2, chunk_size=2,
                code_identity={"commit": "fixture"})


def test_repeat_failure_is_reported_loudly(tmp_path):
    class Dataset:
        identity = {"source": "fixture"}

        def decode(self, rows):
            return [Image.new("RGB", (8, 8))], ["a" * 64]

    class Unstable(SyntheticEncoder):
        def encode(self, images):
            matrix = super().encode(images)
            return np.roll(matrix, self.calls, axis=1)

    selected = [{"image_id": "a", "split": "train", "weather": 0,
                 "scene": 0, "timeofday": 1}]
    with pytest.raises(ValueError, match="repeat"):
        extract(Dataset(), selected, Unstable(), tmp_path, verify_repeat=True,
                code_identity={"commit": "fixture"})


def test_resume_rejects_changed_image_bytes_without_repeat_inference(tmp_path):
    class Dataset:
        identity = {"source": "same-path-and-version"}
        digest = "a" * 64

        def decode(self, rows):
            return [Image.new("RGB", (8, 8)) for _ in rows], [self.digest] * len(rows)

    dataset = Dataset()
    selected = [{"image_id": "a", "split": "train", "weather": 0,
                 "scene": 0, "timeofday": 1}]
    encoder = SyntheticEncoder()
    extract(dataset, selected, encoder, tmp_path, code_identity={"commit": "fixture"})
    dataset.digest = "b" * 64
    with pytest.raises(ValueError, match="source image"):
        extract(dataset, selected, encoder, tmp_path, code_identity={"commit": "fixture"})


def test_recreated_lance_source_cannot_mix_with_a_partial_cache(tmp_path):
    import json
    import shutil

    from embedding_diagnostics.bdd_lance import LanceBdd

    path = tmp_path / "replaceable.lance"

    def write_source(pixel):
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (pixel, 0, 0)).save(buffer, format="PNG")
        records = [{"image_id": str(i), "split": "train", "weather": "clear",
                    "scene": "highway", "timeofday": "daytime",
                    "image_bytes": buffer.getvalue()} for i in range(2)]
        lance.write_dataset(pa.Table.from_pylist(records), path)

    class InterruptedEncoder(SyntheticEncoder):
        def encode(self, images):
            if self.calls == 1:
                raise RuntimeError("fixture interruption")
            return super().encode(images)

    write_source(1)
    original = LanceBdd(path)
    selected = original.select("train", 2)
    root = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="extraction failed"):
        extract(original, selected, InterruptedEncoder(), root, chunk_size=1,
                code_identity={"commit": "fixture"})
    shutil.rmtree(path)
    write_source(2)
    replaced = LanceBdd(path)
    assert replaced.identity == original.identity
    assert replaced.select("train", 2) == selected
    encoder = SyntheticEncoder()
    with pytest.raises(ValueError, match="source image"):
        extract(replaced, selected, encoder, root, chunk_size=1,
                code_identity={"commit": "fixture"})
    assert encoder.calls == 0
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["chunks"][0]["stop"] == 1 and len(manifest["chunks"]) == 1
