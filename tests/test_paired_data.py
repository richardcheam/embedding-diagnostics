"""Synthetic original-release joins and independent image-group selection."""

import copy
import io
import zipfile

import pytest
from PIL import Image

from embedding_diagnostics import paired_data as data


def fixture():
    images = [{"id": i, "file_name": f"{i:012d}.jpg", "width": 8, "height": 6,
               "license": 1, "flickr_url": f"https://flickr.com/photos/user/{i}"}
              for i in range(1, 5)]
    captions = {"images": images, "licenses": [{"id": 1, "name": "fixture", "url": "url"}],
                "annotations": [{"id": 10*i+j, "image_id": i, "caption": f"Caption {j}."}
                                for i in range(1, 5) for j in range(6)]}
    instances = {"images": copy.deepcopy(images), "categories": [{"id": 1, "name": "cat"}],
                 "annotations": [{"id": i, "image_id": i, "category_id": 1}
                                 for i in range(1, 5)]}
    return captions, instances


def test_original_joins_and_five_smallest_caption_ids():
    c, i = fixture()
    rows = data.join_source(c, i)
    assert len(rows) == 4 and rows[0]["categories"] == [1]
    groups = data.group_sources(rows, {r["image_id"]: {"raw_sha256": str(r["image_id"]),
                          "pixel_sha256": str(r["image_id"])} for r in rows})
    selected = data.select_groups(groups, count=3)
    assert len(selected) == 3
    assert [r["image_id"] for r in selected] == sorted(r["image_id"] for r in selected)
    assert all(len(r["captions"]) == 5 for r in selected)
    assert all(r["unused_caption_ids"] == [10*r["image_id"]+5] for r in selected)


@pytest.mark.parametrize("defect", ["image_collision", "caption_collision", "orphan",
                                    "license", "empty_caption", "split_join", "category"])
def test_bad_source_joins_rejected(defect):
    c, i = fixture()
    if defect == "image_collision":
        c["images"].append(c["images"][0])
    elif defect == "caption_collision":
        c["annotations"].append(c["annotations"][0])
    elif defect == "orphan":
        c["annotations"][0]["image_id"] = 99
    elif defect == "license":
        c["images"][0]["license"] = 99
    elif defect == "empty_caption":
        c["annotations"][0]["caption"] = " "
    elif defect == "split_join":
        i["images"][0]["file_name"] = "other.jpg"
    else:
        i["annotations"][0]["category_id"] = 99
    with pytest.raises(ValueError):
        data.join_source(c, i)


def test_grouping_transitive_and_no_replacement():
    c, i = fixture()
    rows = data.join_source(c, i)
    inventory = {j: {"raw_sha256": f"raw{j}", "pixel_sha256": f"pixel{j}"}
                 for j in range(1, 5)}
    inventory[2]["raw_sha256"] = inventory[1]["raw_sha256"]
    inventory[3]["pixel_sha256"] = inventory[2]["pixel_sha256"]
    grouped = data.group_sources(rows, inventory)
    assert [r["aliases"] for r in grouped] == [[1, 2, 3], [4]]
    selected = data.select_groups(grouped, count=2)
    assert [r["image_id"] for r in selected] == [1, 4]
    assert data.select_groups(list(reversed(grouped)), count=2) == selected
    with pytest.raises(ValueError, match="eligible"):
        data.select_groups(grouped, count=3)


def test_photo_id_grouping_is_separate_from_bytes():
    c, i = fixture()
    rows = data.join_source(c, i)
    rows[1]["flickr_url"] = rows[0]["flickr_url"]
    inventory = {j: {"raw_sha256": f"raw{j}", "pixel_sha256": f"pixel{j}"}
                 for j in range(1, 5)}
    assert data.group_sources(rows, inventory)[0]["aliases"] == [1, 2]


def test_archive_members_dimensions_and_byte_pixel_identity(tmp_path):
    c, i = fixture()
    rows = data.join_source(c, i)
    path = tmp_path / "val2017.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for row in rows:
            buf = io.BytesIO()
            Image.new("RGB", (8, 6), "red").save(buf, "PNG")
            archive.writestr(f"val2017/{row['file_name']}", buf.getvalue())
    inventory = data.inventory_zip(path, rows)
    assert len(inventory) == 4
    assert len(set(r["pixel_sha256"] for r in inventory.values())) == 1
    assert len(data.group_sources(rows, inventory)) == 1
    rows[0]["width"] = 9
    with pytest.raises(ValueError, match="dimensions"):
        data.inventory_zip(path, rows)


def test_missing_archive_source_fails_without_substitution(tmp_path):
    c, i = fixture()
    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w"):
        pass
    with pytest.raises(ValueError, match="members"):
        data.inventory_zip(path, data.join_source(c, i))
