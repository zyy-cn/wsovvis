from __future__ import annotations

import json
from types import SimpleNamespace

from videocutler.run_stageb_export import _export_rel_path, _records_from_gt_annotations


def test_run_stageb_export_gt_records_from_official_annotations(tmp_path, monkeypatch):
    lvvis_root = tmp_path / "LV-VIS"
    (lvvis_root / "annotations").mkdir(parents=True)
    ann = {
        "videos": [
            {
                "id": 7,
                "width": 640,
                "height": 360,
                "length": 3,
                "file_names": ["a.jpg", "b.jpg", "c.jpg"],
            }
        ],
        "annotations": [
            {
                "id": 101,
                "video_id": 7,
                "category_id": 3,
                "segmentations": [
                    {"counts": "aaa", "size": [360, 640]},
                    None,
                    {"counts": "bbb", "size": [360, 640]},
                ],
                "bboxes": [
                    [10, 20, 30, 40],
                    None,
                    [11, 21, 31, 41],
                ],
            },
            {
                "id": 102,
                "video_id": 7,
                "category_id": 5,
                "segmentations": [
                    None,
                    {"counts": "ccc", "size": [360, 640]},
                    None,
                ],
                "bboxes": [
                    None,
                    [1, 2, 3, 4],
                    None,
                ],
            },
        ],
        "categories": [],
    }
    (lvvis_root / "annotations" / "val_instances.json").write_text(json.dumps(ann), encoding="utf-8")
    monkeypatch.setenv("WSOVVIS_LVVIS_ROOT", str(lvvis_root))
    args = SimpleNamespace(
        dataset_name="lvvis_val",
        smoke=False,
        generator_ckpt="weights/videocutler_m2f_rn50.pth",
        trajectory_source_branch="gt_upper_bound",
        raw_results_json=None,
    )

    records = _records_from_gt_annotations(args)

    assert len(records) == 2
    assert all(record["dataset_name"] == "lvvis_val" for record in records)
    assert all(record["split_tag"] == "val" for record in records)
    assert all(record["clip_id"] == 7 for record in records)
    assert all(record["video_id"] == 7 for record in records)
    assert [record["rank_in_clip"] for record in records] == [0, 1]
    assert records[0]["trajectory_id"].endswith(":000000")
    assert records[1]["trajectory_id"].endswith(":000001")
    assert all(record["image_size"] == [360, 640] for record in records)
    assert records[0]["boxes_xyxy"][0] == [10.0, 20.0, 40.0, 60.0]
    assert records[1]["boxes_xyxy"][0] == [1.0, 2.0, 4.0, 6.0]


def test_export_rel_path_uses_exports_gt_for_gt_branch():
    assert _export_rel_path("lvvis_train_base", "mainline").as_posix() == "exports/lvvis_train_base/trajectory_records.jsonl"
    assert _export_rel_path("lvvis_val", "gt_upper_bound").as_posix() == "exports_gt/lvvis_val/trajectory_records.jsonl"
