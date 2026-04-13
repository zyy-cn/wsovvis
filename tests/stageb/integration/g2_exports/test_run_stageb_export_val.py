from __future__ import annotations

import json
from types import SimpleNamespace

from videocutler.run_stageb_export import _records_from_raw_results


def test_run_stageb_export_val_raw_records_rank_and_identity(tmp_path, monkeypatch):
    lvvis_root = tmp_path / "LV-VIS"
    (lvvis_root / "annotations").mkdir(parents=True)
    ann = {
        "videos": [
            {
                "id": 1,
                "width": 1280,
                "height": 720,
                "length": 3,
                "file_names": ["a.jpg", "b.jpg", "c.jpg"],
            }
        ],
        "annotations": [],
        "categories": [],
    }
    (lvvis_root / "annotations" / "val_instances.json").write_text(json.dumps(ann), encoding="utf-8")
    raw_path = tmp_path / "results.json"
    raw = [
        {
            "video_id": 1,
            "score": 0.1,
            "category_id": 1,
            "segmentations": [None, {"counts": "aaa", "size": [720, 1280]}, None],
        },
        {
            "video_id": 1,
            "score": 0.9,
            "category_id": 1,
            "segmentations": [{"counts": "bbb", "size": [720, 1280]}, None, None],
        },
    ]
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setenv("WSOVVIS_LVVIS_ROOT", str(lvvis_root))
    args = SimpleNamespace(dataset_name="lvvis_val", raw_results_json=str(raw_path), generator_ckpt="weights/videocutler_m2f_rn50.pth")

    records = _records_from_raw_results(args)
    assert len(records) == 2
    assert [record["rank_in_clip"] for record in records] == [0, 1]
    assert records[0]["trajectory_id"].endswith(":000000")
    assert records[1]["trajectory_id"].endswith(":000001")
    assert all(record["image_size"] == [720, 1280] for record in records)
