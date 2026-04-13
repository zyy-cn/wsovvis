from __future__ import annotations

from videocutler.ext_stageb_ovvis.data.trajectory_dataset import (
    filter_records_by_dataset,
    filter_records_by_split,
    group_records_by_clip,
    read_trajectory_records,
    valid_carrier_records,
)


def test_clip_order_and_valid_carrier_filtering():
    records = read_trajectory_records("package/assets/fixtures/tiny_lvvis_pipeline_case/trajectory_records_train_min.jsonl")

    assert [record["clip_id"] for record in records] == [2232, 2232, 2233]
    assert [record["clip_id"] for record in valid_carrier_records(records)] == [2232, 2232]


def test_dataset_and_split_isolation():
    train_records = read_trajectory_records("package/assets/fixtures/tiny_lvvis_pipeline_case/trajectory_records_train_min.jsonl")
    val_records = read_trajectory_records("package/assets/fixtures/tiny_lvvis_pipeline_case/trajectory_records_val_min.jsonl")

    assert [record["dataset_name"] for record in filter_records_by_dataset(train_records, "lvvis_train_base")] == [
        "lvvis_train_base",
        "lvvis_train_base",
        "lvvis_train_base",
    ]
    assert [record["split_tag"] for record in filter_records_by_split(val_records, "val_smoke")] == [
        "val_smoke",
        "val_smoke",
    ]
    assert sorted(group_records_by_clip(train_records)) == [2232, 2233]
