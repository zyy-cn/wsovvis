from __future__ import annotations

from videocutler.ext_stageb_ovvis.banks.trajectory_bank import (
    GENERATOR_CFG_PATH,
    GENERATOR_TAG,
    materialize_trajectory_bank,
    validate_trajectory_record,
)
from videocutler.ext_stageb_ovvis.data.trajectory_dataset import read_trajectory_records


def test_schema_and_ordering_are_stable():
    records = read_trajectory_records("package/assets/fixtures/tiny_lvvis_pipeline_case/trajectory_records_train_min.jsonl")
    materialized = materialize_trajectory_bank(records)

    assert [record["rank_in_clip"] for record in materialized[:2]] == [0, 1]
    assert materialized[0]["pred_score"] >= materialized[1]["pred_score"]
    assert all(record["generator_tag"] == GENERATOR_TAG for record in materialized)
    assert all(record["generator_cfg_path"] == GENERATOR_CFG_PATH for record in materialized)
    assert all(not validate_trajectory_record(record) for record in materialized)


def test_trajectory_ids_are_unique():
    records = read_trajectory_records("package/assets/fixtures/tiny_lvvis_pipeline_case/trajectory_records_val_min.jsonl")
    materialized = materialize_trajectory_bank(records)

    trajectory_ids = [record["trajectory_id"] for record in materialized]
    assert trajectory_ids == sorted(trajectory_ids)
    assert len(trajectory_ids) == len(set(trajectory_ids))
