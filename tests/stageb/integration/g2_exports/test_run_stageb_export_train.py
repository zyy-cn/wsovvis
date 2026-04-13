from __future__ import annotations

from types import SimpleNamespace

from videocutler.ext_stageb_ovvis.data.trajectory_dataset import valid_carrier_records
from videocutler.run_stageb_export import _records_from_smoke_fixture, _split_tag


def test_run_stageb_export_train_smoke_records_are_readable():
    args = SimpleNamespace(dataset_name="lvvis_train_base", smoke=True)
    records = _records_from_smoke_fixture(args.dataset_name)

    assert records
    assert all(record["dataset_name"] == "lvvis_train_base" for record in records)
    assert all(record["split_tag"] == "train_smoke" for record in records)
    assert all("image_size" in record for record in records)
    assert [record["clip_id"] for record in valid_carrier_records(records)] == [2232, 2232]
    assert _split_tag(args.dataset_name, args.smoke) == "train_smoke"
