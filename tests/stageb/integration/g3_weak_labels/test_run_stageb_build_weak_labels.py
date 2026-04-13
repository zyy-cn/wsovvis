from __future__ import annotations

import json
import subprocess
import sys

from videocutler.ext_stageb_ovvis.data.weak_labels import build_label_map_from_text_prototypes, build_weak_labels, read_json, read_jsonl


def _augment_expected(records, *, run_scope, input_source_type, data_scope, consumer_target, consumer_ready):
    out = []
    for record in records:
        item = dict(record)
        item["run_scope"] = run_scope
        item["input_source_type"] = input_source_type
        item["data_scope"] = data_scope
        item["consumer_target"] = consumer_target
        item["record_count"] = len(records)
        item["coverage_ratio"] = 1.0 if records else 0.0
        item["consumer_ready"] = consumer_ready
        out.append(item)
    return out


def test_run_stageb_build_weak_labels_fixture_contract():
    videos = [
        {"video_id": 2232, "full_raw_ids": [101, 104, 106]},
        {"video_id": 2233, "full_raw_ids": [101, 106]},
    ]
    label_map = build_label_map_from_text_prototypes(
        read_jsonl("package/assets/fixtures/tiny_lvvis_pipeline_case/text_prototype_records_min.jsonl")
    )
    expected = read_json("package/assets/fixtures/tiny_lvvis_pipeline_case/weak_labels_train.keep60_seed42.json")

    actual = build_weak_labels(
        videos,
        dataset_name="lvvis_train_base",
        split_tag="train_smoke",
        protocol_id="keep60_seed42",
        label_map=label_map,
        seed=42,
    )

    assert actual == _augment_expected(
        expected,
        run_scope="smoke",
        input_source_type="smoke_fixture",
        data_scope="train_smoke",
        consumer_target="downstream_train",
        consumer_ready=False,
    )

def test_run_stageb_build_weak_labels_cli_writes_payload_and_contract(tmp_path):
    out_root = tmp_path / "g3_outputs"
    contract_path = out_root / "weak_label_contract_check.json"
    cmd = [
        sys.executable,
        "videocutler/run_stageb_build_weak_labels.py",
        "--exp_name",
        "stageb_smoke",
        "--dataset_name",
        "lvvis_train_base",
        "--protocol_id",
        "keep60_seed42",
        "--output_root",
        str(out_root),
        "--seed",
        "42",
        "--input_json",
        "tests/stageb/fixtures/g3_weak_labels/smoke_input.json",
        "--class_map_json",
        "package/assets/reference/lvvis_class_map_min.json",
        "--split_tag",
        "train_smoke",
        "--contract_check_json",
        str(contract_path),
    ]
    cp = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr or cp.stdout

    payload_path = out_root / "stageb_smoke" / "weak_labels" / "weak_labels_train.json"
    assert payload_path.exists()
    actual = json.loads(payload_path.read_text())
    expected = read_json("package/assets/fixtures/tiny_lvvis_pipeline_case/weak_labels_train.keep60_seed42.json")
    assert actual == _augment_expected(
        expected,
        run_scope="smoke",
        input_source_type="smoke_fixture",
        data_scope="train_smoke",
        consumer_target="downstream_train",
        consumer_ready=False,
    )

    artifact = json.loads(contract_path.read_text())
    assert artifact["status"] == "PASS"
    assert artifact["deliverables"]["weak_labels_module"] == "videocutler/ext_stageb_ovvis/data/weak_labels.py"
    assert artifact["deliverables"]["weak_labels_payload"] == payload_path.as_posix()
    assert artifact["payload_output"] == payload_path.as_posix()
    assert artifact["payload_record_count"] == len(expected)
