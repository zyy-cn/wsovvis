from __future__ import annotations

import json

from videocutler.ext_stageb_ovvis.data.weak_labels import build_weak_labels_from_fixture, protocol_ratio, read_json


def test_keep60_exact_k_matches_expected_fixture():
    fixture = read_json("package/assets/fixtures/deterministic_seed_case/exact_k_input.json")
    expected = read_json("package/assets/fixtures/deterministic_seed_case/weak_labels_keep60_expected.json")
    label_map = {
        1: {"contiguous_id": 0, "class_name": "person"},
        6: {"contiguous_id": 5, "class_name": "dog"},
        10: {"contiguous_id": 9, "class_name": "bus"},
    }

    actual = build_weak_labels_from_fixture(
        fixture,
        protocol_id="keep60_seed42",
        label_map=label_map,
        split_tag="train_smoke",
    )

    assert actual == expected
    assert json.loads(json.dumps(actual)) == actual


def test_protocol_family_is_frozen():
    assert protocol_ratio("keep80_seed42") == 0.8
    assert protocol_ratio("keep60_seed42") == 0.6
    assert protocol_ratio("keep40_seed42") == 0.4
