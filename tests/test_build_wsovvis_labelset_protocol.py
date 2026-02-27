import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "build_wsovvis_labelset_protocol.py"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_builder(
    input_json: Path,
    output_json: Path,
    manifest_json: Path,
    missing_rate: float,
    seed: int,
    min_labels_per_clip: int = 1,
    protocol: str = "uniform",
):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-json",
            str(input_json),
            "--output-json",
            str(output_json),
            "--manifest-json",
            str(manifest_json),
            "--protocol",
            protocol,
            "--missing-rate",
            str(missing_rate),
            "--seed",
            str(seed),
            "--min-labels-per-clip",
            str(min_labels_per_clip),
        ],
        text=True,
        capture_output=True,
    )


def _combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return f"{result.stderr}\n{result.stdout}".lower()


def test_deterministic_output_and_all_videos_included(tmp_path: Path):
    input_payload = {
        "videos": [
            {"id": 1, "name": "v1"},
            {"id": 2, "name": "v2"},
            {"id": 3, "name": "v3_no_annotations"},
        ],
        "categories": [
            {"id": 10, "name": "a"},
            {"id": 20, "name": "b"},
            {"id": 30, "name": "c"},
        ],
        "annotations": [
            {"video_id": 1, "category_id": 10},
            {"video_id": 1, "category_id": 20},
            {"video_id": 1, "category_id": 10},
            {"video_id": 2, "category_id": 20},
            {"video_id": 2, "category_id": 30},
        ],
    }
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(input_payload), encoding="utf-8")

    output1 = tmp_path / "nested" / "run1" / "output.json"
    manifest1 = tmp_path / "nested" / "run1" / "manifest.json"
    output2 = tmp_path / "nested" / "run2" / "output.json"
    manifest2 = tmp_path / "nested" / "run2" / "manifest.json"

    run1 = _run_builder(input_json, output1, manifest1, missing_rate=0.5, seed=123)
    run2 = _run_builder(input_json, output2, manifest2, missing_rate=0.5, seed=123)
    assert run1.returncode == 0, run1.stderr
    assert run2.returncode == 0, run2.stderr

    out1 = _read_json(output1)
    out2 = _read_json(output2)
    man1 = _read_json(manifest1)
    man2 = _read_json(manifest2)

    assert out1 == out2
    assert man1 == man2

    clips = out1["clips"]
    assert [c["video_id"] for c in clips] == [1, 2, 3]
    assert clips[2]["label_set_full_ids"] == []
    assert clips[2]["label_set_observed_ids"] == []
    for clip in clips:
        assert clip["label_set_full_ids"] == sorted(clip["label_set_full_ids"])
        assert clip["label_set_observed_ids"] == sorted(clip["label_set_observed_ids"])


def test_invariants_and_keep_cap_formula(tmp_path: Path):
    input_payload = {
        "videos": [{"id": 7, "name": "clip7"}],
        "categories": [
            {"id": 1, "name": "c1"},
            {"id": 2, "name": "c2"},
            {"id": 3, "name": "c3"},
        ],
        "annotations": [
            {"video_id": 7, "category_id": 1},
            {"video_id": 7, "category_id": 2},
            {"video_id": 7, "category_id": 3},
        ],
    }
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(input_payload), encoding="utf-8")
    output_json = tmp_path / "out.json"
    manifest_json = tmp_path / "manifest.json"

    result = _run_builder(
        input_json,
        output_json,
        manifest_json,
        missing_rate=1.0,
        seed=0,
        min_labels_per_clip=10,
    )
    assert result.returncode == 0, result.stderr

    out = _read_json(output_json)
    clip = out["clips"][0]
    full_ids = clip["label_set_full_ids"]
    observed_ids = clip["label_set_observed_ids"]

    assert len(full_ids) == 3
    assert len(observed_ids) == 3
    assert set(observed_ids).issubset(set(full_ids))
    assert observed_ids
    assert clip["num_full"] == len(full_ids)
    assert clip["num_observed"] == len(observed_ids)
    assert full_ids == sorted(set(full_ids))
    assert observed_ids == sorted(set(observed_ids))


def test_fails_fast_on_unknown_video_or_category(tmp_path: Path):
    base = {
        "videos": [{"id": 1, "name": "v1"}],
        "categories": [{"id": 10, "name": "a"}],
    }

    unknown_video = dict(base)
    unknown_video["annotations"] = [{"video_id": 999, "category_id": 10}]
    input_unknown_video = tmp_path / "unknown_video.json"
    input_unknown_video.write_text(json.dumps(unknown_video), encoding="utf-8")
    result_video = _run_builder(
        input_unknown_video,
        tmp_path / "ov_out.json",
        tmp_path / "ov_manifest.json",
        missing_rate=0.5,
        seed=1,
    )
    assert result_video.returncode != 0
    assert "unknown video_id" in result_video.stderr or "unknown video_id" in result_video.stdout

    unknown_category = dict(base)
    unknown_category["annotations"] = [{"video_id": 1, "category_id": 999}]
    input_unknown_category = tmp_path / "unknown_category.json"
    input_unknown_category.write_text(json.dumps(unknown_category), encoding="utf-8")
    result_cat = _run_builder(
        input_unknown_category,
        tmp_path / "oc_out.json",
        tmp_path / "oc_manifest.json",
        missing_rate=0.5,
        seed=1,
    )
    assert result_cat.returncode != 0
    assert "unknown category_id" in result_cat.stderr or "unknown category_id" in result_cat.stdout


def test_rejects_missing_rate_below_zero(tmp_path: Path):
    input_payload = {"videos": [], "categories": [], "annotations": []}
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(input_payload), encoding="utf-8")

    result = _run_builder(
        input_json,
        tmp_path / "out.json",
        tmp_path / "manifest.json",
        missing_rate=-0.1,
        seed=0,
    )
    assert result.returncode != 0
    assert "missing-rate" in _combined_output(result)


def test_rejects_missing_rate_above_one(tmp_path: Path):
    input_payload = {"videos": [], "categories": [], "annotations": []}
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(input_payload), encoding="utf-8")

    result = _run_builder(
        input_json,
        tmp_path / "out.json",
        tmp_path / "manifest.json",
        missing_rate=1.2,
        seed=0,
    )
    assert result.returncode != 0
    assert "missing-rate" in _combined_output(result)


def test_rejects_unsupported_protocol(tmp_path: Path):
    input_payload = {"videos": [], "categories": [], "annotations": []}
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(input_payload), encoding="utf-8")

    result = _run_builder(
        input_json,
        tmp_path / "out.json",
        tmp_path / "manifest.json",
        missing_rate=0.5,
        seed=0,
        protocol="foo",
    )
    assert result.returncode != 0
    assert "unsupported protocol" in _combined_output(result)


def test_rejects_missing_top_level_annotations(tmp_path: Path):
    input_payload = {"videos": [], "categories": []}
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(input_payload), encoding="utf-8")

    result = _run_builder(
        input_json,
        tmp_path / "out.json",
        tmp_path / "manifest.json",
        missing_rate=0.5,
        seed=0,
    )
    assert result.returncode != 0
    assert "annotations" in _combined_output(result)


def test_rejects_annotation_missing_video_id(tmp_path: Path):
    input_payload = {
        "videos": [{"id": 1, "name": "v1"}],
        "categories": [{"id": 10, "name": "a"}],
        "annotations": [{"category_id": 10}],
    }
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(input_payload), encoding="utf-8")

    result = _run_builder(
        input_json,
        tmp_path / "out.json",
        tmp_path / "manifest.json",
        missing_rate=0.5,
        seed=0,
    )
    assert result.returncode != 0
    assert "video_id" in _combined_output(result)


def test_rejects_annotation_missing_category_id(tmp_path: Path):
    input_payload = {
        "videos": [{"id": 1, "name": "v1"}],
        "categories": [{"id": 10, "name": "a"}],
        "annotations": [{"video_id": 1}],
    }
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(input_payload), encoding="utf-8")

    result = _run_builder(
        input_json,
        tmp_path / "out.json",
        tmp_path / "manifest.json",
        missing_rate=0.5,
        seed=0,
    )
    assert result.returncode != 0
    assert "category_id" in _combined_output(result)


def test_long_tail_happy_path_invariants_and_manifest_fields(tmp_path: Path):
    input_payload = {
        "videos": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}],
        "categories": [{"id": 1, "name": "head"}, {"id": 2, "name": "mid"}, {"id": 3, "name": "tail"}],
        "annotations": [
            {"video_id": 1, "category_id": 1},
            {"video_id": 2, "category_id": 1},
            {"video_id": 3, "category_id": 1},
            {"video_id": 4, "category_id": 1},
            {"video_id": 5, "category_id": 1},
            {"video_id": 2, "category_id": 2},
            {"video_id": 4, "category_id": 2},
            {"video_id": 5, "category_id": 3},
        ],
    }
    input_json = tmp_path / "input.json"
    output_json = tmp_path / "out.json"
    manifest_json = tmp_path / "manifest.json"
    input_json.write_text(json.dumps(input_payload), encoding="utf-8")

    result = _run_builder(
        input_json,
        output_json,
        manifest_json,
        protocol="long_tail",
        missing_rate=0.7,
        seed=123,
        min_labels_per_clip=1,
    )
    assert result.returncode == 0, result.stderr

    out = _read_json(output_json)
    manifest = _read_json(manifest_json)

    assert [c["video_id"] for c in out["clips"]] == [1, 2, 3, 4, 5]
    for clip in out["clips"]:
        full_ids = clip["label_set_full_ids"]
        observed_ids = clip["label_set_observed_ids"]
        assert full_ids == sorted(set(full_ids))
        assert observed_ids == sorted(set(observed_ids))
        assert set(observed_ids).issubset(set(full_ids))
        assert clip["num_full"] == len(full_ids)
        assert clip["num_observed"] == len(observed_ids)

    assert manifest["protocol_metadata"]["protocol_name"] == "long_tail"
    assert manifest["protocol_metadata"]["frequency_count_basis"] == "annotation_count"
    assert "drop_rule" in manifest["protocol_metadata"]
    assert "class_stats" in manifest
    assert "1" in manifest["class_stats"]
    assert "missing_count_stats" in manifest
    assert "min_missing_per_nonempty_clip" in manifest["missing_count_stats"]


def test_long_tail_deterministic_output_and_manifest(tmp_path: Path):
    videos = [{"id": i} for i in range(1, 13)]
    categories = [{"id": 1}, {"id": 2}, {"id": 3}]
    annotations = []
    for video in videos:
        vid = video["id"]
        annotations.extend(
            [
                {"video_id": vid, "category_id": 1},
                {"video_id": vid, "category_id": 1},
            ]
        )
        if vid % 2 == 0:
            annotations.append({"video_id": vid, "category_id": 2})
        if vid % 4 == 0:
            annotations.append({"video_id": vid, "category_id": 3})
    input_payload = {"videos": videos, "categories": categories, "annotations": annotations}

    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(input_payload), encoding="utf-8")
    output1 = tmp_path / "run1_out.json"
    output2 = tmp_path / "run2_out.json"
    manifest1 = tmp_path / "run1_manifest.json"
    manifest2 = tmp_path / "run2_manifest.json"

    run1 = _run_builder(
        input_json, output1, manifest1, protocol="long_tail", missing_rate=0.6, seed=99
    )
    run2 = _run_builder(
        input_json, output2, manifest2, protocol="long_tail", missing_rate=0.6, seed=99
    )
    assert run1.returncode == 0, run1.stderr
    assert run2.returncode == 0, run2.stderr
    assert _read_json(output1) == _read_json(output2)
    assert _read_json(manifest1) == _read_json(manifest2)


def test_long_tail_statistical_tendency_tail_not_more_observed_than_head(tmp_path: Path):
    videos = [{"id": i} for i in range(1, 121)]
    categories = [{"id": 1, "name": "head"}, {"id": 2, "name": "mid"}, {"id": 3, "name": "tail"}]
    annotations = []
    for video in videos:
        vid = video["id"]
        annotations.extend(
            [
                {"video_id": vid, "category_id": 1},
                {"video_id": vid, "category_id": 1},
                {"video_id": vid, "category_id": 1},
            ]
        )
        if vid % 2 == 0:
            annotations.append({"video_id": vid, "category_id": 2})
        if vid % 6 == 0:
            annotations.append({"video_id": vid, "category_id": 3})
    input_payload = {"videos": videos, "categories": categories, "annotations": annotations}

    input_json = tmp_path / "input.json"
    output_json = tmp_path / "out.json"
    manifest_json = tmp_path / "manifest.json"
    input_json.write_text(json.dumps(input_payload), encoding="utf-8")

    result = _run_builder(
        input_json,
        output_json,
        manifest_json,
        protocol="long_tail",
        missing_rate=0.7,
        seed=2026,
        min_labels_per_clip=0,
    )
    assert result.returncode == 0, result.stderr
    manifest = _read_json(manifest_json)
    class_stats = manifest["class_stats"]
    head_rate = class_stats["1"]["observed_rate"]
    mid_rate = class_stats["2"]["observed_rate"]
    tail_rate = class_stats["3"]["observed_rate"]
    assert tail_rate <= mid_rate <= head_rate


def test_long_tail_respects_min_labels_per_clip_cap(tmp_path: Path):
    input_payload = {
        "videos": [{"id": 1}, {"id": 2}, {"id": 3}],
        "categories": [{"id": 1}, {"id": 2}, {"id": 3}],
        "annotations": [
            {"video_id": 1, "category_id": 1},
            {"video_id": 1, "category_id": 2},
            {"video_id": 1, "category_id": 3},
            {"video_id": 2, "category_id": 1},
            {"video_id": 2, "category_id": 2},
            {"video_id": 2, "category_id": 3},
            {"video_id": 3, "category_id": 1},
            {"video_id": 3, "category_id": 2},
            {"video_id": 3, "category_id": 3},
        ],
    }
    input_json = tmp_path / "input.json"
    output_json = tmp_path / "out.json"
    manifest_json = tmp_path / "manifest.json"
    input_json.write_text(json.dumps(input_payload), encoding="utf-8")

    result = _run_builder(
        input_json,
        output_json,
        manifest_json,
        protocol="long_tail",
        missing_rate=1.0,
        seed=7,
        min_labels_per_clip=2,
    )
    assert result.returncode == 0, result.stderr
    out = _read_json(output_json)
    for clip in out["clips"]:
        assert clip["num_full"] == 3
        assert clip["num_observed"] >= 2
