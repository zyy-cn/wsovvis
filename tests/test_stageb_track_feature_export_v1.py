import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wsovvis.track_feature_export import (
    ExportContractError,
    build_track_feature_export_v1,
    validate_track_feature_export_v1,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_CLI = REPO_ROOT / "tools" / "build_stageb_track_feature_export_v1.py"
VALIDATE_CLI = REPO_ROOT / "tools" / "validate_stageb_track_feature_export_v1.py"


def _base_input() -> dict:
    return {
        "split": "train",
        "embedding_dim": 4,
        "embedding_dtype": "float32",
        "embedding_pooling": "track_pooled",
        "embedding_normalization": "l2",
        "producer": {
            "stage_b_checkpoint_id": "ckpt_001",
            "stage_b_checkpoint_hash": "sha256:a",
            "stage_b_config_ref": "configs/stage_b.yaml",
            "stage_b_config_hash": "sha256:b",
            "pseudo_tube_manifest_id": "ptube_v1",
            "pseudo_tube_manifest_hash": "sha256:c",
            "split": "train",
            "extraction_settings": {
                "frame_sampling_rule": "uniform_stride_2",
                "pooling_rule": "mean_over_active_frames",
                "min_track_length": 1,
            },
        },
        "videos": [
            {
                "video_id": "vid_b",
                "status": "processed_with_tracks",
                "tracks": [
                    {
                        "track_id": 7,
                        "start_frame_idx": 20,
                        "end_frame_idx": 30,
                        "num_active_frames": 11,
                        "objectness_score": 0.6,
                        "embedding": [7.0, 7.0, 7.0, 7.0],
                    },
                    {
                        "track_id": 3,
                        "start_frame_idx": 5,
                        "end_frame_idx": 10,
                        "num_active_frames": 6,
                        "objectness_score": 0.8,
                        "embedding": [3.0, 3.0, 3.0, 3.0],
                    },
                ],
            },
            {"video_id": "vid_a", "status": "processed_zero_tracks", "tracks": []},
            {"video_id": "vid_d", "status": "failed", "tracks": []},
            {"video_id": "vid_c", "status": "unprocessed", "tracks": []},
        ],
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True)


def test_happy_path_mixed_statuses_and_validator_pass(tmp_path: Path):
    out_root = tmp_path / "export_train"
    build_track_feature_export_v1(_base_input(), out_root)

    manifest = _read_json(out_root / "manifest.v1.json")
    assert manifest["schema_name"] == "stage_b_track_feature_export"
    assert manifest["schema_version"] == "1.0.0"

    by_id = {v["video_id"]: v for v in manifest["videos"]}

    assert [v["video_id"] for v in manifest["videos"]] == ["vid_a", "vid_b", "vid_c", "vid_d"]

    zero = by_id["vid_a"]
    assert zero["status"] == "processed_zero_tracks"
    assert zero["num_tracks"] == 0
    assert zero["track_metadata_path"] is not None
    assert zero["track_arrays_path"] is not None

    failed = by_id["vid_d"]
    assert failed["track_metadata_path"] is None
    assert failed["track_arrays_path"] is None

    unprocessed = by_id["vid_c"]
    assert unprocessed["track_metadata_path"] is None
    assert unprocessed["track_arrays_path"] is None

    zero_npz = np.load(out_root / by_id["vid_a"]["track_arrays_path"])
    assert zero_npz["embeddings"].dtype == np.float32
    assert zero_npz["embeddings"].shape == (0, 4)
    assert zero_npz["track_row_index"].dtype == np.int64
    assert zero_npz["track_row_index"].shape == (0,)

    validate_track_feature_export_v1(split_root=out_root)


def test_deterministic_sorting_and_row_index_rewrite(tmp_path: Path):
    payload = _base_input()

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    build_track_feature_export_v1(payload, out1)
    build_track_feature_export_v1(payload, out2)

    m1 = _read_json(out1 / "manifest.v1.json")
    m2 = _read_json(out2 / "manifest.v1.json")
    assert m1 == m2

    video_b_meta_path = out1 / "videos" / "vid_b" / "track_metadata.v1.json"
    video_b_meta = _read_json(video_b_meta_path)
    tracks = video_b_meta["tracks"]
    assert [t["row_index"] for t in tracks] == [0, 1]
    assert [t["track_id"] for t in tracks] == [3, 7]

    npz_1 = np.load(out1 / "videos" / "vid_b" / "track_arrays.v1.npz")
    npz_2 = np.load(out2 / "videos" / "vid_b" / "track_arrays.v1.npz")
    assert np.array_equal(npz_1["track_row_index"], np.array([0, 1], dtype=np.int64))
    assert np.array_equal(npz_1["embeddings"], npz_2["embeddings"])
    assert np.array_equal(npz_1["track_row_index"], npz_2["track_row_index"])


def test_validator_catches_row_alignment_and_count_mismatch(tmp_path: Path):
    out_root = tmp_path / "export_train"
    build_track_feature_export_v1(_base_input(), out_root)

    npz_path = out_root / "videos" / "vid_b" / "track_arrays.v1.npz"
    np.savez(npz_path, embeddings=np.ones((1, 4), dtype=np.float32), track_row_index=np.array([0], dtype=np.int64))

    with pytest.raises(ExportContractError, match="embeddings.shape\\[0\\].*num_tracks"):
        validate_track_feature_export_v1(split_root=out_root)


def test_validator_catches_status_and_path_consistency_failures(tmp_path: Path):
    out_root = tmp_path / "export_train"
    build_track_feature_export_v1(_base_input(), out_root)

    manifest_path = out_root / "manifest.v1.json"
    manifest = _read_json(manifest_path)

    for video in manifest["videos"]:
        if video["video_id"] == "vid_b":
            video["track_metadata_path"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ExportContractError, match="track_metadata_path"):
        validate_track_feature_export_v1(split_root=out_root)

    build_track_feature_export_v1(_base_input(), out_root, overwrite=True)
    manifest = _read_json(manifest_path)
    for video in manifest["videos"]:
        if video["video_id"] == "vid_a":
            video["num_tracks"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ExportContractError, match="processed_zero_tracks"):
        validate_track_feature_export_v1(split_root=out_root)

    build_track_feature_export_v1(_base_input(), out_root, overwrite=True)
    manifest = _read_json(manifest_path)
    for video in manifest["videos"]:
        if video["video_id"] == "vid_d":
            video["track_arrays_path"] = "videos/vid_d/track_arrays.v1.npz"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ExportContractError, match="failed/unprocessed"):
        validate_track_feature_export_v1(split_root=out_root)

    build_track_feature_export_v1(_base_input(), out_root, overwrite=True)
    manifest = _read_json(manifest_path)
    for video in manifest["videos"]:
        if video["video_id"] == "vid_b":
            video["track_arrays_path"] = "/abs/forbidden.npz"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ExportContractError, match="absolute path"):
        validate_track_feature_export_v1(split_root=out_root)


def test_validator_catches_schema_name_or_version_failure(tmp_path: Path):
    out_root = tmp_path / "export_train"
    build_track_feature_export_v1(_base_input(), out_root)

    manifest_path = out_root / "manifest.v1.json"
    manifest = _read_json(manifest_path)
    manifest["schema_name"] = "bad_name"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ExportContractError, match="schema_name"):
        validate_track_feature_export_v1(split_root=out_root)

    build_track_feature_export_v1(_base_input(), out_root, overwrite=True)
    manifest = _read_json(manifest_path)
    manifest["schema_version"] = "2.0.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ExportContractError, match="schema_version"):
        validate_track_feature_export_v1(split_root=out_root)


def test_producer_fail_fast_on_invalid_task_input_and_no_partial_output(tmp_path: Path):
    payload = _base_input()
    payload["videos"][0]["tracks"][0]["embedding"] = [1.0, 2.0]

    out_root = tmp_path / "export_train"
    with pytest.raises(ExportContractError, match="embedding"):
        build_track_feature_export_v1(payload, out_root)
    assert not out_root.exists()

    payload = _base_input()
    payload["videos"][0]["tracks"].append(
        {
            "track_id": 7,
            "start_frame_idx": 0,
            "end_frame_idx": 1,
            "num_active_frames": 2,
            "objectness_score": 0.9,
            "embedding": [1.0, 1.0, 1.0, 1.0],
        }
    )
    with pytest.raises(ExportContractError, match="duplicate track_id"):
        build_track_feature_export_v1(payload, out_root)


def test_cli_smoke_build_validate_help(tmp_path: Path):
    payload = _base_input()
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")
    out_root = tmp_path / "export_train"

    help_build = _run([sys.executable, str(BUILD_CLI), "--help"])
    assert help_build.returncode == 0

    help_validate = _run([sys.executable, str(VALIDATE_CLI), "--help"])
    assert help_validate.returncode == 0

    build = _run(
        [
            sys.executable,
            str(BUILD_CLI),
            "--input-json",
            str(input_json),
            "--output-split-root",
            str(out_root),
        ]
    )
    assert build.returncode == 0, build.stderr

    validate = _run([sys.executable, str(VALIDATE_CLI), "--split-root", str(out_root)])
    assert validate.returncode == 0, validate.stderr

    validate_manifest = _run(
        [sys.executable, str(VALIDATE_CLI), "--manifest", str(out_root / "manifest.v1.json")]
    )
    assert validate_manifest.returncode == 0, validate_manifest.stderr


def test_validator_fails_processed_with_tracks_if_payload_file_missing(tmp_path: Path):
    out_root = tmp_path / "export_train"
    build_track_feature_export_v1(_base_input(), out_root)

    npz_path = out_root / "videos" / "vid_b" / "track_arrays.v1.npz"
    npz_path.unlink()
    with pytest.raises(ExportContractError, match="target file missing"):
        validate_track_feature_export_v1(split_root=out_root)


def test_validator_fails_on_noncanonical_track_order(tmp_path: Path):
    out_root = tmp_path / "export_train"
    build_track_feature_export_v1(_base_input(), out_root)

    meta_path = out_root / "videos" / "vid_b" / "track_metadata.v1.json"
    meta = _read_json(meta_path)
    meta["tracks"] = list(reversed(meta["tracks"]))
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ExportContractError, match="must be sorted"):
        validate_track_feature_export_v1(split_root=out_root)


def test_atomic_overwrite_keeps_previous_on_failure(tmp_path: Path):
    out_root = tmp_path / "export_train"
    valid = _base_input()
    build_track_feature_export_v1(valid, out_root)

    before = _read_json(out_root / "manifest.v1.json")
    bad = _base_input()
    bad["videos"][0]["tracks"][0]["end_frame_idx"] = -1

    with pytest.raises(ExportContractError):
        build_track_feature_export_v1(bad, out_root, overwrite=True)

    after = _read_json(out_root / "manifest.v1.json")
    assert before == after
