import json
import math
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from wsovvis.track_feature_export import (
    ExportContractError,
    build_track_feature_export_v1_from_stageb_bridge_input,
    convert_stageb_bridge_input_to_task_input_v1,
    validate_track_feature_export_v1,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_BUILD_CLI = REPO_ROOT / "tools" / "build_stageb_track_feature_export_from_stageb_bridge_v1.py"


def _producer(split: str = "train") -> dict:
    return {
        "stage_b_checkpoint_id": "ckpt_001",
        "stage_b_checkpoint_hash": "sha256:a",
        "stage_b_config_ref": "configs/stage_b.yaml",
        "stage_b_config_hash": "sha256:b",
        "pseudo_tube_manifest_id": "ptube_v1",
        "pseudo_tube_manifest_hash": "sha256:c",
        "split": split,
        "extraction_settings": {
            "frame_sampling_rule": "uniform_stride_2",
            "pooling_rule": "mean_over_active_frames",
            "min_track_length": 1,
        },
    }


def _base_bridge_payload() -> dict:
    return {
        "split": "train",
        "producer": _producer("train"),
        "split_domain_video_ids": ["vid_a", "vid_b", "vid_c", "vid_d"],
        "embedding_pooling": "track_pooled",
        "stageb_video_results": [
            {
                "video_id": "vid_b",
                "runtime_status": "success",
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
            {
                "video_id": "vid_a",
                "runtime_status": "success",
                "tracks": [],
            },
            {
                "video_id": "vid_c",
                "runtime_status": "failed",
                "tracks": [],
            },
        ],
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True)


def _npz_embeddings(split_root: Path, manifest: dict, video_id: str) -> np.ndarray:
    by_id = {v["video_id"]: v for v in manifest["videos"]}
    rel = by_id[video_id]["track_arrays_path"]
    return np.load(split_root / rel)["embeddings"]


def test_happy_path_mixed_statuses_to_validator_clean_export(tmp_path: Path):
    out_root = tmp_path / "export_train"
    payload = _base_bridge_payload()

    output_path, summary = build_track_feature_export_v1_from_stageb_bridge_input(payload, out_root)
    assert output_path == out_root
    validate_track_feature_export_v1(split_root=out_root)

    manifest = _read_json(out_root / "manifest.v1.json")
    assert [v["video_id"] for v in manifest["videos"]] == ["vid_a", "vid_b", "vid_c", "vid_d"]

    by_id = {v["video_id"]: v for v in manifest["videos"]}
    assert by_id["vid_a"]["status"] == "processed_zero_tracks"
    assert by_id["vid_b"]["status"] == "processed_with_tracks"
    assert by_id["vid_c"]["status"] == "failed"
    assert by_id["vid_d"]["status"] == "unprocessed"

    assert by_id["vid_c"]["track_metadata_path"] is None
    assert by_id["vid_c"]["track_arrays_path"] is None
    assert by_id["vid_d"]["track_metadata_path"] is None
    assert by_id["vid_d"]["track_arrays_path"] is None

    assert summary["total_split_domain_videos"] == 4
    assert summary["stageb_result_records_seen"] == 3
    assert summary["processed_with_tracks_videos"] == 1
    assert summary["processed_zero_tracks_videos"] == 1
    assert summary["failed_videos"] == 1
    assert summary["unprocessed_videos"] == 1


def test_default_normalization_missing_upstream_is_none(tmp_path: Path):
    payload = _base_bridge_payload()
    payload.pop("embedding_normalization", None)

    task_input, _ = convert_stageb_bridge_input_to_task_input_v1(payload)
    assert task_input["embedding_normalization"] == "none"

    out_root = tmp_path / "export_train"
    build_track_feature_export_v1_from_stageb_bridge_input(payload, out_root)
    manifest = _read_json(out_root / "manifest.v1.json")
    assert manifest["embedding_normalization"] == "none"

    embeddings = _npz_embeddings(out_root, manifest, "vid_b")
    expected = np.asarray([[3.0, 3.0, 3.0, 3.0], [7.0, 7.0, 7.0, 7.0]], dtype=np.float32)
    assert np.array_equal(embeddings, expected)


def test_explicit_l2_normalization_path(tmp_path: Path):
    payload = _base_bridge_payload()
    payload["embedding_normalization"] = "l2"
    payload["stageb_video_results"][0]["tracks"][0]["embedding"] = [2.0, 0.0, 0.0, 0.0]
    payload["stageb_video_results"][0]["tracks"][1]["embedding"] = [0.0, 0.0, 0.0, 0.0]

    out_root = tmp_path / "export_train"
    build_track_feature_export_v1_from_stageb_bridge_input(payload, out_root)

    manifest = _read_json(out_root / "manifest.v1.json")
    embeddings = _npz_embeddings(out_root, manifest, "vid_b")

    assert np.allclose(embeddings[0], np.zeros((4,), dtype=np.float32), atol=1e-6)
    assert math.isclose(float(np.linalg.norm(embeddings[1])), 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_duplicate_stageb_video_result_records_hard_fail_and_no_output(tmp_path: Path):
    payload = _base_bridge_payload()
    payload["stageb_video_results"].append(
        {
            "video_id": "vid_b",
            "runtime_status": "success",
            "tracks": [],
        }
    )

    out_root = tmp_path / "export_train"
    with pytest.raises(ExportContractError, match="duplicate Stage-B result"):
        build_track_feature_export_v1_from_stageb_bridge_input(payload, out_root)
    assert not out_root.exists()


def test_non_finite_embeddings_reject_only_with_status_reclassification(tmp_path: Path):
    payload = _base_bridge_payload()
    payload["split_domain_video_ids"] = ["vid_b", "vid_e", "vid_c", "vid_d"]
    payload["stageb_video_results"] = [
        {
            "video_id": "vid_b",
            "runtime_status": "success",
            "tracks": [
                {
                    "track_id": 1,
                    "start_frame_idx": 0,
                    "end_frame_idx": 3,
                    "num_active_frames": 4,
                    "objectness_score": 0.5,
                    "embedding": [1.0, 0.0, 0.0, 0.0],
                },
                {
                    "track_id": 2,
                    "start_frame_idx": 4,
                    "end_frame_idx": 6,
                    "num_active_frames": 3,
                    "objectness_score": 0.7,
                    "embedding": [float("nan"), 1.0, 0.0, 0.0],
                },
            ],
        },
        {
            "video_id": "vid_e",
            "runtime_status": "success",
            "tracks": [
                {
                    "track_id": 9,
                    "start_frame_idx": 0,
                    "end_frame_idx": 1,
                    "num_active_frames": 2,
                    "objectness_score": 0.5,
                    "embedding": [float("inf"), 0.0, 0.0, 0.0],
                }
            ],
        },
        {"video_id": "vid_c", "runtime_status": "failed", "tracks": []},
    ]

    out_root = tmp_path / "export_train"
    _, summary = build_track_feature_export_v1_from_stageb_bridge_input(payload, out_root)
    manifest = _read_json(out_root / "manifest.v1.json")
    by_id = {v["video_id"]: v for v in manifest["videos"]}

    assert by_id["vid_b"]["status"] == "processed_with_tracks"
    assert by_id["vid_e"]["status"] == "processed_zero_tracks"
    assert summary["invalid_tracks_dropped"] == 2


def test_split_domain_reconciliation_hard_fails_for_duplicates_and_extras():
    payload = _base_bridge_payload()
    payload["split_domain_video_ids"] = ["vid_a", "vid_a"]
    with pytest.raises(ExportContractError, match="duplicate video_id"):
        convert_stageb_bridge_input_to_task_input_v1(payload)

    payload = _base_bridge_payload()
    payload["stageb_video_results"].append(
        {"video_id": "vid_extra", "runtime_status": "failed", "tracks": []}
    )
    with pytest.raises(ExportContractError, match="not present in split_domain_video_ids"):
        convert_stageb_bridge_input_to_task_input_v1(payload)


def test_split_wide_track_id_type_consistency_hard_fail():
    payload = _base_bridge_payload()
    payload["split_domain_video_ids"] = ["vid_x", "vid_y"]
    payload["stageb_video_results"] = [
        {
            "video_id": "vid_x",
            "runtime_status": "success",
            "tracks": [
                {
                    "track_id": 1,
                    "start_frame_idx": 0,
                    "end_frame_idx": 1,
                    "num_active_frames": 2,
                    "objectness_score": 0.9,
                    "embedding": [1.0, 1.0, 1.0, 1.0],
                }
            ],
        },
        {
            "video_id": "vid_y",
            "runtime_status": "success",
            "tracks": [
                {
                    "track_id": "2",
                    "start_frame_idx": 2,
                    "end_frame_idx": 3,
                    "num_active_frames": 2,
                    "objectness_score": 0.8,
                    "embedding": [2.0, 2.0, 2.0, 2.0],
                }
            ],
        },
    ]

    with pytest.raises(ExportContractError, match="track_id serialized type must be consistent"):
        convert_stageb_bridge_input_to_task_input_v1(payload)


def test_determinism_with_shuffled_input_orders(tmp_path: Path):
    payload_a = _base_bridge_payload()
    payload_b = deepcopy(payload_a)
    payload_b["split_domain_video_ids"] = ["vid_d", "vid_c", "vid_b", "vid_a"]
    payload_b["stageb_video_results"] = list(reversed(payload_b["stageb_video_results"]))
    payload_b["stageb_video_results"][1]["tracks"] = list(
        reversed(payload_b["stageb_video_results"][1]["tracks"])
    )

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    _, summary_a = build_track_feature_export_v1_from_stageb_bridge_input(payload_a, out_a)
    _, summary_b = build_track_feature_export_v1_from_stageb_bridge_input(payload_b, out_b)

    manifest_a = _read_json(out_a / "manifest.v1.json")
    manifest_b = _read_json(out_b / "manifest.v1.json")
    assert manifest_a == manifest_b
    assert summary_a == summary_b

    emb_a = _npz_embeddings(out_a, manifest_a, "vid_b")
    emb_b = _npz_embeddings(out_b, manifest_b, "vid_b")
    assert np.array_equal(emb_a, emb_b)


def test_runtime_status_alias_is_rejected():
    payload = _base_bridge_payload()
    payload["stageb_video_results"][0]["runtime_status"] = "ok"
    with pytest.raises(ExportContractError, match="runtime_status"):
        convert_stageb_bridge_input_to_task_input_v1(payload)


def test_cli_smoke_help_and_build_with_summary(tmp_path: Path):
    help_out = _run([sys.executable, str(BRIDGE_BUILD_CLI), "--help"])
    assert help_out.returncode == 0

    payload = _base_bridge_payload()
    input_json = tmp_path / "bridge_input.json"
    out_root = tmp_path / "export_train"
    summary_json = tmp_path / "bridge_summary.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")

    run = _run(
        [
            sys.executable,
            str(BRIDGE_BUILD_CLI),
            "--input-json",
            str(input_json),
            "--output-split-root",
            str(out_root),
            "--bridge-summary-json",
            str(summary_json),
        ]
    )
    assert run.returncode == 0, run.stderr
    assert (out_root / "manifest.v1.json").exists()
    summary = _read_json(summary_json)
    assert summary["total_split_domain_videos"] == 4
    assert summary["duplicate_stageb_result_records"] == 0
