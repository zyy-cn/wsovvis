from __future__ import annotations

import json
from pathlib import Path

from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import (
    Phase1MaterializationConfig,
    materialize_phase1_training_samples,
)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prepare_fixture(tmp_path: Path) -> Path:
    root = tmp_path
    _write_jsonl(
        root / "exports" / "lvvis_train_base" / "trajectory_records.jsonl",
        [
            {
                "dataset_name": "lvvis_train_base",
                "split_tag": "train",
                "clip_id": 101,
                "video_id": 101,
                "rank_in_clip": 0,
                "trajectory_id": "traj_000001",
                "generator_tag": "videocutler_r50_native",
                "pred_score": 0.9,
                "frame_indices": [1, 2],
                "masks_rle": [{}, {}],
                "boxes_xyxy": [[0, 0, 10, 10], [0, 0, 10, 10]],
                "valid_carrier": True,
                "invalid_reason": None,
                "image_size": [100, 100],
            },
            {
                "dataset_name": "lvvis_train_base",
                "split_tag": "train",
                "clip_id": 102,
                "video_id": 102,
                "rank_in_clip": 0,
                "trajectory_id": "traj_000002",
                "generator_tag": "videocutler_r50_native",
                "pred_score": 0.8,
                "frame_indices": [3],
                "masks_rle": [{}],
                "boxes_xyxy": [[0, 0, 10, 10]],
                "valid_carrier": True,
                "invalid_reason": None,
                "image_size": [100, 100],
            },
        ],
    )
    _write_jsonl(
        root / "carrier_bank" / "lvvis_train_base" / "carrier_records.jsonl",
        [
            {
                "trajectory_id": "traj_000001",
                "clip_id": "101",
                "frame_indices": [1, 2],
                "z_raw_path": "carrier_vectors_traj.npz#z_raw[0]",
                "z_norm_path": "carrier_vectors_traj.npz#z_norm[0]",
                "frame_carriers_norm_paths": [
                    "carrier_vectors_frame.npz#z_norm[0]",
                    "carrier_vectors_frame.npz#z_norm[1]",
                ],
                "path_base_mode": "artifact_parent_dir",
            }
        ],
    )
    _write_json(
        root / "weak_labels" / "weak_labels_train.json",
        [
            {
                "clip_id": "101",
                "video_id": 101,
                "observed_raw_ids": [1, 2],
                "observation_protocol_id": "keep80_seed42",
                "completeness_status": "unknown",
            }
        ],
    )
    _write_jsonl(
        root / "frame_bank" / "lvvis_train_base" / "frame_records.jsonl",
        [
            {"clip_id": "101", "frame_index": 1, "feat_path": "payload/clip_101_feats.npz#0", "path_base_mode": "artifact_parent_dir"},
            {"clip_id": "101", "frame_index": 2, "feat_path": "payload/clip_101_feats.npz#1", "path_base_mode": "artifact_parent_dir"},
            {"clip_id": "102", "frame_index": 3, "feat_path": "payload/clip_102_feats.npz#0", "path_base_mode": "artifact_parent_dir"},
        ],
    )
    _write_jsonl(
        root / "frame_bank" / "lvvis_train_base" / "frame_geom_records.jsonl",
        [
            {
                "clip_id": "101",
                "frame_index": 1,
                "orig_h": 100,
                "orig_w": 100,
                "resized_h": 100,
                "resized_w": 100,
                "padded_h": 112,
                "padded_w": 112,
                "scale_y": 1.0,
                "scale_x": 1.0,
                "pad_left": 0,
                "pad_top": 0,
                "pad_right": 12,
                "pad_bottom": 12,
                "patch_size": 14,
                "grid_h": 8,
                "grid_w": 8,
                "valid_token_mask_path": "frame_geom_records.jsonl#0",
                "path_base_mode": "artifact_parent_dir",
            },
            {
                "clip_id": "101",
                "frame_index": 2,
                "orig_h": 100,
                "orig_w": 100,
                "resized_h": 100,
                "resized_w": 100,
                "padded_h": 112,
                "padded_w": 112,
                "scale_y": 1.0,
                "scale_x": 1.0,
                "pad_left": 0,
                "pad_top": 0,
                "pad_right": 12,
                "pad_bottom": 12,
                "patch_size": 14,
                "grid_h": 8,
                "grid_w": 8,
                "valid_token_mask_path": "frame_geom_records.jsonl#1",
                "path_base_mode": "artifact_parent_dir",
            },
            {
                "clip_id": "102",
                "frame_index": 3,
                "orig_h": 100,
                "orig_w": 100,
                "resized_h": 100,
                "resized_w": 100,
                "padded_h": 112,
                "padded_w": 112,
                "scale_y": 1.0,
                "scale_x": 1.0,
                "pad_left": 0,
                "pad_top": 0,
                "pad_right": 12,
                "pad_bottom": 12,
                "patch_size": 14,
                "grid_h": 8,
                "grid_w": 8,
                "valid_token_mask_path": "frame_geom_records.jsonl#2",
                "path_base_mode": "artifact_parent_dir",
            },
        ],
    )
    _write_jsonl(
        root / "text_bank" / "text_prototype_records.jsonl",
        [
            {"raw_id": 1, "proto_path": "payload/text_prototypes.npz#protos[0]", "path_base_mode": "artifact_parent_dir"},
            {"raw_id": 2, "proto_path": "payload/text_prototypes.npz#protos[1]", "path_base_mode": "artifact_parent_dir"},
        ],
    )
    return root


def test_phase1_materialization_deterministic(tmp_path: Path) -> None:
    root = _prepare_fixture(tmp_path)
    result1 = materialize_phase1_training_samples(
        root,
        Phase1MaterializationConfig(dataset_name="lvvis_train_base", smoke=True, smoke_max_trajectories=16),
    )
    result2 = materialize_phase1_training_samples(
        root,
        Phase1MaterializationConfig(dataset_name="lvvis_train_base", smoke=True, smoke_max_trajectories=16),
    )
    assert result1["stats"]["determinism_ok"] is True
    assert result2["stats"]["determinism_ok"] is True
    assert result1["stats"]["determinism_hash_a"] == result2["stats"]["determinism_hash_a"]
    assert result1["samples"][0]["trajectory_id"] == "traj_000001"


def test_phase1_materialization_flags_missing_views(tmp_path: Path) -> None:
    root = _prepare_fixture(tmp_path)
    result = materialize_phase1_training_samples(
        root,
        Phase1MaterializationConfig(dataset_name="lvvis_train_base", smoke=True, smoke_max_trajectories=16),
    )
    by_tid = {sample["trajectory_id"]: sample for sample in result["samples"]}
    assert by_tid["traj_000001"]["sample_valid"] is True
    assert by_tid["traj_000002"]["sample_valid"] is False
    assert "missing_carrier_record" in by_tid["traj_000002"]["invalid_reasons"]
    assert "missing_weak_label_record" in by_tid["traj_000002"]["invalid_reasons"]

