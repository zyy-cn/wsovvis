from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wsovvis.semantics.prototype_bank_v9 import (
    PrototypeBankConfig,
    build_prototype_bank_v9,
    build_prototype_bank_v9_worked_example,
    load_prototype_bank_v9,
    summarize_prototype_bank_v9,
)
from wsovvis.track_feature_export.stagec1_attribution_mil_v1 import load_stagec_label_prototype_inventory_v1


def _write_semantic_cache(
    root: Path,
    *,
    split: str,
    videos: list[dict],
    embedding_dim: int,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest_videos = []
    for video in sorted(videos, key=lambda row: str(row["video_id"])):
        video_id = str(video["video_id"])
        tracks = sorted(video["tracks"], key=lambda row: int(row["global_track_id"]))
        video_dir = root / "videos" / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = video_dir / "semantic_track_metadata.v1.json"
        arrays_path = video_dir / "semantic_track_arrays.v1.npz"
        metadata_tracks = []
        z_tau_rows = []
        row_index = []
        for index, track in enumerate(tracks):
            metadata_tracks.append(
                {
                    "row_index": index,
                    "global_track_id": int(track["global_track_id"]),
                    "start_frame_idx": int(track.get("start_frame_idx", 0)),
                    "end_frame_idx": int(track.get("end_frame_idx", 4)),
                    "num_active_frames": int(track.get("num_active_frames", 5)),
                    "member_count": int(track.get("member_count", 1)),
                    "member_track_ids": [int(track.get("representative_source_track_id", track["global_track_id"]))],
                    "representative_source_track_id": int(
                        track.get("representative_source_track_id", track["global_track_id"])
                    ),
                    "source_track_objectness_score": float(track["o_tau"]),
                    "o_tau": float(track["o_tau"]),
                    "o_tau_components": {
                        "mask_score_mean": float(track.get("mask_score_mean", track["o_tau"] / 4.0)),
                        "duration_ratio": float(track.get("duration_ratio", 1.0)),
                        "temporal_consistency": float(track.get("temporal_consistency", 1.0)),
                    },
                    "provenance": {
                        "selected_frame_indices": [0, 2, 4],
                        "frame_weights": [1.0, 0.0, 0.0],
                        "mask_bbox_xyxy": [[0.0, 0.0, 1.0, 1.0]] * 3,
                        "crop_box_xyxy": [[0, 0, 2, 2]] * 3,
                    },
                }
            )
            z_tau_rows.append(np.asarray(track["z_tau"], dtype=np.float32))
            row_index.append(index)
        metadata = {
            "schema_name": "wsovvis.track_dino_semantic_cache_video",
            "schema_version": "1.0.0",
            "split": split,
            "video_id": video_id,
            "num_global_tracks": len(metadata_tracks),
            "semantic_tracks": metadata_tracks,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        np.savez_compressed(
            arrays_path,
            z_tau=np.asarray(z_tau_rows, dtype=np.float32),
            semantic_track_row_index=np.asarray(row_index, dtype=np.int64),
        )
        metadata_rel = f"videos/{video_id}/semantic_track_metadata.v1.json"
        arrays_rel = f"videos/{video_id}/semantic_track_arrays.v1.npz"
        manifest_videos.append(
            {
                "video_id": video_id,
                "status": "processed_with_tracks" if tracks else "processed_zero_tracks",
                "num_global_tracks": len(metadata_tracks),
                "semantic_track_metadata_path": metadata_rel,
                "semantic_track_arrays_path": arrays_rel,
            }
        )
    manifest = {
        "schema_name": "wsovvis.track_dino_semantic_cache",
        "schema_version": "1.0.0",
        "split": split,
        "embedding_dim": embedding_dim,
        "embedding_dtype": "float32",
        "embedding_normalization": "l2",
        "embedding_pooling": "visible_area_weighted_mean",
        "producer": {
            "patch_pooling_rule": "mask_aware_patch_mean",
            "frame_weighting_rule": "visible_area_weighted_mean",
            "crop_padding_ratio": 0.1,
        },
        "videos": manifest_videos,
    }
    (root / "manifest.v1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def _write_protocol(output_path: Path, manifest_path: Path) -> tuple[Path, Path]:
    protocol_output = {
        "version": "wsovvis-labelset-protocol-v1",
        "protocol": "long_tail",
        "missing_rate": 0.4,
        "seed": 42,
        "clips": [
            {"video_id": 100, "label_set_full_ids": [1], "label_set_observed_ids": [1], "num_full": 1, "num_observed": 1},
            {"video_id": 101, "label_set_full_ids": [1], "label_set_observed_ids": [1], "num_full": 1, "num_observed": 1},
            {"video_id": 102, "label_set_full_ids": [2, 3], "label_set_observed_ids": [2, 3], "num_full": 2, "num_observed": 2},
            {"video_id": 103, "label_set_full_ids": [4], "label_set_observed_ids": [4], "num_full": 1, "num_observed": 1},
        ],
    }
    protocol_manifest = {
        "version": "wsovvis-labelset-protocol-v1",
        "category_id_to_name": {
            "1": "apple",
            "2": "banana",
            "3": "carrot",
            "4": "donut",
        },
    }
    output_path.write_text(json.dumps(protocol_output, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(protocol_manifest, indent=2), encoding="utf-8")
    return output_path, manifest_path


def test_build_prototype_bank_v9_uses_single_label_clips_and_weighted_mean(tmp_path: Path) -> None:
    semantic_root = _write_semantic_cache(
        tmp_path / "semantic_cache",
        split="val",
        embedding_dim=4,
        videos=[
            {
                "video_id": "100",
                "tracks": [
                    {"global_track_id": 0, "z_tau": [1.0, 0.0, 0.0, 0.0], "o_tau": 0.9},
                    {"global_track_id": 1, "z_tau": [0.0, 1.0, 0.0, 0.0], "o_tau": 0.2},
                ],
            },
            {
                "video_id": "101",
                "tracks": [
                    {"global_track_id": 0, "z_tau": [0.70710677, 0.70710677, 0.0, 0.0], "o_tau": 0.3},
                ],
            },
            {
                "video_id": "102",
                "tracks": [
                    {"global_track_id": 0, "z_tau": [0.0, 0.0, 1.0, 0.0], "o_tau": 0.8},
                ],
            },
            {
                "video_id": "103",
                "tracks": [],
            },
        ],
    )
    protocol_output_path, protocol_manifest_path = _write_protocol(
        tmp_path / "protocol_output.json",
        tmp_path / "protocol_manifest.json",
    )
    output_root = build_prototype_bank_v9(
        semantic_root,
        protocol_output_path,
        protocol_manifest_path,
        tmp_path / "prototype_bank",
        overwrite=False,
        config=PrototypeBankConfig(),
    )

    view = load_prototype_bank_v9(output_root)
    summary = summarize_prototype_bank_v9(output_root)
    inventory = load_stagec_label_prototype_inventory_v1(output_root / "prototype_manifest.v1.json")
    assert inventory.prototypes.shape == (1, 4)
    expected = np.asarray([0.98229027, 0.18736555, 0.0, 0.0], dtype=np.float32)
    record = view.get_record(1)
    assert np.allclose(record.prototype, expected, atol=1e-6)
    assert record.metadata.support_video_count == 2
    assert record.metadata.support_track_count == 2
    assert len(record.metadata.support_refs) == 2
    assert [ref.video_id for ref in record.metadata.support_refs] == ["100", "101"]
    assert summary["prototype_bank_coverage"]["num_seen_labels_total"] == 4
    assert summary["prototype_bank_coverage"]["num_labels_with_prototype"] == 1
    assert summary["prototype_bank_coverage"]["eligible_single_label_clips"] == 3
    assert summary["prototype_bank_coverage"]["eligible_single_label_clips_with_tracks"] == 2


def test_build_prototype_bank_worked_example_reports_selected_support(tmp_path: Path) -> None:
    semantic_root = _write_semantic_cache(
        tmp_path / "semantic_cache",
        split="val",
        embedding_dim=4,
        videos=[
            {
                "video_id": "100",
                "tracks": [
                    {"global_track_id": 3, "z_tau": [1.0, 0.0, 0.0, 0.0], "o_tau": 0.5},
                ],
            },
            {
                "video_id": "101",
                "tracks": [
                    {"global_track_id": 2, "z_tau": [0.0, 1.0, 0.0, 0.0], "o_tau": 0.7},
                ],
            },
        ],
    )
    protocol_output = {
        "version": "wsovvis-labelset-protocol-v1",
        "protocol": "uniform",
        "missing_rate": 0.4,
        "seed": 42,
        "clips": [
            {"video_id": 100, "label_set_full_ids": [5], "label_set_observed_ids": [5], "num_full": 1, "num_observed": 1},
            {"video_id": 101, "label_set_full_ids": [5], "label_set_observed_ids": [5], "num_full": 1, "num_observed": 1},
        ],
    }
    protocol_manifest = {"version": "wsovvis-labelset-protocol-v1", "category_id_to_name": {"5": "egg"}}
    (tmp_path / "protocol_output.json").write_text(json.dumps(protocol_output, indent=2), encoding="utf-8")
    (tmp_path / "protocol_manifest.json").write_text(json.dumps(protocol_manifest, indent=2), encoding="utf-8")
    output_root = build_prototype_bank_v9(
        semantic_root,
        tmp_path / "protocol_output.json",
        tmp_path / "protocol_manifest.json",
        tmp_path / "prototype_bank",
    )
    worked_example = build_prototype_bank_v9_worked_example(output_root, selected_label_id=5)
    assert worked_example["selected_label_id"] == 5
    assert worked_example["selected_label_text"] == "egg"
    assert worked_example["representative_support"]["video_id"] == "101"
    assert worked_example["representative_support"]["global_track_id"] == 2
    assert worked_example["nearest_visual_prototypes"][0]["label_id"] == 5
