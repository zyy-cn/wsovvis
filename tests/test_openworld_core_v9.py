from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wsovvis.attribution.openworld_core_v9 import (
    OpenWorldCoreConfig,
    build_openworld_core_v9,
    build_openworld_core_v9_worked_example,
    summarize_openworld_core_v9,
)


def _write_semantic_cache(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    videos = {
        "10": [
            {"global_track_id": 0, "track_id": 100, "z_tau": [1.0, 0.0, 0.0], "o_tau": 0.95},
            {"global_track_id": 1, "track_id": 101, "z_tau": [0.0, 1.0, 0.0], "o_tau": 0.94},
        ],
        "11": [
            {"global_track_id": 0, "track_id": 200, "z_tau": [0.9, 0.1, 0.0], "o_tau": 0.93},
            {"global_track_id": 1, "track_id": 201, "z_tau": [0.0, 0.0, 1.0], "o_tau": 0.10},
        ],
    }
    manifest_videos = []
    for video_id, tracks in videos.items():
        video_dir = root / "videos" / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        metadata_tracks = []
        z_tau_rows = []
        row_index = []
        for index, track in enumerate(tracks):
            metadata_tracks.append(
                {
                    "row_index": index,
                    "global_track_id": int(track["global_track_id"]),
                    "start_frame_idx": 0,
                    "end_frame_idx": 4,
                    "num_active_frames": 5,
                    "member_count": 1,
                    "member_track_ids": [int(track["track_id"])],
                    "representative_source_track_id": int(track["track_id"]),
                    "source_track_objectness_score": float(track["o_tau"]),
                    "o_tau": float(track["o_tau"]),
                    "o_tau_components": {
                        "mask_score_mean": 0.2,
                        "duration_ratio": 1.0,
                        "temporal_consistency": 1.0,
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
            "split": "val",
            "video_id": video_id,
            "num_global_tracks": len(metadata_tracks),
            "semantic_tracks": metadata_tracks,
        }
        (video_dir / "semantic_track_metadata.v1.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        np.savez_compressed(
            video_dir / "semantic_track_arrays.v1.npz",
            z_tau=np.asarray(z_tau_rows, dtype=np.float32),
            semantic_track_row_index=np.asarray(row_index, dtype=np.int64),
        )
        manifest_videos.append(
            {
                "video_id": video_id,
                "status": "processed_with_tracks",
                "num_global_tracks": len(metadata_tracks),
                "semantic_track_metadata_path": f"videos/{video_id}/semantic_track_metadata.v1.json",
                "semantic_track_arrays_path": f"videos/{video_id}/semantic_track_arrays.v1.npz",
            }
        )
    manifest = {
        "schema_name": "wsovvis.track_dino_semantic_cache",
        "schema_version": "1.0.0",
        "split": "val",
        "embedding_dim": 3,
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


def _write_text_map_root(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    mapped = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(root / "mapped_text_prototype_arrays.v1.npz", prototypes=mapped)
    np.savez_compressed(
        root / "text_map_state.v1.npz",
        A=np.eye(3, dtype=np.float32),
        text_features=np.eye(3, dtype=np.float32),
        mapped_text_prototypes=mapped,
    )
    labels = [
        {"label_id": 1, "label_text": "alpha", "row_index": 0, "prompt_text": "a photo of alpha", "support_video_count": 2, "support_track_count": 2},
        {"label_id": 2, "label_text": "beta", "row_index": 1, "prompt_text": "a photo of beta", "support_video_count": 1, "support_track_count": 1},
        {"label_id": 3, "label_text": "gamma", "row_index": 2, "prompt_text": "a photo of gamma", "support_video_count": 1, "support_track_count": 1},
    ]
    mapped_manifest = {
        "schema_name": "wsovvis.stagec.label_prototypes.v1",
        "schema_version": "1.0.0",
        "prototype_source": "g5_mapped_text_prototypes_v9",
        "split": "val",
        "embedding_dim": 3,
        "dtype": "float32",
        "array_key": "prototypes",
        "arrays_path": "mapped_text_prototype_arrays.v1.npz",
        "producer": {"prototype_bank_root_rel": "../prototype_bank"},
        "labels": labels,
    }
    text_map_manifest = {
        "schema_name": "wsovvis.text_map_v9",
        "schema_version": "1.0.0",
        "split": "val",
        "text_embedding_dim": 3,
        "mapped_embedding_dim": 3,
        "dtype": "float32",
        "state_arrays_path": "text_map_state.v1.npz",
        "mapped_text_manifest_path": "mapped_text_prototype_manifest.v1.json",
        "prototype_bank_root_rel": "../prototype_bank",
        "selected_label_id": 1,
        "producer": {"text_model_name": "fake"},
        "alignment_metrics": {"num_labels_aligned": 3, "top1_retrieval_accuracy": 1.0},
        "labels": labels,
    }
    (root / "mapped_text_prototype_manifest.v1.json").write_text(json.dumps(mapped_manifest, indent=2), encoding="utf-8")
    (root / "text_map_manifest.v1.json").write_text(json.dumps(text_map_manifest, indent=2), encoding="utf-8")
    return root


def _write_protocol(tmp_path: Path) -> tuple[Path, Path]:
    output = {
        "version": "wsovvis-labelset-protocol-v1",
        "protocol": "long_tail",
        "missing_rate": 0.4,
        "seed": 42,
        "clips": [
            {"video_id": 10, "label_set_full_ids": [1, 2], "label_set_observed_ids": [1], "num_full": 2, "num_observed": 1},
            {"video_id": 11, "label_set_full_ids": [1], "label_set_observed_ids": [1], "num_full": 1, "num_observed": 1},
        ],
    }
    manifest = {
        "version": "wsovvis-labelset-protocol-v1",
        "category_id_to_name": {
            "1": "alpha",
            "2": "beta",
            "3": "gamma",
        },
    }
    output_path = tmp_path / "protocol_output.json"
    manifest_path = tmp_path / "protocol_manifest.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path, manifest_path


def test_build_openworld_core_v9_improves_hidden_positive_metrics(tmp_path: Path) -> None:
    semantic_root = _write_semantic_cache(tmp_path / "semantic_cache")
    text_map_root = _write_text_map_root(tmp_path / "text_map")
    protocol_output_path, protocol_manifest_path = _write_protocol(tmp_path)

    output_root = build_openworld_core_v9(
        semantic_root,
        text_map_root,
        protocol_output_path,
        protocol_manifest_path,
        tmp_path / "openworld_core",
        overwrite=False,
        config=OpenWorldCoreConfig(
            bg_score_threshold=0.25,
            observed_min_score=0.25,
            unknown_min_score=0.25,
            unknown_margin=0.20,
            unknown_min_objectness=0.70,
        ),
    )
    summary = summarize_openworld_core_v9(output_root)
    assert summary["closed_world_metrics"]["macro_hpr_hidden_positive_only"] == 0.0
    assert summary["open_world_metrics"]["macro_hpr_hidden_positive_only"] == 1.0
    assert summary["open_world_metrics"]["macro_uar_hidden_positive_only"] == 1.0
    assert summary["comparison"]["delta_macro_hpr_hidden_positive_only"] > 0.0
    assert summary["comparison"]["delta_macro_uar_hidden_positive_only"] > 0.0
    assert summary["allocation_metrics"]["open_world"]["unknown_track_count"] >= 1


def test_build_openworld_core_v9_worked_example_reports_cost_matrix(tmp_path: Path) -> None:
    semantic_root = _write_semantic_cache(tmp_path / "semantic_cache")
    text_map_root = _write_text_map_root(tmp_path / "text_map")
    protocol_output_path, protocol_manifest_path = _write_protocol(tmp_path)
    output_root = build_openworld_core_v9(
        semantic_root,
        text_map_root,
        protocol_output_path,
        protocol_manifest_path,
        tmp_path / "openworld_core",
        overwrite=False,
        config=OpenWorldCoreConfig(
            bg_score_threshold=0.25,
            observed_min_score=0.25,
            unknown_min_score=0.25,
            unknown_margin=0.20,
            unknown_min_objectness=0.70,
        ),
        selected_video_id="10",
    )
    worked_example = build_openworld_core_v9_worked_example(output_root, selected_video_id="10")
    assert worked_example["selected_video_id"] == "10"
    assert worked_example["aligned_hidden_positive_label_ids"] == [2]
    assert 2 in worked_example["assignment_matrix"]["column_label_ids"]
    assert any(
        row["open_world_assignment"]["assignment_source"] == "unknown_resolved"
        for row in worked_example["assignment_summary"]
    )
    assert len(worked_example["assignment_matrix"]["cost_matrix"]) == 2
