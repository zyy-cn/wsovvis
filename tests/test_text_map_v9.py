from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wsovvis.semantics.prototype_bank_v9 import PrototypeBankConfig, build_prototype_bank_v9
from wsovvis.semantics.text_map_v9 import (
    TextMapConfig,
    build_text_map_v9,
    build_text_map_v9_worked_example,
    compute_text_map_alignment_metrics_v9,
    render_text_map_alignment_svg,
    summarize_text_map_v9,
)
from wsovvis.track_feature_export.stagec1_attribution_mil_v1 import load_stagec_label_prototype_inventory_v1


def _write_semantic_cache(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for video_id, tracks in {
        "10": [
            {"global_track_id": 0, "z_tau": [1.0, 0.0, 0.0, 0.0], "o_tau": 0.9},
        ],
        "11": [
            {"global_track_id": 0, "z_tau": [0.0, 1.0, 0.0, 0.0], "o_tau": 0.8},
        ],
    }.items():
        video_dir = root / "videos" / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_name": "wsovvis.track_dino_semantic_cache_video",
            "schema_version": "1.0.0",
            "split": "val",
            "video_id": video_id,
            "num_global_tracks": 1,
            "semantic_tracks": [
                {
                    "row_index": 0,
                    "global_track_id": 0,
                    "start_frame_idx": 0,
                    "end_frame_idx": 3,
                    "num_active_frames": 4,
                    "member_count": 1,
                    "member_track_ids": [0],
                    "representative_source_track_id": 0,
                    "source_track_objectness_score": float(tracks[0]["o_tau"]),
                    "o_tau": float(tracks[0]["o_tau"]),
                    "o_tau_components": {
                        "mask_score_mean": 0.2,
                        "duration_ratio": 1.0,
                        "temporal_consistency": 1.0,
                    },
                    "provenance": {
                        "selected_frame_indices": [0, 1, 2],
                        "frame_weights": [1.0, 0.0, 0.0],
                        "mask_bbox_xyxy": [[0.0, 0.0, 1.0, 1.0]] * 3,
                        "crop_box_xyxy": [[0, 0, 2, 2]] * 3,
                    },
                }
            ],
        }
        (video_dir / "semantic_track_metadata.v1.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        np.savez_compressed(
            video_dir / "semantic_track_arrays.v1.npz",
            z_tau=np.asarray([tracks[0]["z_tau"]], dtype=np.float32),
            semantic_track_row_index=np.asarray([0], dtype=np.int64),
        )
    manifest = {
        "schema_name": "wsovvis.track_dino_semantic_cache",
        "schema_version": "1.0.0",
        "split": "val",
        "embedding_dim": 4,
        "embedding_dtype": "float32",
        "embedding_normalization": "l2",
        "embedding_pooling": "visible_area_weighted_mean",
        "producer": {
            "patch_pooling_rule": "mask_aware_patch_mean",
            "frame_weighting_rule": "visible_area_weighted_mean",
            "crop_padding_ratio": 0.1,
        },
        "videos": [
            {
                "video_id": "10",
                "status": "processed_with_tracks",
                "num_global_tracks": 1,
                "semantic_track_metadata_path": "videos/10/semantic_track_metadata.v1.json",
                "semantic_track_arrays_path": "videos/10/semantic_track_arrays.v1.npz",
            },
            {
                "video_id": "11",
                "status": "processed_with_tracks",
                "num_global_tracks": 1,
                "semantic_track_metadata_path": "videos/11/semantic_track_metadata.v1.json",
                "semantic_track_arrays_path": "videos/11/semantic_track_arrays.v1.npz",
            },
        ],
    }
    (root / "manifest.v1.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def _write_protocol(tmp_path: Path) -> tuple[Path, Path]:
    output = {
        "version": "wsovvis-labelset-protocol-v1",
        "protocol": "long_tail",
        "missing_rate": 0.4,
        "seed": 42,
        "clips": [
            {"video_id": 10, "label_set_full_ids": [1], "label_set_observed_ids": [1], "num_full": 1, "num_observed": 1},
            {"video_id": 11, "label_set_full_ids": [2], "label_set_observed_ids": [2], "num_full": 1, "num_observed": 1},
        ],
    }
    manifest = {
        "version": "wsovvis-labelset-protocol-v1",
        "category_id_to_name": {
            "1": "alpha",
            "2": "beta",
        },
    }
    output_path = tmp_path / "protocol_output.json"
    manifest_path = tmp_path / "protocol_manifest.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path, manifest_path


def _build_prototype_bank(tmp_path: Path) -> Path:
    semantic_root = _write_semantic_cache(tmp_path / "semantic_cache")
    protocol_output_path, protocol_manifest_path = _write_protocol(tmp_path)
    return build_prototype_bank_v9(
        semantic_root,
        protocol_output_path,
        protocol_manifest_path,
        tmp_path / "prototype_bank",
        config=PrototypeBankConfig(),
    )


def test_compute_text_map_alignment_metrics_v9_reports_top1_accuracy() -> None:
    visual = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mapped = np.asarray([[0.99, 0.01], [0.02, 0.98]], dtype=np.float32)
    metrics = compute_text_map_alignment_metrics_v9(visual, mapped)
    assert metrics["num_labels_aligned"] == 2
    assert metrics["top1_retrieval_accuracy"] == 1.0
    assert metrics["mean_diagonal_cosine"] > 0.98


def test_build_text_map_v9_emits_stagec_compatible_mapped_prototypes(tmp_path: Path) -> None:
    prototype_bank_root = _build_prototype_bank(tmp_path)

    def fake_backend(label_texts: list[str], config: TextMapConfig) -> np.ndarray:
        assert config.prompt_variant == "default"
        mapping = {
            "alpha": np.asarray([1.0, 0.0], dtype=np.float32),
            "beta": np.asarray([0.0, 1.0], dtype=np.float32),
        }
        return np.stack([mapping[text] for text in label_texts], axis=0).astype(np.float32)

    output_root = build_text_map_v9(
        prototype_bank_root,
        tmp_path / "text_map",
        config=TextMapConfig(ridge_lambda=0.0),
        text_feature_backend=fake_backend,
    )
    summary = summarize_text_map_v9(output_root)
    worked_example = build_text_map_v9_worked_example(output_root, selected_label_id=1)
    inventory = load_stagec_label_prototype_inventory_v1(output_root / "mapped_text_prototype_manifest.v1.json")
    assert inventory.prototypes.shape == (2, 4)
    assert summary["text_map_alignment"]["top1_retrieval_accuracy"] == 1.0
    assert summary["text_map_alignment"]["mean_diagonal_cosine"] > 0.999
    assert worked_example["selected_label_id"] == 1
    assert worked_example["selected_label_text"] == "alpha"
    assert worked_example["text_map"]["top1_is_correct"] is True
    assert worked_example["text_map"]["nearest_labels"][0]["label_id"] == 1


def test_render_text_map_alignment_svg_includes_selected_label(tmp_path: Path) -> None:
    prototype_bank_root = _build_prototype_bank(tmp_path)

    def fake_backend(label_texts: list[str], config: TextMapConfig) -> np.ndarray:
        mapping = {
            "alpha": np.asarray([1.0, 0.0], dtype=np.float32),
            "beta": np.asarray([0.0, 1.0], dtype=np.float32),
        }
        return np.stack([mapping[text] for text in label_texts], axis=0).astype(np.float32)

    output_root = build_text_map_v9(
        prototype_bank_root,
        tmp_path / "text_map",
        config=TextMapConfig(ridge_lambda=0.0),
        text_feature_backend=fake_backend,
    )
    svg_path = render_text_map_alignment_svg(output_root, tmp_path / "alignment.svg", selected_label_id=2)
    text = svg_path.read_text(encoding="utf-8")
    assert "label 2" in text
    assert "beta" in text
