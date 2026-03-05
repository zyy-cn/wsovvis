from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


def _load_c5_script_module():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_stagec_c5_micro_training.py"
    spec = importlib.util.spec_from_file_location("stagec_c5_micro_training_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_real_backed_micro_batch_loader_smoke(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "wsovvis_seqformer" / "18"
    sidecar_root = run_root / "d2" / "inference" / "feature_export_v1"
    sidecar_videos = sidecar_root / "videos"
    sidecar_videos.mkdir(parents=True, exist_ok=True)

    split_json_rel = Path("data/LV-VIS/annotations/lvvis_val_agnostic_for_test.json")
    _write_json(
        tmp_path / split_json_rel,
        {
            "videos": [{"id": 1001}],
            "annotations": [{"video_id": 1001, "category_id": 7}],
            "categories": [{"id": 7, "name": "cat"}],
        },
    )
    _write_json(run_root / "config.json", {"data": {"val_json": str(split_json_rel)}})
    _write_json(
        sidecar_root / "manifest.json",
        {
            "split": "val",
            "embedding_dim": 4,
            "embedding_normalization": "none",
            "video_shards": ["videos/1001.json"],
            "stageb_checkpoint_ref": "d2/model_final.pth",
            "stageb_checkpoint_hash": "sha256:a",
            "stageb_config_ref": "config.json",
            "stageb_config_hash": "sha256:b",
            "pseudo_tube_manifest_ref": "outputs/pseudo.json",
            "pseudo_tube_manifest_hash": "sha256:c",
            "extraction_settings": {"frame_sampling_rule": "x"},
        },
    )
    _write_json(
        sidecar_videos / "1001.json",
        {
            "video_id": 1001,
            "runtime_evidence": {"stageb_completion_marker": "completed", "evidence_source": "instances_predictions.pth"},
            "tracks": [
                {
                    "track_id": 11,
                    "embedding": [0.1, 0.2, 0.3, 0.4],
                    "embedding_normalization": "none",
                    "start_frame_idx": 0,
                    "end_frame_idx": 2,
                    "num_active_frames": 3,
                    "objectness_score": 0.9,
                }
            ],
        },
    )
    torch.save(
        [
            {"video_id": 1001, "track_id": 11, "score": 0.9, "start_frame_idx": 0, "end_frame_idx": 2, "num_active_frames": 3}
        ],
        run_root / "d2" / "inference" / "instances_predictions.pth",
    )

    module = _load_c5_script_module()
    batch = module.load_real_backed_micro_batch(
        run_root=run_root,
        sample_video_limit=4,
        max_tracks=2,
        max_positive_labels=3,
    )
    assert tuple(batch["track_features_tensor"].shape) == (1, 4)
    assert tuple(batch["track_objectness_tensor"].shape) == (1,)
    assert batch["positive_label_ids"] == (7,)
    assert batch["topk_label_ids"] == (7,)
    assert batch["selected_video_id"] == "1001"
    assert batch["selected_num_tracks_used"] == 1


def test_real_backed_loader_respects_min_positive_labels_and_preferred_video(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "wsovvis_seqformer" / "18"
    sidecar_root = run_root / "d2" / "inference" / "feature_export_v1"
    sidecar_videos = sidecar_root / "videos"
    sidecar_videos.mkdir(parents=True, exist_ok=True)

    split_json_rel = Path("data/LV-VIS/annotations/lvvis_val_agnostic_for_test.json")
    _write_json(
        tmp_path / split_json_rel,
        {
            "videos": [{"id": 1001}, {"id": 1002}],
            "annotations": [
                {"video_id": 1001, "category_id": 7},
                {"video_id": 1002, "category_id": 8},
                {"video_id": 1002, "category_id": 9},
            ],
            "categories": [{"id": 7, "name": "cat"}, {"id": 8, "name": "dog"}, {"id": 9, "name": "bird"}],
        },
    )
    _write_json(run_root / "config.json", {"data": {"val_json": str(split_json_rel)}})
    _write_json(
        sidecar_root / "manifest.json",
        {
            "split": "val",
            "embedding_dim": 4,
            "embedding_normalization": "none",
            "video_shards": ["videos/1001.json", "videos/1002.json"],
            "stageb_checkpoint_ref": "d2/model_final.pth",
            "stageb_checkpoint_hash": "sha256:a",
            "stageb_config_ref": "config.json",
            "stageb_config_hash": "sha256:b",
            "pseudo_tube_manifest_ref": "outputs/pseudo.json",
            "pseudo_tube_manifest_hash": "sha256:c",
            "extraction_settings": {"frame_sampling_rule": "x"},
        },
    )
    _write_json(
        sidecar_videos / "1001.json",
        {
            "video_id": 1001,
            "runtime_evidence": {"stageb_completion_marker": "completed", "evidence_source": "instances_predictions.pth"},
            "tracks": [
                {
                    "track_id": 11,
                    "embedding": [0.1, 0.2, 0.3, 0.4],
                    "embedding_normalization": "none",
                    "start_frame_idx": 0,
                    "end_frame_idx": 2,
                    "num_active_frames": 3,
                    "objectness_score": 0.9,
                }
            ],
        },
    )
    _write_json(
        sidecar_videos / "1002.json",
        {
            "video_id": 1002,
            "runtime_evidence": {"stageb_completion_marker": "completed", "evidence_source": "instances_predictions.pth"},
            "tracks": [
                {
                    "track_id": 21,
                    "embedding": [0.4, 0.3, 0.2, 0.1],
                    "embedding_normalization": "none",
                    "start_frame_idx": 0,
                    "end_frame_idx": 2,
                    "num_active_frames": 3,
                    "objectness_score": 0.8,
                }
            ],
        },
    )
    torch.save(
        [
            {"video_id": 1001, "track_id": 11, "score": 0.9, "start_frame_idx": 0, "end_frame_idx": 2, "num_active_frames": 3},
            {"video_id": 1002, "track_id": 21, "score": 0.8, "start_frame_idx": 0, "end_frame_idx": 2, "num_active_frames": 3},
        ],
        run_root / "d2" / "inference" / "instances_predictions.pth",
    )

    module = _load_c5_script_module()
    batch = module.load_real_backed_micro_batch(
        run_root=run_root,
        sample_video_limit=8,
        max_tracks=2,
        max_positive_labels=4,
        min_positive_labels=2,
        preferred_video_id="1002",
    )
    assert batch["selected_video_id"] == "1002"
    assert batch["positive_label_ids"] == (8, 9)
