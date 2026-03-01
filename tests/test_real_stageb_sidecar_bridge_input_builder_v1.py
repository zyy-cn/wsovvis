import json
from pathlib import Path

import pytest

from wsovvis.track_feature_export import (
    ExportContractError,
    build_normalized_bridge_input_from_real_stageb_sidecar,
)


torch = pytest.importorskip("torch")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_pth(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _base_layout(tmp_path: Path) -> Path:
    repo_root = tmp_path
    run_root = repo_root / "runs" / "wsovvis_seqformer" / "18"
    (run_root / "d2" / "inference" / "feature_export_v1" / "videos").mkdir(parents=True, exist_ok=True)

    split_domain = {
        "videos": [
            {"id": 0},
            {"id": 1},
            {"id": 2},
        ]
    }
    _write_json(repo_root / "data" / "LV-VIS" / "annotations" / "lvvis_val_agnostic_for_test.json", split_domain)

    _write_json(
        run_root / "config.json",
        {
            "data": {
                "val_json": "data/LV-VIS/annotations/lvvis_val_agnostic_for_test.json",
            }
        },
    )

    _write_json(
        run_root / "d2" / "inference" / "feature_export_v1" / "manifest.json",
        {
            "contract_name": "stageb_feature_export_enablement_contract_v1",
            "contract_version": "v1",
            "split": "val",
            "embedding_dim": 4,
            "embedding_dtype": "float32",
            "embedding_normalization": "none",
            "video_shards": ["videos/0.json", "videos/1.json"],
            "stageb_checkpoint_ref": "d2/model_final.pth",
            "stageb_checkpoint_hash": "sha256:a",
            "stageb_config_ref": "config.json",
            "stageb_config_hash": "sha256:b",
            "pseudo_tube_manifest_ref": "outputs/pseudo.json",
            "pseudo_tube_manifest_hash": "sha256:c",
            "extraction_settings": {
                "frame_sampling_rule": "seqformer_runtime_default",
                "pooling_rule": "track_feature_vector_direct",
                "min_track_length": 1,
            },
        },
    )

    _write_json(
        run_root / "d2" / "inference" / "feature_export_v1" / "videos" / "0.json",
        {
            "video_id": 0,
            "runtime_evidence": {
                "stageb_completion_marker": "completed",
                "evidence_source": "instances_predictions.pth",
            },
            "tracks": [
                {
                    "track_id": 10,
                    "embedding": [0.1, 0.2, 0.3, 0.4],
                    "embedding_normalization": "none",
                }
            ],
        },
    )

    _write_json(
        run_root / "d2" / "inference" / "feature_export_v1" / "videos" / "1.json",
        {
            "video_id": 1,
            "runtime_evidence": {
                "stageb_completion_marker": "unknown",
                "evidence_source": "instances_predictions.pth",
            },
            "tracks": [
                {
                    "track_id": 20,
                    "embedding": [1.0, 0.0, 0.0, 0.0],
                    "embedding_normalization": "none",
                    "start_frame_idx": 1,
                    "end_frame_idx": 4,
                    "num_active_frames": 4,
                    "objectness_score": 0.7,
                }
            ],
        },
    )

    _write_pth(
        run_root / "d2" / "inference" / "instances_predictions.pth",
        [
            {
                "video_id": 0,
                "track_id": 10,
                "score": 0.95,
                "start_frame_idx": 2,
                "end_frame_idx": 6,
                "num_active_frames": 5,
            },
            {
                "video_id": 1,
                "track_id": 20,
                "score": 0.75,
                "start_frame_idx": 1,
                "end_frame_idx": 4,
                "num_active_frames": 4,
            },
        ],
    )
    return run_root


def test_builder_happy_path_and_runtime_domain(tmp_path: Path):
    run_root = _base_layout(tmp_path)

    payload, summary = build_normalized_bridge_input_from_real_stageb_sidecar(run_root=run_root)
    assert payload["split"] == "val"
    assert payload["embedding_pooling"] == "track_pooled"
    assert payload["split_domain_video_ids"] == ["0", "1", "2"]
    assert payload["producer"]["stage_b_checkpoint_id"] == "d2/model_final.pth"
    assert payload["producer"]["pseudo_tube_manifest_id"] == "outputs/pseudo.json"

    results = {r["video_id"]: r for r in payload["stageb_video_results"]}
    assert set(results.keys()) == {"0", "1", "2"}
    assert results["0"]["runtime_status"] == "success"
    assert results["0"]["tracks"][0]["start_frame_idx"] == 2
    assert results["0"]["tracks"][0]["objectness_score"] == pytest.approx(0.95, rel=0, abs=1e-6)
    assert results["1"]["runtime_status"] == "failed"
    assert results["1"]["tracks"] == []
    assert results["2"]["runtime_status"] == "failed"

    assert summary["total_join_pairs"] == 2
    assert summary["missing_counterparts"]["prediction_missing_sidecar"] == 0
    assert summary["missing_counterparts"]["sidecar_missing_prediction"] == 0


def test_duplicate_join_key_hard_fails(tmp_path: Path):
    run_root = _base_layout(tmp_path)

    _write_pth(
        run_root / "d2" / "inference" / "instances_predictions.pth",
        [
            {
                "video_id": 0,
                "track_id": 10,
                "score": 0.9,
            },
            {
                "video_id": 0,
                "track_id": 10,
                "score": 0.8,
            },
        ],
    )

    with pytest.raises(ExportContractError, match="duplicate join key in predictions"):
        build_normalized_bridge_input_from_real_stageb_sidecar(run_root=run_root)


def test_one_sided_join_mismatch_hard_fails(tmp_path: Path):
    run_root = _base_layout(tmp_path)

    _write_pth(
        run_root / "d2" / "inference" / "instances_predictions.pth",
        [
            {
                "video_id": 0,
                "track_id": 10,
                "score": 0.95,
            }
        ],
    )

    with pytest.raises(ExportContractError, match=r"sidecar key\(s\) missing prediction counterpart"):
        build_normalized_bridge_input_from_real_stageb_sidecar(run_root=run_root)


def test_non_finite_embedding_rejected_and_dropped(tmp_path: Path):
    run_root = _base_layout(tmp_path)
    _write_json(
        run_root / "d2" / "inference" / "feature_export_v1" / "videos" / "0.json",
        {
            "video_id": 0,
            "runtime_evidence": {
                "stageb_completion_marker": "completed",
                "evidence_source": "instances_predictions.pth",
            },
            "tracks": [
                {
                    "track_id": 10,
                    "embedding": [0.1, 0.2, float("nan"), 0.4],
                    "embedding_normalization": "none",
                    "start_frame_idx": 2,
                    "end_frame_idx": 6,
                    "num_active_frames": 5,
                    "objectness_score": 0.9,
                }
            ],
        },
    )

    payload, summary = build_normalized_bridge_input_from_real_stageb_sidecar(run_root=run_root)
    result0 = [r for r in payload["stageb_video_results"] if r["video_id"] == "0"][0]
    assert result0["runtime_status"] == "success"
    assert result0["tracks"] == []
    assert summary["dropped_tracks"] == 1
    assert summary["non_finite_rejects"] == 1


def test_sample_video_limit_is_deterministic(tmp_path: Path):
    run_root = _base_layout(tmp_path)
    payload, summary = build_normalized_bridge_input_from_real_stageb_sidecar(run_root=run_root, sample_video_limit=2)

    assert payload["split_domain_video_ids"] == ["0", "1"]
    assert [r["video_id"] for r in payload["stageb_video_results"]] == ["0", "1"]
    assert summary["total_split_domain_videos"] == 2
