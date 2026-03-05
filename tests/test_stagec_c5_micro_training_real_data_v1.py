from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
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
    assert batch["selected_num_temporal_pairs"] == 0


def test_c5_parser_supports_em_backend_and_defaults() -> None:
    module = _load_c5_script_module()
    parser = module._build_parser()
    args = parser.parse_args(["--assignment-backend", "c9_em_minimal_v1"])
    assert args.assignment_backend == "c9_em_minimal_v1"
    assert args.em_temperature == pytest.approx(0.10)
    assert args.em_iterations == 6


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
    assert batch["selected_num_temporal_pairs"] == 0


def test_real_backed_loader_prefers_non_agnostic_annotation_variant(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "wsovvis_seqformer" / "18"
    sidecar_root = run_root / "d2" / "inference" / "feature_export_v1"
    sidecar_videos = sidecar_root / "videos"
    sidecar_videos.mkdir(parents=True, exist_ok=True)

    agnostic_rel = Path("data/LV-VIS/annotations/lvvis_val_agnostic_for_test.json")
    non_agnostic_rel = Path("data/LV-VIS/annotations/lvvis_val.json")
    _write_json(
        tmp_path / agnostic_rel,
        {
            "videos": [{"id": 1001}, {"id": 1002}],
            "annotations": [
                {"video_id": 1001, "category_id": 1},
                {"video_id": 1002, "category_id": 1},
            ],
            "categories": [{"id": 1, "name": "object"}],
        },
    )
    _write_json(
        tmp_path / non_agnostic_rel,
        {
            "videos": [{"id": 1001}, {"id": 1002}],
            "annotations": [
                {"video_id": 1001, "category_id": 7},
                {"video_id": 1001, "category_id": 8},
                {"video_id": 1002, "category_id": 9},
            ],
            "categories": [{"id": 7, "name": "cat"}, {"id": 8, "name": "dog"}, {"id": 9, "name": "bird"}],
        },
    )
    _write_json(run_root / "config.json", {"data": {"val_json": str(agnostic_rel)}})
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
    )
    assert batch["selected_video_id"] == "1001"
    assert batch["positive_label_ids"] == (7, 8)
    assert batch["split_annotation_json_path"].endswith("lvvis_val.json")


def test_c10c_parser_defaults_ws_metrics_summary_disabled() -> None:
    module = _load_c5_script_module()
    parser = module._build_parser()
    args = parser.parse_args([])
    assert args.emit_ws_metrics_summary_v1 is False
    assert args.ws_metrics_summary_out_json is None


def test_c10c_synthetic_run_can_emit_ws_metrics_summary_artifact(tmp_path: Path) -> None:
    out_json = tmp_path / "micro_summary.json"
    ws_json = tmp_path / "ws_metrics_summary_v1.json"
    proc = subprocess.run(
        [
            sys.executable,
            "tools/run_stagec_c5_micro_training.py",
            "--data-mode",
            "synthetic_v1",
            "--steps",
            "1",
            "--out-json",
            str(out_json),
            "--emit-ws-metrics-summary-v1",
            "--ws-metrics-summary-out-json",
            str(ws_json),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "C5_WS_METRICS_SUMMARY_PATH" in proc.stdout
    run_summary = json.loads(out_json.read_text(encoding="utf-8"))
    ws_summary = json.loads(ws_json.read_text(encoding="utf-8"))
    assert run_summary["ws_metrics_summary_v1_enabled"] is True
    assert run_summary["ws_metrics_summary_v1"]["schema_name"] == "wsovvis.ws_metrics_summary_v1"
    assert ws_summary["schema_name"] == "wsovvis.ws_metrics_summary_v1"
    assert "metrics" in ws_summary


def test_c10c_default_behavior_no_ws_metrics_summary_when_flag_disabled(tmp_path: Path) -> None:
    out_json = tmp_path / "micro_summary_default.json"
    proc = subprocess.run(
        [
            sys.executable,
            "tools/run_stagec_c5_micro_training.py",
            "--data-mode",
            "synthetic_v1",
            "--steps",
            "1",
            "--out-json",
            str(out_json),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "C5_WS_METRICS_SUMMARY_PATH" not in proc.stdout
    run_summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert run_summary["ws_metrics_summary_v1_enabled"] is False
    assert "ws_metrics_summary_v1" not in run_summary


def test_c11a_synthetic_run_emits_unknown_handling_diagnostics_v1(tmp_path: Path) -> None:
    out_json = tmp_path / "micro_summary_c11a.json"
    subprocess.run(
        [
            sys.executable,
            "tools/run_stagec_c5_micro_training.py",
            "--data-mode",
            "synthetic_v1",
            "--steps",
            "1",
            "--out-json",
            str(out_json),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    run_summary = json.loads(out_json.read_text(encoding="utf-8"))
    diag = run_summary["unknown_handling_diagnostics_v1"]
    assert diag["schema_name"] == "wsovvis.stagec_unknown_handling_diagnostics_v1"
    assert diag["schema_version"] == "1.0"
    assert diag["assignment_backend"] == run_summary["final"]["assignment_backend"]
    assert diag["selected_num_positive_labels"] == run_summary["selected_num_positive_labels"]
    assert "mass" in diag and "distribution" in diag and "coverage" in diag and "losses" in diag
    assert diag["mass"]["bg_mass"] >= 0.0
    assert diag["mass"]["unk_fg_mass"] >= 0.0
    assert diag["mass"]["non_special_mass"] >= 0.0
    assert "unk_vs_bg_ratio" in diag["mass"]


def test_c11a_unknown_compare_tool_smoke(tmp_path: Path) -> None:
    sink_json = tmp_path / "sink.json"
    mil_json = tmp_path / "mil.json"
    compare_json = tmp_path / "compare.json"
    subprocess.run(
        [
            sys.executable,
            "tools/run_stagec_c5_micro_training.py",
            "--data-mode",
            "synthetic_v1",
            "--steps",
            "1",
            "--assignment-backend",
            "c2_sinkhorn_minimal_v1",
            "--out-json",
            str(sink_json),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "tools/run_stagec_c5_micro_training.py",
            "--data-mode",
            "synthetic_v1",
            "--steps",
            "1",
            "--assignment-backend",
            "c9_mil_minimal_v1",
            "--out-json",
            str(mil_json),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    proc = subprocess.run(
        [
            sys.executable,
            "tools/compare_stagec_unknown_handling_v1.py",
            "--inputs",
            str(sink_json),
            str(mil_json),
            "--out-json",
            str(compare_json),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "backend|video_id|selected_num_positive_labels|bg_mass|unk_fg_mass|non_special_mass|" in proc.stdout
    payload = json.loads(compare_json.read_text(encoding="utf-8"))
    assert payload["schema_name"] == "wsovvis.stagec_unknown_handling_compare_v1"
    assert payload["num_rows"] == 2
    assert all("backend_config_echo" in row for row in payload["rows"])
