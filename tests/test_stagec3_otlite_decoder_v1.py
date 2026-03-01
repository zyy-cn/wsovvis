import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from wsovvis.track_feature_export import StageC1AttributionError, build_track_feature_export_v1, run_stagec1_mil_baseline_offline


def _build_single_video_export(tmp_path: Path, *, tracks: list[dict], embedding_dim: int) -> Path:
    payload = {
        "split": "train",
        "embedding_dim": embedding_dim,
        "embedding_dtype": "float32",
        "embedding_pooling": "track_pooled",
        "embedding_normalization": "none",
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
        "videos": [{"video_id": "vid_a", "status": "processed_with_tracks", "tracks": tracks}],
    }
    out_root = tmp_path / "single_video_export"
    build_track_feature_export_v1(payload, out_root)
    return out_root


def _write_proto_manifest(tmp_path: Path, *, prototypes: np.ndarray, labels: list[dict]) -> Path:
    arrays_path = tmp_path / "prototypes.npz"
    np.savez(arrays_path, prototypes=prototypes.astype(np.float32))
    manifest = {
        "schema_name": "wsovvis.stagec.label_prototypes.v1",
        "schema_version": "1.0.0",
        "embedding_dim": int(prototypes.shape[1]),
        "dtype": "float32",
        "labels": labels,
        "arrays_path": arrays_path.name,
        "array_key": "prototypes",
    }
    path = tmp_path / "prototype_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _write_labelset(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "labelset.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_otlite_decoder_assignment_and_ot_prob_fields(tmp_path: Path) -> None:
    split_root = _build_single_video_export(
        tmp_path,
        embedding_dim=2,
        tracks=[
            {
                "track_id": 1,
                "start_frame_idx": 0,
                "end_frame_idx": 2,
                "num_active_frames": 3,
                "objectness_score": 0.8,
                "embedding": [1.0, 0.0],
            },
            {
                "track_id": 2,
                "start_frame_idx": 3,
                "end_frame_idx": 5,
                "num_active_frames": 3,
                "objectness_score": 0.7,
                "embedding": [0.8, 0.2],
            },
        ],
    )
    manifest_path = _write_proto_manifest(
        tmp_path,
        prototypes=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        labels=[{"label_id": 100, "row_index": 0}, {"label_id": 200, "row_index": 1}],
    )
    labelset_path = _write_labelset(tmp_path, {"videos": [{"video_id": "vid_a", "label_set_observed_ids": [100, 200]}]})
    out_dir = tmp_path / "out_otlite"

    run_stagec1_mil_baseline_offline(
        split_root=split_root,
        output_dir=out_dir,
        scorer_backend="labelset_proto_v1",
        decoder_backend="otlite_v1",
        labelset_json=labelset_path,
        prototype_manifest_json=manifest_path,
    )

    rows = [json.loads(line) for line in (out_dir / "track_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["decoder_assignment_source"] == "otlite"
    assert rows[1]["decoder_assignment_source"] == "otlite"
    assert "decoder_ot_prob" in rows[0]
    assert 0.0 <= rows[0]["decoder_ot_prob"] <= 1.0
    assert 0.0 <= rows[1]["decoder_ot_prob"] <= 1.0

    per_video = json.loads((out_dir / "per_video_summary.json").read_text(encoding="utf-8"))[0]
    assert per_video["decoder_backend"] == "otlite_v1"
    assert "decoder_otlite_col_mass_l1_error" in per_video
    assert "decoder_otlite_row_mass_l1_error" in per_video

    run_summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    decoder_summary = run_summary["decoder_summary"]
    assert decoder_summary["policy_version"] == "otlite_v1.r2a"
    assert decoder_summary["otlite_temperature"] == pytest.approx(0.10)
    assert decoder_summary["otlite_iters"] == 8
    assert decoder_summary["otlite_eps"] == pytest.approx(1e-12)


def test_otlite_bg_reason_ot_prob_min(tmp_path: Path) -> None:
    split_root = _build_single_video_export(
        tmp_path,
        embedding_dim=2,
        tracks=[
            {
                "track_id": 1,
                "start_frame_idx": 0,
                "end_frame_idx": 2,
                "num_active_frames": 3,
                "objectness_score": 0.8,
                "embedding": [1.0, 0.0],
            }
        ],
    )
    manifest_path = _write_proto_manifest(
        tmp_path,
        prototypes=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        labels=[{"label_id": 10, "row_index": 0}, {"label_id": 20, "row_index": 1}],
    )
    labelset_path = _write_labelset(tmp_path, {"videos": [{"video_id": "vid_a", "label_set_observed_ids": [10, 20]}]})

    out_dir = tmp_path / "out_otprob_bg"
    run_stagec1_mil_baseline_offline(
        split_root=split_root,
        output_dir=out_dir,
        scorer_backend="labelset_proto_v1",
        decoder_backend="otlite_v1",
        decoder_otlite_ot_prob_min=0.9,
        labelset_json=labelset_path,
        prototype_manifest_json=manifest_path,
    )

    row = json.loads((out_dir / "track_scores.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["decoder_assigned_bg"] is True
    assert row["decoder_assignment_source"] == "otlite_bg"
    assert row["decoder_bg_reason"] == "ot_prob_min"

    run_summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["decoder_summary"]["bg_reason_counts"]["ot_prob_min"] == 1
    assert run_summary["decoder_summary"]["otlite_bg_ot_prob_count"] == 1


def test_otlite_param_validation_rejected(tmp_path: Path) -> None:
    split_root = _build_single_video_export(
        tmp_path,
        embedding_dim=2,
        tracks=[
            {
                "track_id": 1,
                "start_frame_idx": 0,
                "end_frame_idx": 2,
                "num_active_frames": 3,
                "objectness_score": 0.8,
                "embedding": [1.0, 0.0],
            }
        ],
    )

    with pytest.raises(StageC1AttributionError, match="decoder.otlite_temperature"):
        run_stagec1_mil_baseline_offline(split_root=split_root, output_dir=tmp_path / "bad_t", decoder_otlite_temperature=0.0)
    with pytest.raises(StageC1AttributionError, match="decoder.otlite_iters"):
        run_stagec1_mil_baseline_offline(split_root=split_root, output_dir=tmp_path / "bad_i", decoder_otlite_iters=0)
    with pytest.raises(StageC1AttributionError, match="decoder.otlite_eps"):
        run_stagec1_mil_baseline_offline(split_root=split_root, output_dir=tmp_path / "bad_e", decoder_otlite_eps=0.0)
    with pytest.raises(StageC1AttributionError, match="decoder.otlite_ot_prob_min"):
        run_stagec1_mil_baseline_offline(split_root=split_root, output_dir=tmp_path / "bad_p", decoder_otlite_ot_prob_min=1.1)


def test_otlite_numeric_stability_extreme_scores(tmp_path: Path) -> None:
    split_root = _build_single_video_export(
        tmp_path,
        embedding_dim=3,
        tracks=[
            {
                "track_id": 1,
                "start_frame_idx": 0,
                "end_frame_idx": 2,
                "num_active_frames": 3,
                "objectness_score": 0.8,
                "embedding": [1e20, 1.0, 0.0],
            },
            {
                "track_id": 2,
                "start_frame_idx": 3,
                "end_frame_idx": 5,
                "num_active_frames": 3,
                "objectness_score": 0.8,
                "embedding": [-1e20, 0.0, 1.0],
            },
        ],
    )
    manifest_path = _write_proto_manifest(
        tmp_path,
        prototypes=np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        labels=[{"label_id": 1, "row_index": 0}, {"label_id": 2, "row_index": 1}, {"label_id": 3, "row_index": 2}],
    )
    labelset_path = _write_labelset(tmp_path, {"videos": [{"video_id": "vid_a", "label_set_observed_ids": [1, 2, 3]}]})
    out_dir = tmp_path / "out_stability"

    run_stagec1_mil_baseline_offline(
        split_root=split_root,
        output_dir=out_dir,
        scorer_backend="labelset_proto_v1",
        decoder_backend="otlite_v1",
        decoder_otlite_temperature=0.05,
        labelset_json=labelset_path,
        prototype_manifest_json=manifest_path,
    )

    rows = [json.loads(line) for line in (out_dir / "track_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        assert np.isfinite(float(row["decoder_score"]))
        assert np.isfinite(float(row["decoder_margin"]))
        assert np.isfinite(float(row["decoder_ot_prob"]))


def test_otlite_determinism_double_run_hash(tmp_path: Path) -> None:
    split_root = _build_single_video_export(
        tmp_path,
        embedding_dim=2,
        tracks=[
            {
                "track_id": 1,
                "start_frame_idx": 0,
                "end_frame_idx": 2,
                "num_active_frames": 3,
                "objectness_score": 0.8,
                "embedding": [1.0, 0.0],
            },
            {
                "track_id": 2,
                "start_frame_idx": 3,
                "end_frame_idx": 5,
                "num_active_frames": 3,
                "objectness_score": 0.8,
                "embedding": [0.4, 0.6],
            },
        ],
    )
    manifest_path = _write_proto_manifest(
        tmp_path,
        prototypes=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        labels=[{"label_id": 1, "row_index": 0}, {"label_id": 2, "row_index": 1}],
    )
    labelset_path = _write_labelset(tmp_path, {"videos": [{"video_id": "vid_a", "label_set_observed_ids": [1, 2]}]})

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    common = dict(
        split_root=split_root,
        scorer_backend="labelset_proto_v1",
        decoder_backend="otlite_v1",
        labelset_json=labelset_path,
        prototype_manifest_json=manifest_path,
    )
    run_stagec1_mil_baseline_offline(output_dir=out_a, **common)
    run_stagec1_mil_baseline_offline(output_dir=out_b, **common)

    for name in ("track_scores.jsonl", "per_video_summary.json", "run_summary.json"):
        a = (out_a / name).read_bytes()
        b = (out_b / name).read_bytes()
        assert a == b
        assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()
