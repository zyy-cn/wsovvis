import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wsovvis.track_feature_export import (
    StageC1AttributionError,
    StageC1MilConfig,
    build_track_feature_export_v1,
    compute_stagec1_mil_baseline_scores,
    load_stageb_export_split_v1,
    run_stagec1_mil_baseline_offline,
)


def _base_input() -> dict:
    return {
        "split": "train",
        "embedding_dim": 4,
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
        "videos": [
            {
                "video_id": "vid_c",
                "status": "processed_with_tracks",
                "tracks": [
                    {
                        "track_id": 9,
                        "start_frame_idx": 0,
                        "end_frame_idx": 3,
                        "num_active_frames": 4,
                        "objectness_score": 0.25,
                        "embedding": [2.0, 2.0, 2.0, 2.0],
                    }
                ],
            },
            {
                "video_id": "vid_a",
                "status": "processed_with_tracks",
                "tracks": [
                    {
                        "track_id": 1,
                        "start_frame_idx": 10,
                        "end_frame_idx": 15,
                        "num_active_frames": 6,
                        "objectness_score": 0.5,
                        "embedding": [1.0, 1.0, 1.0, 1.0],
                    },
                    {
                        "track_id": 2,
                        "start_frame_idx": 20,
                        "end_frame_idx": 25,
                        "num_active_frames": 6,
                        "objectness_score": 0.75,
                        "embedding": [4.0, 4.0, 4.0, 4.0],
                    },
                ],
            },
            {"video_id": "vid_b", "status": "processed_zero_tracks", "tracks": []},
            {"video_id": "vid_d", "status": "failed", "tracks": []},
        ],
    }


def _build_export(tmp_path: Path) -> Path:
    out_root = tmp_path / "export_train"
    build_track_feature_export_v1(_base_input(), out_root)
    return out_root


def _expected_stagec0_track_order(view) -> list[tuple[str, str | int]]:
    keys: list[tuple[str, str | int]] = []
    for video in view.iter_videos():
        if video.status != "processed_with_tracks":
            continue
        for track in view.iter_tracks(video.video_id):
            keys.append((video.video_id, track.metadata.track_id))
    return keys


def test_stagec1_mil_nominal_order_and_finiteness(tmp_path: Path) -> None:
    view = load_stageb_export_split_v1(_build_export(tmp_path))
    result = compute_stagec1_mil_baseline_scores(view)

    emitted_keys = [(r.video_id, r.track_id) for r in result.track_scores]
    assert emitted_keys == _expected_stagec0_track_order(view)
    assert len(set(emitted_keys)) == len(emitted_keys)
    assert all(np.isfinite(r.score) for r in result.track_scores)


def test_stagec1_mil_determinism_exact_match(tmp_path: Path) -> None:
    view = load_stageb_export_split_v1(_build_export(tmp_path))
    a = compute_stagec1_mil_baseline_scores(view)
    b = compute_stagec1_mil_baseline_scores(view)

    a_rows = [(r.video_id, r.track_id, r.row_index, r.status, r.score) for r in a.track_scores]
    b_rows = [(r.video_id, r.track_id, r.row_index, r.status, r.score) for r in b.track_scores]
    assert a_rows == b_rows
    assert a.per_video_summary == b.per_video_summary
    assert a.run_summary == b.run_summary


def test_stagec1_mil_invalid_config_rejected() -> None:
    empty_view = SimpleNamespace(
        split="train",
        embedding_dim=4,
        iter_videos=lambda: iter(()),
        iter_tracks=lambda _video_id: iter(()),
    )
    with pytest.raises(StageC1AttributionError, match="contains unknown statuses"):
        compute_stagec1_mil_baseline_scores(
            empty_view,
            config=StageC1MilConfig(supported_video_statuses=("processed_with_tracks", "bad_status")),
        )


def test_stagec1_mil_rejects_unsupported_processed_status(tmp_path: Path) -> None:
    view = load_stageb_export_split_v1(_build_export(tmp_path))
    config = StageC1MilConfig(supported_video_statuses=("processed_with_tracks",))

    with pytest.raises(StageC1AttributionError, match="unsupported processed status 'processed_zero_tracks'"):
        compute_stagec1_mil_baseline_scores(view, config=config)


def test_stagec1_mil_missing_boundary_field_fails_clear() -> None:
    bad_track = SimpleNamespace(
        metadata=SimpleNamespace(video_id="vid_a", track_id=1, row_index=0, num_active_frames=3),
        embedding=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )
    bad_view = SimpleNamespace(
        split="train",
        embedding_dim=4,
        iter_videos=lambda: iter([SimpleNamespace(video_id="vid_a", status="processed_with_tracks")]),
        iter_tracks=lambda _video_id: iter([bad_track]),
    )

    with pytest.raises(StageC1AttributionError, match="track.metadata.objectness_score"):
        compute_stagec1_mil_baseline_scores(bad_view)


def test_stagec1_mil_smoke_stagec0_to_artifacts(tmp_path: Path) -> None:
    split_root = _build_export(tmp_path)
    out_dir = tmp_path / "stagec1_out"
    report = run_stagec1_mil_baseline_offline(split_root=split_root, output_dir=out_dir)

    run_summary_path = Path(report["artifacts"]["run_summary"])
    track_scores_path = Path(report["artifacts"]["track_scores"])
    assert run_summary_path.exists()
    assert track_scores_path.exists()

    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    assert run_summary["num_tracks_scored"] == 3
    assert run_summary["split"] == "train"
    assert np.isfinite(run_summary["score_mean"])

    rows = [json.loads(line) for line in track_scores_path.read_text(encoding="utf-8").strip().splitlines()]
    assert [(r["video_id"], r["track_id"]) for r in rows] == [
        ("vid_a", 1),
        ("vid_a", 2),
        ("vid_c", 9),
    ]
