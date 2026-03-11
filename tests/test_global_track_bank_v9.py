import hashlib
import json
from pathlib import Path

import numpy as np

from wsovvis.track_feature_export import build_track_feature_export_v1
from wsovvis.tracking.global_track_bank_v9 import (
    build_global_track_bank_v9,
    build_global_track_bank_v9_worked_example,
    load_global_track_bank_v9,
    render_global_track_bank_coverage_svg,
    summarize_global_track_bank_v9,
)


def _build_input_payload() -> dict:
    return {
        "split": "train",
        "embedding_dim": 2,
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
                "pooling_rule": "track_feature_vector_direct",
                "min_track_length": 1,
            },
        },
        "videos": [
            {
                "video_id": "vid_a",
                "status": "processed_with_tracks",
                "tracks": [
                    {
                        "track_id": 11,
                        "start_frame_idx": 0,
                        "end_frame_idx": 3,
                        "num_active_frames": 4,
                        "objectness_score": 0.9,
                        "embedding": [1.0, 0.0],
                    },
                    {
                        "track_id": 12,
                        "start_frame_idx": 0,
                        "end_frame_idx": 3,
                        "num_active_frames": 4,
                        "objectness_score": 0.7,
                        "embedding": [0.999, 0.001],
                    },
                    {
                        "track_id": 31,
                        "start_frame_idx": 4,
                        "end_frame_idx": 6,
                        "num_active_frames": 3,
                        "objectness_score": 0.6,
                        "embedding": [0.0, 1.0],
                    },
                ],
            },
            {"video_id": "vid_b", "status": "processed_zero_tracks", "tracks": []},
        ],
    }


def _build_source_split(tmp_path: Path) -> Path:
    out_root = tmp_path / "stageb_split"
    build_track_feature_export_v1(_build_input_payload(), out_root)
    return out_root


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_global_track_bank_v9_merges_strict_duplicate_intervals(tmp_path: Path) -> None:
    source_split = _build_source_split(tmp_path)
    bank_root = build_global_track_bank_v9(source_split, tmp_path / "g3_bank")

    view = load_global_track_bank_v9(bank_root)
    videos = [(video.video_id, video.status, video.num_local_tracklets, video.num_global_tracks) for video in view.iter_videos()]
    assert videos == [
        ("vid_a", "processed_with_tracks", 3, 2),
        ("vid_b", "processed_zero_tracks", 0, 0),
    ]

    vid_a_tracks = list(view.iter_global_tracks("vid_a"))
    assert [record.metadata.member_track_ids for record in vid_a_tracks] == [(11, 12), (31,)]
    assert vid_a_tracks[0].metadata.start_frame_idx == 0
    assert vid_a_tracks[0].metadata.end_frame_idx == 3
    assert vid_a_tracks[0].embedding.shape == (2,)

    trace = json.loads((bank_root / "videos" / "vid_a" / "stitching_trace.v1.json").read_text(encoding="utf-8"))
    assert len(trace["candidate_edges"]) == 1
    assert trace["candidate_edges"][0]["left_row_index"] == 0
    assert trace["candidate_edges"][0]["right_row_index"] == 1


def test_global_track_bank_v9_summary_worked_example_and_svg(tmp_path: Path) -> None:
    source_split = _build_source_split(tmp_path)
    bank_root = build_global_track_bank_v9(source_split, tmp_path / "g3_bank")

    summary = summarize_global_track_bank_v9(bank_root)
    assert summary["stitching_stats"]["videos_with_merges"] == 1
    assert summary["stitching_stats"]["fragmentation_reduction_total"] == 1
    assert summary["selected_video_id"] == "vid_a"

    worked_example = build_global_track_bank_v9_worked_example(bank_root)
    assert worked_example["selected_video_id"] == "vid_a"
    assert worked_example["best_edge"]["match_score"] > 0.997
    assert worked_example["merge_result"][0]["member_track_ids"] == [11, 12]

    svg_path = render_global_track_bank_coverage_svg(bank_root, tmp_path / "coverage.svg")
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "G3 coverage for video vid_a" in svg_text
    assert "11,12" in svg_text


def test_global_track_bank_v9_determinism_double_build(tmp_path: Path) -> None:
    source_split = _build_source_split(tmp_path)
    root_a = build_global_track_bank_v9(source_split, tmp_path / "g3_bank_a")
    root_b = build_global_track_bank_v9(source_split, tmp_path / "g3_bank_b")

    assert _hash_tree(root_a) == _hash_tree(root_b)
