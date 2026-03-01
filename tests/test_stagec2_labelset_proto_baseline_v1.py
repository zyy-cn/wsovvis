import json
from pathlib import Path

import numpy as np
import pytest

from wsovvis.track_feature_export import (
    StageC1AttributionError,
    build_track_feature_export_v1,
    compute_stagec1_labelset_proto_baseline_scores,
    load_stageb_export_split_v1,
    load_stagec_label_prototype_inventory_v1,
    load_stagec_labelset_lookup,
    run_stagec1_mil_baseline_offline,
)


def _build_export(tmp_path: Path, *, embedding_dim: int = 4) -> Path:
    base = {
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
        "videos": [
            {
                "video_id": "vid_a",
                "status": "processed_with_tracks",
                "tracks": [
                    {
                        "track_id": 1,
                        "start_frame_idx": 0,
                        "end_frame_idx": 2,
                        "num_active_frames": 3,
                        "objectness_score": 0.5,
                        "embedding": [1.0, 0.0, 0.0, 0.0] if embedding_dim == 4 else [1.0, 0.0, 0.0],
                    }
                ],
            },
            {
                "video_id": "vid_b",
                "status": "processed_with_tracks",
                "tracks": [
                    {
                        "track_id": 2,
                        "start_frame_idx": 0,
                        "end_frame_idx": 2,
                        "num_active_frames": 3,
                        "objectness_score": 0.5,
                        "embedding": [0.0, 1.0, 0.0, 0.0] if embedding_dim == 4 else [0.0, 1.0, 0.0],
                    }
                ],
            },
        ],
    }
    out_root = tmp_path / "export_train"
    build_track_feature_export_v1(base, out_root)
    return out_root


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


def _write_proto_manifest(
    tmp_path: Path,
    *,
    prototypes: np.ndarray,
    labels: list[dict],
    schema_name: str = "wsovvis.stagec.label_prototypes.v1",
    dtype: str = "float32",
    embedding_dim: int | None = None,
) -> Path:
    arrays_path = tmp_path / "proto_arrays.npz"
    np.savez(arrays_path, prototypes=prototypes)

    manifest = {
        "schema_name": schema_name,
        "schema_version": "1.0.0",
        "embedding_dim": int(embedding_dim if embedding_dim is not None else prototypes.shape[1]),
        "dtype": dtype,
        "labels": labels,
        "arrays_path": arrays_path.name,
        "array_key": "prototypes",
    }
    manifest_path = tmp_path / "prototype_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _write_labelset(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "labelset.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_prototype_manifest_validation_schema_name(tmp_path: Path) -> None:
    manifest_path = _write_proto_manifest(
        tmp_path,
        prototypes=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        labels=[{"label_id": 1, "row_index": 0}, {"label_id": 2, "row_index": 1}],
        schema_name="bad.schema",
    )
    with pytest.raises(StageC1AttributionError, match="schema_name"):
        load_stagec_label_prototype_inventory_v1(manifest_path)


def test_proto_embedding_dim_mismatch_fail_fast(tmp_path: Path) -> None:
    split_root = _build_export(tmp_path, embedding_dim=4)
    view = load_stageb_export_split_v1(split_root)

    manifest_path = _write_proto_manifest(
        tmp_path,
        prototypes=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        labels=[{"label_id": 1, "row_index": 0}],
        embedding_dim=3,
    )
    inventory = load_stagec_label_prototype_inventory_v1(manifest_path)
    labelset = {"vid_a": ((0, 1),), "vid_b": ((0, 1),)}

    with pytest.raises(StageC1AttributionError, match="must match split embedding_dim"):
        compute_stagec1_labelset_proto_baseline_scores(view, prototype_inventory=inventory, labelset_lookup=labelset)


def test_labelset_lookup_and_video_matching_coverage(tmp_path: Path) -> None:
    split_root = _build_export(tmp_path)
    out_dir = tmp_path / "out"

    manifest_path = _write_proto_manifest(
        tmp_path,
        prototypes=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        labels=[{"label_id": 1, "row_index": 0}, {"label_id": 2, "row_index": 1}],
    )
    labelset_path = _write_labelset(
        tmp_path,
        {
            "videos": [
                {"video_id": "vid_a", "label_set_observed_ids": [1, 1, 2]},
            ]
        },
    )

    report = run_stagec1_mil_baseline_offline(
        split_root=split_root,
        output_dir=out_dir,
        scorer_backend="labelset_proto_v1",
        labelset_json=labelset_path,
        prototype_manifest_json=manifest_path,
    )
    run_summary = report["run_summary"]
    assert run_summary["labelset_coverage"]["num_videos_missing_labelset"] == 1
    assert run_summary["labelset_coverage"]["videos_missing_labelset"] == ["vid_b"]


def test_cosine_scoring_and_deterministic_tiebreak_int_vs_str(tmp_path: Path) -> None:
    split_root = _build_export(tmp_path)
    out_dir = tmp_path / "out"

    manifest_path = _write_proto_manifest(
        tmp_path,
        prototypes=np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],  # int label 7
                [1.0, 0.0, 0.0, 0.0],  # str label "7" exact tie
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        labels=[
            {"label_id": 7, "row_index": 0},
            {"label_id": "7", "row_index": 1},
            {"label_id": 9, "row_index": 2},
        ],
    )
    labelset_path = _write_labelset(
        tmp_path,
        {
            "videos": [
                {"video_id": "vid_a", "label_set_observed_ids": ["7", 7]},
                {"video_id": "vid_b", "label_set_observed_ids": [9]},
            ]
        },
    )

    report = run_stagec1_mil_baseline_offline(
        split_root=split_root,
        output_dir=out_dir,
        scorer_backend="labelset_proto_v1",
        labelset_json=labelset_path,
        prototype_manifest_json=manifest_path,
    )

    rows = [json.loads(line) for line in (out_dir / "track_scores.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    assert rows[0]["video_id"] == "vid_a"
    assert rows[0]["predicted_label_id"] == 7
    assert rows[0]["score"] == pytest.approx(1.0)
    assert rows[1]["video_id"] == "vid_b"
    assert rows[1]["predicted_label_id"] == 9
    assert rows[1]["score"] == pytest.approx(1.0)
    assert report["run_summary"]["scorer_backend"] == "labelset_proto_v1"


def test_empty_labelset_policy_error_and_fallback(tmp_path: Path) -> None:
    split_root = _build_export(tmp_path)
    manifest_path = _write_proto_manifest(
        tmp_path,
        prototypes=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        labels=[{"label_id": 101, "row_index": 0}],
    )
    labelset_path = _write_labelset(
        tmp_path,
        {
            "videos": [
                {"video_id": "vid_a", "label_set_observed_ids": [999]},
                {"video_id": "vid_b", "label_set_observed_ids": [999]},
            ]
        },
    )

    with pytest.raises(StageC1AttributionError, match="empty candidate label intersection"):
        run_stagec1_mil_baseline_offline(
            split_root=split_root,
            output_dir=tmp_path / "out_error",
            scorer_backend="labelset_proto_v1",
            labelset_json=labelset_path,
            prototype_manifest_json=manifest_path,
            empty_labelset_policy="error",
        )

    report = run_stagec1_mil_baseline_offline(
        split_root=split_root,
        output_dir=tmp_path / "out_fallback",
        scorer_backend="labelset_proto_v1",
        labelset_json=labelset_path,
        prototype_manifest_json=manifest_path,
        empty_labelset_policy="use_all_prototypes",
    )
    coverage = report["run_summary"]["labelset_coverage"]
    assert coverage["num_videos_using_fallback_label_pool"] == 2
    assert coverage["num_tracks_scored_with_fallback_label_pool"] == 2


def test_proto_backend_deterministic_double_run(tmp_path: Path) -> None:
    split_root = _build_export(tmp_path)
    manifest_path = _write_proto_manifest(
        tmp_path,
        prototypes=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        labels=[{"label_id": 1, "row_index": 0}, {"label_id": 2, "row_index": 1}],
    )
    labelset_path = _write_labelset(
        tmp_path,
        {
            "videos": [
                {"video_id": "vid_a", "label_set_observed_ids": [1]},
                {"video_id": "vid_b", "label_set_observed_ids": [2]},
            ]
        },
    )

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    run_stagec1_mil_baseline_offline(
        split_root=split_root,
        output_dir=out_a,
        scorer_backend="labelset_proto_v1",
        labelset_json=labelset_path,
        prototype_manifest_json=manifest_path,
    )
    run_stagec1_mil_baseline_offline(
        split_root=split_root,
        output_dir=out_b,
        scorer_backend="labelset_proto_v1",
        labelset_json=labelset_path,
        prototype_manifest_json=manifest_path,
    )

    for name in ("track_scores.jsonl", "per_video_summary.json", "run_summary.json"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()


def test_decoder_independent_matches_scorer_predictions(tmp_path: Path) -> None:
    split_root = _build_export(tmp_path)
    manifest_path = _write_proto_manifest(
        tmp_path,
        prototypes=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        labels=[{"label_id": 1, "row_index": 0}, {"label_id": 2, "row_index": 1}],
    )
    labelset_path = _write_labelset(
        tmp_path,
        {
            "videos": [
                {"video_id": "vid_a", "label_set_observed_ids": [1]},
                {"video_id": "vid_b", "label_set_observed_ids": [2]},
            ]
        },
    )
    out_dir = tmp_path / "out_independent"
    run_stagec1_mil_baseline_offline(
        split_root=split_root,
        output_dir=out_dir,
        scorer_backend="labelset_proto_v1",
        decoder_backend="independent",
        labelset_json=labelset_path,
        prototype_manifest_json=manifest_path,
    )
    rows = [json.loads(line) for line in (out_dir / "track_scores.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    assert all(row["predicted_label_id"] == row["decoder_predicted_label_id"] for row in rows)
    assert all(row["decoder_assigned_bg"] is False for row in rows)
    summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["decoder_backend"] == "independent"


def test_decoder_coverage_greedy_tie_break_is_deterministic(tmp_path: Path) -> None:
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
                "embedding": [1.0, 0.0],
            },
        ],
    )
    manifest_path = _write_proto_manifest(
        tmp_path,
        prototypes=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        labels=[{"label_id": 10, "row_index": 0}, {"label_id": 20, "row_index": 1}],
    )
    labelset_path = _write_labelset(tmp_path, {"videos": [{"video_id": "vid_a", "label_set_observed_ids": [10, 20]}]})
    out_dir = tmp_path / "out_tie"

    run_stagec1_mil_baseline_offline(
        split_root=split_root,
        output_dir=out_dir,
        scorer_backend="labelset_proto_v1",
        decoder_backend="coverage_greedy_v1",
        labelset_json=labelset_path,
        prototype_manifest_json=manifest_path,
    )
    rows = [json.loads(line) for line in (out_dir / "track_scores.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    assert rows[0]["track_id"] == 1
    assert rows[0]["decoder_assignment_source"] == "coverage"
    assert rows[0]["decoder_predicted_label_id"] == 10
    assert rows[1]["track_id"] == 2
    assert rows[1]["decoder_assignment_source"] == "coverage"
    assert rows[1]["decoder_predicted_label_id"] == 20


def test_decoder_coverage_changes_assignment_vs_independent(tmp_path: Path) -> None:
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
                "embedding": [0.8, 0.6],
            },
        ],
    )
    manifest_path = _write_proto_manifest(
        tmp_path,
        prototypes=np.asarray([[1.0, 0.0], [0.9, 0.4358899]], dtype=np.float32),
        labels=[{"label_id": 100, "row_index": 0}, {"label_id": 200, "row_index": 1}],
    )
    labelset_path = _write_labelset(tmp_path, {"videos": [{"video_id": "vid_a", "label_set_observed_ids": [100, 200]}]})

    out_ind = tmp_path / "out_ind"
    out_cov = tmp_path / "out_cov"
    run_stagec1_mil_baseline_offline(
        split_root=split_root,
        output_dir=out_ind,
        scorer_backend="labelset_proto_v1",
        decoder_backend="independent",
        labelset_json=labelset_path,
        prototype_manifest_json=manifest_path,
    )
    run_stagec1_mil_baseline_offline(
        split_root=split_root,
        output_dir=out_cov,
        scorer_backend="labelset_proto_v1",
        decoder_backend="coverage_greedy_v1",
        labelset_json=labelset_path,
        prototype_manifest_json=manifest_path,
    )

    rows_ind = [json.loads(line) for line in (out_ind / "track_scores.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    rows_cov = [json.loads(line) for line in (out_cov / "track_scores.jsonl").read_text(encoding="utf-8").strip().splitlines()]
    assert [r["decoder_predicted_label_id"] for r in rows_ind] == [100, 100]
    assert [r["decoder_predicted_label_id"] for r in rows_cov] == [100, 200]


def test_decoder_additive_serialization_fields_present(tmp_path: Path) -> None:
    split_root = _build_export(tmp_path)
    manifest_path = _write_proto_manifest(
        tmp_path,
        prototypes=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        labels=[{"label_id": 1, "row_index": 0}, {"label_id": 2, "row_index": 1}],
    )
    labelset_path = _write_labelset(
        tmp_path,
        {
            "videos": [
                {"video_id": "vid_a", "label_set_observed_ids": [1, 2]},
                {"video_id": "vid_b", "label_set_observed_ids": [1, 2]},
            ]
        },
    )
    out_dir = tmp_path / "out_serialization"
    run_stagec1_mil_baseline_offline(
        split_root=split_root,
        output_dir=out_dir,
        scorer_backend="labelset_proto_v1",
        decoder_backend="coverage_greedy_v1",
        labelset_json=labelset_path,
        prototype_manifest_json=manifest_path,
    )

    row = json.loads((out_dir / "track_scores.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "video_id" in row
    assert "track_id" in row
    assert "score" in row
    assert "predicted_label_id" in row
    assert "decoder_predicted_label_id" in row
    assert "decoder_assignment_source" in row
    assert "decoder_assigned_bg" in row

    per_video = json.loads((out_dir / "per_video_summary.json").read_text(encoding="utf-8"))
    assert "num_tracks" in per_video[0]
    assert "decoder_backend" in per_video[0]
    assert "decoder_coverage_hit_count" in per_video[0]

    run_summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["scorer_backend"] == "labelset_proto_v1"
    assert run_summary["decoder_backend"] == "coverage_greedy_v1"
    assert "decoder_summary" in run_summary


def test_decoder_backend_invalid_or_incompatible_rejected(tmp_path: Path) -> None:
    split_root = _build_export(tmp_path)
    with pytest.raises(StageC1AttributionError, match="decoder.backend"):
        run_stagec1_mil_baseline_offline(
            split_root=split_root,
            output_dir=tmp_path / "bad_backend",
            scorer_backend="mil_v1",
            decoder_backend="bad_backend",
        )
    with pytest.raises(StageC1AttributionError, match="must be 'independent' when scorer_backend='mil_v1'"):
        run_stagec1_mil_baseline_offline(
            split_root=split_root,
            output_dir=tmp_path / "incompatible_backend",
            scorer_backend="mil_v1",
            decoder_backend="coverage_greedy_v1",
        )


def test_stagec1_default_backend_matches_explicit_mil(tmp_path: Path) -> None:
    split_root = _build_export(tmp_path)
    implicit = run_stagec1_mil_baseline_offline(split_root=split_root, output_dir=tmp_path / "implicit")
    explicit = run_stagec1_mil_baseline_offline(
        split_root=split_root,
        output_dir=tmp_path / "explicit",
        scorer_backend="mil_v1",
    )
    assert implicit["run_summary"] == explicit["run_summary"]


def test_load_stagec_labelset_lookup_accepts_mapping_and_key(tmp_path: Path) -> None:
    labelset_path = _write_labelset(
        tmp_path,
        {
            "vid_a": {"label_set_observed_ids": [1, 2]},
            "vid_b": {"label_set_observed_ids": ["x"]},
        },
    )
    loaded = load_stagec_labelset_lookup(labelset_path, labelset_key="label_set_observed_ids")
    assert loaded["vid_a"] == ((0, 1), (0, 2))
    assert loaded["vid_b"] == ((1, "x"),)
