import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from wsovvis.track_feature_export import StageC1AttributionError, build_track_feature_export_v1, run_stagec1_mil_baseline_offline


def _load_stagec1_cli_build_parser():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "run_stagec1_mil_baseline_offline.py"
    spec = importlib.util.spec_from_file_location("stagec1_cli_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_parser


def _build_export(tmp_path: Path, *, embedding_dim: int = 2, videos: list[dict] | None = None) -> Path:
    videos_payload = videos or [
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
                    "embedding": [1.0, 0.0],
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
                    "embedding": [0.0, 1.0],
                }
            ],
        },
    ]
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
        "videos": videos_payload,
    }
    out_root = tmp_path / "export_train"
    build_track_feature_export_v1(payload, out_root)
    return out_root


def _write_proto_manifest(tmp_path: Path) -> Path:
    arrays_path = tmp_path / "proto_arrays.npz"
    np.savez(arrays_path, prototypes=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    manifest = {
        "schema_name": "wsovvis.stagec.label_prototypes.v1",
        "schema_version": "1.0.0",
        "embedding_dim": 2,
        "dtype": "float32",
        "labels": [{"label_id": 10, "row_index": 0}, {"label_id": 20, "row_index": 1}],
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


def _artifact_bytes(output_dir: Path) -> dict[str, bytes]:
    return {
        "track_scores.jsonl": (output_dir / "track_scores.jsonl").read_bytes(),
        "per_video_summary.json": (output_dir / "per_video_summary.json").read_bytes(),
        "run_summary.json": (output_dir / "run_summary.json").read_bytes(),
    }


def test_sinkhorn_backend_cli_parser_and_defaults() -> None:
    parser = _load_stagec1_cli_build_parser()()
    args = parser.parse_args(["--split-root", "/tmp/s", "--output-dir", "/tmp/o", "--scorer-backend", "sinkhorn_v1"])
    assert args.scorer_backend == "sinkhorn_v1"
    assert args.sinkhorn_temperature == pytest.approx(0.10)
    assert args.sinkhorn_iterations == 12
    assert args.sinkhorn_tolerance == pytest.approx(1e-6)
    assert args.sinkhorn_eps == pytest.approx(1e-12)
    assert args.sinkhorn_c43_enable is False
    assert args.sinkhorn_c43_enable_bg is False
    assert args.sinkhorn_c43_enable_unk_fg is False
    assert args.sinkhorn_c43_bg_prior_weight == pytest.approx(0.0)
    assert args.sinkhorn_c43_unk_fg_prior_weight == pytest.approx(0.0)
    assert args.sinkhorn_c43_unk_fg_min_top_obs_score is None
    assert args.sinkhorn_c43_unk_fg_max_top_obs_score is None
    assert args.sinkhorn_c43_bg_score == pytest.approx(0.0)


def test_sinkhorn_backend_nominal_with_decoder_coverage(tmp_path: Path) -> None:
    split_root = _build_export(tmp_path)
    manifest_path = _write_proto_manifest(tmp_path)
    labelset_path = _write_labelset(
        tmp_path,
        {
            "videos": [
                {"video_id": "vid_a", "label_set_observed_ids": [10, 20]},
                {"video_id": "vid_b", "label_set_observed_ids": [10, 20]},
            ]
        },
    )
    out_dir = tmp_path / "out_sinkhorn"
    report = run_stagec1_mil_baseline_offline(
        split_root=split_root,
        output_dir=out_dir,
        scorer_backend="sinkhorn_v1",
        decoder_backend="coverage_greedy_v1",
        labelset_json=labelset_path,
        prototype_manifest_json=manifest_path,
    )

    summary = report["run_summary"]
    assert summary["scorer_backend"] == "sinkhorn_v1"
    assert summary["decoder_backend"] == "coverage_greedy_v1"
    assert summary["sinkhorn_summary"]["policy_version"] == "sinkhorn_v1.r1"
    assert summary["sinkhorn_summary"]["videos_total"] == 2

    rows = [json.loads(line) for line in (out_dir / "track_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert {row["video_id"] for row in rows} == {"vid_a", "vid_b"}
    assert all(row["predicted_label_id"] in {10, 20} for row in rows)
    assert all(0.0 <= float(row["score"]) <= 1.0 for row in rows)

    per_video = json.loads((out_dir / "per_video_summary.json").read_text(encoding="utf-8"))
    assert all("sinkhorn_converged" in row for row in per_video)
    assert all("sinkhorn_row_mass_l1_error" in row for row in per_video)


def test_sinkhorn_backend_invalid_config_rejected(tmp_path: Path) -> None:
    split_root = _build_export(tmp_path)
    manifest_path = _write_proto_manifest(tmp_path)
    labelset_path = _write_labelset(
        tmp_path,
        {
            "videos": [
                {"video_id": "vid_a", "label_set_observed_ids": [10, 20]},
                {"video_id": "vid_b", "label_set_observed_ids": [10, 20]},
            ]
        },
    )

    with pytest.raises(StageC1AttributionError, match="sinkhorn.temperature"):
        run_stagec1_mil_baseline_offline(
            split_root=split_root,
            output_dir=tmp_path / "out_bad",
            scorer_backend="sinkhorn_v1",
            labelset_json=labelset_path,
            prototype_manifest_json=manifest_path,
            sinkhorn_temperature=0.0,
        )


def test_sinkhorn_c42_parity_hard_gate_snapshot(tmp_path: Path) -> None:
    split_root = _build_export(tmp_path)
    manifest_path = _write_proto_manifest(tmp_path)
    labelset_path = _write_labelset(
        tmp_path,
        {
            "videos": [
                {"video_id": "vid_a", "label_set_observed_ids": [10, 20]},
                {"video_id": "vid_b", "label_set_observed_ids": [10, 20]},
            ]
        },
    )
    out_dir = tmp_path / "out_parity"
    run_stagec1_mil_baseline_offline(
        split_root=split_root,
        output_dir=out_dir,
        scorer_backend="sinkhorn_v1",
        labelset_json=labelset_path,
        prototype_manifest_json=manifest_path,
    )

    assert _artifact_bytes(out_dir)["track_scores.jsonl"] == (
        b'{"decoder_assigned_bg": false, "decoder_assignment_source": "independent", "decoder_margin": 0.0, '
        b'"decoder_predicted_label_id": 10, "decoder_score": 0.5, "predicted_label_id": 10, "row_index": 0, '
        b'"score": 0.5, "status": "processed_with_tracks", "track_id": 1, "video_id": "vid_a"}\n'
        b'{"decoder_assigned_bg": false, "decoder_assignment_source": "independent", "decoder_margin": 0.0, '
        b'"decoder_predicted_label_id": 10, "decoder_score": 0.5, "predicted_label_id": 10, "row_index": 0, '
        b'"score": 0.5, "status": "processed_with_tracks", "track_id": 2, "video_id": "vid_b"}\n'
    )
    assert b'"policy_version": "sinkhorn_v1.r1"' in _artifact_bytes(out_dir)["run_summary.json"]
    assert b'"sinkhorn_c43_enabled"' not in _artifact_bytes(out_dir)["per_video_summary.json"]


def test_sinkhorn_c43_bg_path_schema_and_source_tagging(tmp_path: Path) -> None:
    split_root = _build_export(
        tmp_path,
        videos=[
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
                        "embedding": [-1.0, -1.0],
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
                        "embedding": [0.0, 1.0],
                    }
                ],
            },
        ],
    )
    manifest_path = _write_proto_manifest(tmp_path)
    labelset_path = _write_labelset(
        tmp_path,
        {
            "videos": [
                {"video_id": "vid_a", "label_set_observed_ids": [10, 20]},
                {"video_id": "vid_b", "label_set_observed_ids": [10, 20]},
            ]
        },
    )
    out_dir = tmp_path / "out_c43_bg"
    report = run_stagec1_mil_baseline_offline(
        split_root=split_root,
        output_dir=out_dir,
        scorer_backend="sinkhorn_v1",
        labelset_json=labelset_path,
        prototype_manifest_json=manifest_path,
        sinkhorn_c43_enable=True,
        sinkhorn_c43_enable_bg=True,
        sinkhorn_c43_bg_prior_weight=3.0,
    )
    summary = report["run_summary"]["sinkhorn_summary"]
    assert summary["policy_version"] == "sinkhorn_v1.r2"
    assert summary["c43_enabled"] is True
    assert summary["c43_enable_bg"] is True
    assert summary["c43_num_tracks_bg"] >= 1

    rows = [json.loads(line) for line in (out_dir / "track_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["predicted_label_source"] for row in rows} <= {"observed", "bg"}
    assert all(row["sinkhorn_active_special_columns"] == ["__bg__"] for row in rows)
    assert any(row["predicted_label_id"] == "__bg__" for row in rows)
    assert all("sinkhorn_bg_posterior" in row for row in rows)
    assert all("sinkhorn_top_observed_score" in row for row in rows)

    per_video = json.loads((out_dir / "per_video_summary.json").read_text(encoding="utf-8"))
    assert all(row["sinkhorn_c43_enabled"] is True for row in per_video)
    assert all("sinkhorn_c43_mass_bg_total" in row for row in per_video)


def test_sinkhorn_backend_deterministic_double_run(tmp_path: Path) -> None:
    split_root = _build_export(tmp_path)
    manifest_path = _write_proto_manifest(tmp_path)
    labelset_path = _write_labelset(
        tmp_path,
        {
            "videos": [
                {"video_id": "vid_a", "label_set_observed_ids": [10, 20]},
                {"video_id": "vid_b", "label_set_observed_ids": [10, 20]},
            ]
        },
    )

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    common = {
        "split_root": split_root,
        "scorer_backend": "sinkhorn_v1",
        "labelset_json": labelset_path,
        "prototype_manifest_json": manifest_path,
        "sinkhorn_iterations": 9,
        "sinkhorn_temperature": 0.09,
        "sinkhorn_tolerance": 1e-8,
    }
    run_stagec1_mil_baseline_offline(output_dir=out_a, **common)
    run_stagec1_mil_baseline_offline(output_dir=out_b, **common)

    for name in ("track_scores.jsonl", "per_video_summary.json", "run_summary.json"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()

    out_c43_a = tmp_path / "out_c43_a"
    out_c43_b = tmp_path / "out_c43_b"
    run_stagec1_mil_baseline_offline(
        output_dir=out_c43_a,
        sinkhorn_c43_enable=True,
        sinkhorn_c43_enable_bg=True,
        sinkhorn_c43_bg_prior_weight=2.0,
        **common,
    )
    run_stagec1_mil_baseline_offline(
        output_dir=out_c43_b,
        sinkhorn_c43_enable=True,
        sinkhorn_c43_enable_bg=True,
        sinkhorn_c43_bg_prior_weight=2.0,
        **common,
    )
    for name in ("track_scores.jsonl", "per_video_summary.json", "run_summary.json"):
        assert (out_c43_a / name).read_bytes() == (out_c43_b / name).read_bytes()


def test_sinkhorn_c43_invalid_flag_combinations_fail_fast(tmp_path: Path) -> None:
    split_root = _build_export(tmp_path)
    manifest_path = _write_proto_manifest(tmp_path)
    labelset_path = _write_labelset(
        tmp_path,
        {
            "videos": [
                {"video_id": "vid_a", "label_set_observed_ids": [10, 20]},
                {"video_id": "vid_b", "label_set_observed_ids": [10, 20]},
            ]
        },
    )
    with pytest.raises(StageC1AttributionError, match="sinkhorn.c43.enable_bg"):
        run_stagec1_mil_baseline_offline(
            split_root=split_root,
            output_dir=tmp_path / "out_bad_c43_1",
            scorer_backend="sinkhorn_v1",
            labelset_json=labelset_path,
            prototype_manifest_json=manifest_path,
            sinkhorn_c43_enable=False,
            sinkhorn_c43_enable_bg=True,
        )
    with pytest.raises(StageC1AttributionError, match="sinkhorn.c43.bg_prior_weight"):
        run_stagec1_mil_baseline_offline(
            split_root=split_root,
            output_dir=tmp_path / "out_bad_c43_2",
            scorer_backend="sinkhorn_v1",
            labelset_json=labelset_path,
            prototype_manifest_json=manifest_path,
            sinkhorn_c43_enable=True,
            sinkhorn_c43_enable_bg=True,
            sinkhorn_c43_bg_prior_weight=0.0,
        )
    with pytest.raises(StageC1AttributionError, match="sinkhorn.c43.enable_unk_fg"):
        run_stagec1_mil_baseline_offline(
            split_root=split_root,
            output_dir=tmp_path / "out_bad_c43_3",
            scorer_backend="sinkhorn_v1",
            labelset_json=labelset_path,
            prototype_manifest_json=manifest_path,
            sinkhorn_c43_enable=True,
            sinkhorn_c43_enable_bg=True,
            sinkhorn_c43_bg_prior_weight=1.0,
            sinkhorn_c43_enable_unk_fg=True,
        )
