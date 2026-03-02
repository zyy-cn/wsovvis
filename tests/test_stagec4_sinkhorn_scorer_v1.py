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


def _build_export(tmp_path: Path, *, embedding_dim: int = 2) -> Path:
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
        ],
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


def test_sinkhorn_backend_cli_parser_and_defaults() -> None:
    parser = _load_stagec1_cli_build_parser()()
    args = parser.parse_args(["--split-root", "/tmp/s", "--output-dir", "/tmp/o", "--scorer-backend", "sinkhorn_v1"])
    assert args.scorer_backend == "sinkhorn_v1"
    assert args.sinkhorn_temperature == pytest.approx(0.10)
    assert args.sinkhorn_iterations == 12
    assert args.sinkhorn_tolerance == pytest.approx(1e-6)
    assert args.sinkhorn_eps == pytest.approx(1e-12)


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
    assert rows[0]["predicted_label_id"] == 10
    assert rows[1]["predicted_label_id"] == 20
    assert 0.0 <= float(rows[0]["score"]) <= 1.0
    assert 0.0 <= float(rows[1]["score"]) <= 1.0

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
