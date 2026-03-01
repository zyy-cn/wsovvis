import json
from pathlib import Path

import numpy as np
import pytest

from wsovvis.track_feature_export import (
    StageCExportLoadError,
    build_track_feature_export_v1,
    load_stageb_export_split_v1,
)


def _base_input() -> dict:
    return {
        "split": "train",
        "embedding_dim": 4,
        "embedding_dtype": "float32",
        "embedding_pooling": "track_pooled",
        "embedding_normalization": "l2",
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
                "video_id": "vid_b",
                "status": "processed_with_tracks",
                "tracks": [
                    {
                        "track_id": 3,
                        "start_frame_idx": 5,
                        "end_frame_idx": 10,
                        "num_active_frames": 6,
                        "objectness_score": 0.8,
                        "embedding": [3.0, 3.0, 3.0, 3.0],
                    },
                    {
                        "track_id": 7,
                        "start_frame_idx": 20,
                        "end_frame_idx": 30,
                        "num_active_frames": 11,
                        "objectness_score": 0.6,
                        "embedding": [7.0, 7.0, 7.0, 7.0],
                    },
                ],
            },
            {"video_id": "vid_a", "status": "processed_zero_tracks", "tracks": []},
            {"video_id": "vid_d", "status": "failed", "tracks": []},
            {"video_id": "vid_c", "status": "unprocessed", "tracks": []},
        ],
    }


def _build_export(tmp_path: Path) -> Path:
    out_root = tmp_path / "export_train"
    build_track_feature_export_v1(_base_input(), out_root)
    return out_root


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_precedence_prefers_manifest_v1_json(tmp_path: Path) -> None:
    out_root = _build_export(tmp_path)
    (out_root / "manifest.json").write_text("{\"bad\": true}", encoding="utf-8")

    view = load_stageb_export_split_v1(out_root)
    assert view.manifest_path.name == "manifest.v1.json"
    assert [v.video_id for v in view.iter_videos()] == ["vid_a", "vid_b", "vid_c", "vid_d"]


def test_manifest_fallback_to_manifest_json(tmp_path: Path) -> None:
    out_root = _build_export(tmp_path)
    manifest_v1 = out_root / "manifest.v1.json"
    manifest_json = out_root / "manifest.json"
    manifest_v1.rename(manifest_json)

    view = load_stageb_export_split_v1(out_root)
    assert view.manifest_path.name == "manifest.json"
    assert view.split == "train"


def test_happy_path_load_iteration_and_direct_access(tmp_path: Path) -> None:
    view = load_stageb_export_split_v1(_build_export(tmp_path))
    assert [v.video_id for v in view.iter_videos()] == ["vid_a", "vid_b", "vid_c", "vid_d"]

    tracks = list(view.iter_tracks("vid_b"))
    assert [tr.metadata.track_id for tr in tracks] == [3, 7]
    assert [tr.metadata.row_index for tr in tracks] == [0, 1]

    meta = view.get_track_metadata("vid_b", 7)
    assert meta.start_frame_idx == 20
    emb = view.get_track_embedding("vid_b", 7)
    assert emb.dtype == np.float32
    assert emb.shape == (4,)
    assert np.allclose(emb, np.array([7.0, 7.0, 7.0, 7.0], dtype=np.float32))
    assert list(view.iter_tracks("vid_c")) == []


def test_missing_shard_hard_fail(tmp_path: Path) -> None:
    out_root = _build_export(tmp_path)
    manifest = _read_json(out_root / "manifest.v1.json")
    vid_b = [video for video in manifest["videos"] if video["video_id"] == "vid_b"][0]
    (out_root / vid_b["track_arrays_path"]).unlink()

    with pytest.raises(StageCExportLoadError, match="track_arrays_path"):
        load_stageb_export_split_v1(out_root)


@pytest.mark.parametrize(
    "rewrite_fn,error_pattern",
    [
        (
            lambda p: np.savez(
                p,
                embeddings=np.ones((2, 4), dtype=np.float64),
                track_row_index=np.array([0, 1], dtype=np.int64),
            ),
            "embeddings.dtype",
        ),
        (
            lambda p: np.savez(
                p,
                embeddings=np.ones((2, 5), dtype=np.float32),
                track_row_index=np.array([0, 1], dtype=np.int64),
            ),
            "embeddings.shape\\[1\\]",
        ),
        (
            lambda p: np.savez(
                p,
                embeddings=np.array([[1.0, 2.0, np.nan, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32),
                track_row_index=np.array([0, 1], dtype=np.int64),
            ),
            "finite",
        ),
    ],
)
def test_bad_embedding_dtype_shape_or_nonfinite_hard_fail(
    tmp_path: Path, rewrite_fn, error_pattern: str
) -> None:
    out_root = _build_export(tmp_path)
    manifest = _read_json(out_root / "manifest.v1.json")
    vid_b = [video for video in manifest["videos"] if video["video_id"] == "vid_b"][0]
    npz_path = out_root / vid_b["track_arrays_path"]
    rewrite_fn(npz_path)

    with pytest.raises(StageCExportLoadError, match=error_pattern):
        load_stageb_export_split_v1(out_root)


def test_duplicate_track_id_hard_fail(tmp_path: Path) -> None:
    out_root = _build_export(tmp_path)
    meta_path = out_root / "videos" / "vid_b" / "track_metadata.v1.json"
    meta = _read_json(meta_path)
    meta["tracks"][1]["track_id"] = meta["tracks"][0]["track_id"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(StageCExportLoadError, match="duplicate track_id"):
        load_stageb_export_split_v1(out_root)


def test_deterministic_iteration_and_indexing_behavior(tmp_path: Path) -> None:
    out_root = _build_export(tmp_path)
    view_a = load_stageb_export_split_v1(out_root)
    view_b = load_stageb_export_split_v1(out_root)

    fingerprint_a = (
        [(v.video_id, v.status, v.num_tracks) for v in view_a.iter_videos()],
        [(tr.metadata.video_id, tr.metadata.track_id, tr.metadata.row_index) for tr in view_a.iter_tracks("vid_b")],
    )
    fingerprint_b = (
        [(v.video_id, v.status, v.num_tracks) for v in view_b.iter_videos()],
        [(tr.metadata.video_id, tr.metadata.track_id, tr.metadata.row_index) for tr in view_b.iter_tracks("vid_b")],
    )
    assert fingerprint_a == fingerprint_b

    row_record = view_a.get_track_by_index("vid_b", 1)
    id_record = view_a.get_track_metadata("vid_b", 7)
    assert row_record.metadata.track_id == id_record.track_id == 7
    assert row_record.metadata.row_index == 1


def test_smoke_stagec_loader_v1_end_to_end_fingerprint(tmp_path: Path) -> None:
    out_root = _build_export(tmp_path)
    view = load_stageb_export_split_v1(out_root)

    keys = [(v.video_id, v.status) for v in view.iter_videos()]
    track_keys = [(tr.metadata.video_id, tr.metadata.track_id) for tr in view.iter_tracks("vid_b")]
    first_emb_sum = float(view.get_track_embedding("vid_b", 3).sum())
    last_emb_sum = float(view.get_track_embedding("vid_b", 7).sum())
    fingerprint = {
        "video_keys": keys,
        "track_keys": track_keys,
        "first_emb_sum": first_emb_sum,
        "last_emb_sum": last_emb_sum,
    }

    view2 = load_stageb_export_split_v1(out_root)
    fingerprint2 = {
        "video_keys": [(v.video_id, v.status) for v in view2.iter_videos()],
        "track_keys": [(tr.metadata.video_id, tr.metadata.track_id) for tr in view2.iter_tracks("vid_b")],
        "first_emb_sum": float(view2.get_track_embedding("vid_b", 3).sum()),
        "last_emb_sum": float(view2.get_track_embedding("vid_b", 7).sum()),
    }
    assert fingerprint == fingerprint2
