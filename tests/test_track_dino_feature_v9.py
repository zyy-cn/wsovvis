import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

mask_utils = pytest.importorskip("pycocotools.mask")

from wsovvis.features.track_dino_feature_v9 import (
    SemanticCacheConfig,
    TrackCropRequest,
    build_track_dino_feature_cache_v9,
    build_track_dino_feature_cache_v9_worked_example,
    load_track_dino_feature_cache_v9,
    render_track_dino_feature_provenance_svg,
    summarize_track_dino_feature_cache_v9,
)
from wsovvis.track_feature_export import build_track_feature_export_v1
from wsovvis.tracking import build_global_track_bank_v9


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_synthetic_g3_input(tmp_path: Path) -> Path:
    payload = {
        "split": "val",
        "embedding_dim": 2,
        "embedding_dtype": "float32",
        "embedding_pooling": "track_pooled",
        "embedding_normalization": "none",
        "producer": {
            "stage_b_checkpoint_id": "ckpt_001",
            "stage_b_checkpoint_hash": "sha256:a",
            "stage_b_config_ref": "config.json",
            "stage_b_config_hash": "sha256:b",
            "pseudo_tube_manifest_id": "pseudo.json",
            "pseudo_tube_manifest_hash": "sha256:c",
            "split": "val",
            "extraction_settings": {
                "frame_sampling_rule": "uniform_stride_1",
                "pooling_rule": "track_feature_vector_direct",
                "min_track_length": 1,
            },
        },
        "videos": [
            {
                "video_id": "0",
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
                        "objectness_score": 0.8,
                        "embedding": [0.999, 0.001],
                    },
                ],
            },
            {"video_id": "1", "status": "processed_zero_tracks", "tracks": []},
        ],
    }
    source_root = tmp_path / "stageb_split"
    build_track_feature_export_v1(payload, source_root)
    return build_global_track_bank_v9(source_root, tmp_path / "g3_bank")


def _encode_square_mask(height: int, width: int, *, x0: int, y0: int, size: int) -> dict:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0 : y0 + size, x0 : x0 + size] = 1
    encoded = mask_utils.encode(np.asfortranarray(mask))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        encoded["counts"] = counts.decode("ascii")
    return encoded


def _build_run_root(tmp_path: Path) -> Path:
    repo_root = tmp_path
    run_root = repo_root / "runs" / "wsovvis_seqformer" / "19"
    image_root = repo_root / "data" / "LV-VIS" / "val" / "JPEGImages"
    frames = image_root / "00000"
    frames.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(4):
        pixels = np.zeros((12, 12, 3), dtype=np.uint8)
        pixels[..., 0] = 10 + frame_idx
        pixels[..., 1] = 20 + frame_idx
        pixels[..., 2] = 30 + frame_idx
        Image.fromarray(pixels, mode="RGB").save(frames / f"{frame_idx:05d}.jpg")

    _write_json(
        run_root / "config.json",
        {
            "data": {
                "val_img_root": "data/LV-VIS/val/JPEGImages",
                "val_json": "data/LV-VIS/annotations/lvvis_val_agnostic_for_test.json",
            }
        },
    )
    _write_json(
        repo_root / "data" / "LV-VIS" / "annotations" / "lvvis_val_agnostic_for_test.json",
        {
            "videos": [
                {
                    "id": 0,
                    "width": 12,
                    "height": 12,
                    "length": 4,
                    "file_names": [f"00000/{frame_idx:05d}.jpg" for frame_idx in range(4)],
                },
                {"id": 1, "width": 12, "height": 12, "length": 0, "file_names": []},
            ]
        },
    )
    _write_json(
        run_root / "d2" / "inference" / "results.json",
        [
            {
                "video_id": 0,
                "track_id": 11,
                "score": 0.9,
                "start_frame_idx": 0,
                "end_frame_idx": 3,
                "num_active_frames": 4,
                "segmentations": [
                    _encode_square_mask(12, 12, x0=1, y0=1, size=4),
                    _encode_square_mask(12, 12, x0=2, y0=1, size=4),
                    _encode_square_mask(12, 12, x0=3, y0=1, size=4),
                    _encode_square_mask(12, 12, x0=4, y0=1, size=4),
                ],
            },
            {
                "video_id": 0,
                "track_id": 12,
                "score": 0.8,
                "start_frame_idx": 0,
                "end_frame_idx": 3,
                "num_active_frames": 4,
                "segmentations": [
                    _encode_square_mask(12, 12, x0=1, y0=2, size=4),
                    _encode_square_mask(12, 12, x0=2, y0=2, size=4),
                    _encode_square_mask(12, 12, x0=3, y0=2, size=4),
                    _encode_square_mask(12, 12, x0=4, y0=2, size=4),
                ],
            },
        ],
    )
    return run_root


def _fake_feature_extractor(
    requests: list[TrackCropRequest],
    repo_root: Path,
    config: SemanticCacheConfig,
) -> list[np.ndarray]:
    result = []
    for request in requests:
        crop_w = request.crop_box_xyxy[2] - request.crop_box_xyxy[0]
        crop_h = request.crop_box_xyxy[3] - request.crop_box_xyxy[1]
        result.append(
            np.asarray(
                [
                    float(request.frame_idx + 1),
                    float(request.mask_area),
                    float(crop_w),
                    float(crop_h),
                ],
                dtype=np.float32,
            )
        )
    return result


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_track_dino_feature_v9_build_summary_worked_example_and_svg(tmp_path: Path) -> None:
    g3_root = _build_synthetic_g3_input(tmp_path)
    run_root = _build_run_root(tmp_path)
    semantic_root = build_track_dino_feature_cache_v9(
        global_track_bank_root=g3_root,
        run_root=run_root,
        output_split_root=tmp_path / "g4_cache",
        overwrite=True,
        config=SemanticCacheConfig(resize_edge=28, max_visible_frames_per_track=2),
        frame_feature_extractor=_fake_feature_extractor,
    )

    view = load_track_dino_feature_cache_v9(semantic_root)
    videos = [(row.video_id, row.status, row.num_global_tracks) for row in view.iter_videos()]
    assert videos == [("0", "processed_with_tracks", 1), ("1", "processed_zero_tracks", 0)]

    track_rows = list(view.iter_tracks("0"))
    assert len(track_rows) == 1
    assert track_rows[0].metadata.member_track_ids == (11, 12)
    assert track_rows[0].metadata.representative_source_track_id == 11
    assert len(track_rows[0].metadata.selected_frame_indices) == 2
    assert track_rows[0].z_tau.shape == (4,)
    assert np.isclose(np.linalg.norm(track_rows[0].z_tau), 1.0, atol=1e-6)

    summary = summarize_track_dino_feature_cache_v9(semantic_root)
    assert summary["semantic_cache_coverage"]["tracks_with_z_tau"] == 1
    assert summary["objectness_distribution"]["count"] == 1
    assert summary["selected_video_id"] == "0"

    worked_example = build_track_dino_feature_cache_v9_worked_example(semantic_root)
    assert worked_example["selected_video_id"] == "0"
    assert worked_example["global_track"]["member_track_ids"] == [11, 12]
    assert worked_example["semantic_carrier"]["representative_source_track_id"] == 11

    svg_path = render_track_dino_feature_provenance_svg(semantic_root, tmp_path / "provenance.svg")
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "G4 crop/pooling provenance" in svg_text
    assert "frame 0" in svg_text


def test_track_dino_feature_v9_double_build_is_deterministic(tmp_path: Path) -> None:
    g3_root = _build_synthetic_g3_input(tmp_path)
    run_root = _build_run_root(tmp_path)
    root_a = build_track_dino_feature_cache_v9(
        global_track_bank_root=g3_root,
        run_root=run_root,
        output_split_root=tmp_path / "g4_cache_a",
        overwrite=True,
        config=SemanticCacheConfig(resize_edge=28, max_visible_frames_per_track=2),
        frame_feature_extractor=_fake_feature_extractor,
    )
    root_b = build_track_dino_feature_cache_v9(
        global_track_bank_root=g3_root,
        run_root=run_root,
        output_split_root=tmp_path / "g4_cache_b",
        overwrite=True,
        config=SemanticCacheConfig(resize_edge=28, max_visible_frames_per_track=2),
        frame_feature_extractor=_fake_feature_extractor,
    )
    assert _hash_tree(root_a) == _hash_tree(root_b)
