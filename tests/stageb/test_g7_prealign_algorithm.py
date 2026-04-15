from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from videocutler.ext_stageb_ovvis.algorithms.prealign import PrealignConfig, train_prealign


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_fixture(root: Path) -> None:
    carrier_dir = root / "carrier_bank" / "lvvis_train_base"
    carrier_dir.mkdir(parents=True, exist_ok=True)
    z_raw = np.random.RandomState(0).randn(1, 768).astype(np.float16)
    z_norm = z_raw / np.maximum(np.linalg.norm(z_raw, axis=1, keepdims=True), 1e-6)
    np.savez(carrier_dir / "carrier_vectors_traj.npz", z_raw=z_raw.astype(np.float16), z_norm=z_norm.astype(np.float16))
    _write_jsonl(
        carrier_dir / "carrier_records.jsonl",
        [
            {
                "trajectory_id": "traj_1",
                "clip_id": "1",
                "frame_indices": [0],
                "z_raw_path": "carrier_vectors_traj.npz#z_raw[0]",
                "z_norm_path": "carrier_vectors_traj.npz#z_norm[0]",
                "frame_carriers_norm_paths": [],
                "path_base_mode": "artifact_parent_dir",
            }
        ],
    )
    text_dir = root / "text_bank" / "payload"
    text_dir.mkdir(parents=True, exist_ok=True)
    protos = np.random.RandomState(1).randn(2, 512).astype(np.float32)
    protos = protos / np.maximum(np.linalg.norm(protos, axis=1, keepdims=True), 1e-6)
    np.savez(text_dir / "text_prototypes.npz", protos=protos.astype(np.float32))
    _write_jsonl(
        root / "text_bank" / "text_prototype_records.jsonl",
        [
            {"raw_id": 3, "proto_path": "payload/text_prototypes.npz#protos[0]", "path_base_mode": "artifact_parent_dir"},
            {"raw_id": 7, "proto_path": "payload/text_prototypes.npz#protos[1]", "path_base_mode": "artifact_parent_dir"},
        ],
    )
    frame_dir = root / "frame_bank" / "lvvis_train_base"
    (frame_dir / "payload").mkdir(parents=True, exist_ok=True)
    frame_tokens = np.random.RandomState(2).randn(1, 4, 768).astype(np.float16)
    np.savez(frame_dir / "payload" / "clip_1_feats.npz", slot_0=frame_tokens[0])
    _write_jsonl(
        frame_dir / "frame_records.jsonl",
        [{"clip_id": "1", "frame_index": 0, "feat_path": "payload/clip_1_feats.npz#0", "path_base_mode": "artifact_parent_dir"}],
    )
    _write_jsonl(
        frame_dir / "frame_geom_records.jsonl",
        [
            {
                "clip_id": "1",
                "frame_index": 0,
                "orig_h": 28,
                "orig_w": 28,
                "resized_h": 28,
                "resized_w": 28,
                "padded_h": 28,
                "padded_w": 28,
                "scale_y": 1.0,
                "scale_x": 1.0,
                "pad_left": 0,
                "pad_top": 0,
                "pad_right": 0,
                "pad_bottom": 0,
                "patch_size": 14,
                "grid_h": 2,
                "grid_w": 2,
                "valid_token_mask_path": "frame_geom_records.jsonl#0",
                "path_base_mode": "artifact_parent_dir",
            }
        ],
    )


def test_train_prealign_writes_canonical_stage_local_artifacts(tmp_path: Path) -> None:
    _prepare_fixture(tmp_path)
    sample = {
        "trajectory_id": "traj_1",
        "clip_id": "1",
        "trajectory_record": {"video_id": 1},
        "carrier_record": {"z_norm_path": "carrier_vectors_traj.npz#z_norm[0]"},
        "weak_label_record": {"observed_raw_ids": [3], "clip_id": "1", "video_id": 1},
        "frame_feature_rows": [{"feat_path": "payload/clip_1_feats.npz#0", "path_base_mode": "artifact_parent_dir"}],
        "frame_geometry_rows": [
            {
                "clip_id": "1",
                "frame_index": 0,
                "orig_h": 28,
                "orig_w": 28,
                "resized_h": 28,
                "resized_w": 28,
                "padded_h": 28,
                "padded_w": 28,
                "scale_y": 1.0,
                "scale_x": 1.0,
                "pad_left": 0,
                "pad_top": 0,
                "pad_right": 0,
                "pad_bottom": 0,
                "patch_size": 14,
                "grid_h": 2,
                "grid_w": 2,
                "valid_token_mask_path": "frame_geom_records.jsonl#0",
                "path_base_mode": "artifact_parent_dir",
            }
        ],
        "candidate_text_prototypes": [
            {"raw_id": 3, "proto_path": "payload/text_prototypes.npz#protos[0]", "path_base_mode": "artifact_parent_dir"},
            {"raw_id": 7, "proto_path": "payload/text_prototypes.npz#protos[1]", "path_base_mode": "artifact_parent_dir"},
        ],
        "observed_raw_ids": [3],
        "candidate_ids_known": [3, 7],
        "candidate_ids_extra": [],
        "missing_views": [],
        "invalid_reasons": [],
        "sample_valid": True,
    }
    result = train_prealign(
        output_root=tmp_path,
        materialized_samples=[sample],
        config=PrealignConfig(dataset_name="lvvis_train_base", smoke=True, epochs=1, seed=0, device="cpu"),
    )
    assert result["record_count_output"] == 1
    train_state = json.loads((tmp_path / "train" / "prealign" / "train_state.json").read_text(encoding="utf-8"))
    assert train_state["stage_id"] == "prealign"
    assert train_state["selected_for_infer"] == "prealign_only"
    proxy_lines = (tmp_path / "train" / "prealign" / "proxy_records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(proxy_lines) == 1
    proxy = json.loads(proxy_lines[0])
    assert proxy["trajectory_id"] == "traj_1"
    assert "3" in proxy["proxy_mass"]
    assert (tmp_path / "train" / "prealign" / "checkpoints" / "prealign_last.pth").is_file()
