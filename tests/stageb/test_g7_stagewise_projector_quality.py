from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from videocutler.ext_stageb_ovvis.audit.projector_quality_audit import run_projector_quality_audit
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_fixture(root: Path) -> None:
    carrier_dir = root / "carrier_bank" / "lvvis_train_base"
    frame_dir = root / "frame_bank" / "lvvis_train_base"
    text_dir = root / "text_bank"
    audit_dir = root / "audit"
    for path in (carrier_dir, frame_dir / "payload", text_dir / "payload", audit_dir):
        path.mkdir(parents=True, exist_ok=True)

    protos = np.zeros((5, 512), dtype=np.float32)
    for idx in range(5):
        protos[idx, idx] = 1.0
    np.savez(text_dir / "payload" / "text_prototypes.npz", protos=protos)
    _write_jsonl(
        text_dir / "text_prototype_records.jsonl",
        [
            {"raw_id": 1, "proto_path": "payload/text_prototypes.npz#protos[0]", "path_base_mode": "artifact_parent_dir"},
            {"raw_id": 3, "proto_path": "payload/text_prototypes.npz#protos[1]", "path_base_mode": "artifact_parent_dir"},
            {"raw_id": 5, "proto_path": "payload/text_prototypes.npz#protos[2]", "path_base_mode": "artifact_parent_dir"},
            {"raw_id": 7, "proto_path": "payload/text_prototypes.npz#protos[3]", "path_base_mode": "artifact_parent_dir"},
            {"raw_id": 9, "proto_path": "payload/text_prototypes.npz#protos[4]", "path_base_mode": "artifact_parent_dir"},
        ],
    )

    carrier = np.zeros((2, 768), dtype=np.float16)
    carrier[0, 2] = 1.0
    carrier[1, 1] = 1.0
    np.savez(carrier_dir / "carrier_vectors_traj_a.npz", z_norm=carrier[:1])
    np.savez(carrier_dir / "carrier_vectors_traj_b.npz", z_norm=carrier[1:2])
    _write_jsonl(
        carrier_dir / "carrier_records.jsonl",
        [
            {
                "trajectory_id": "traj-a",
                "clip_id": "10",
                "z_norm_path": "carrier_vectors_traj_a.npz#z_norm[0]",
                "frame_indices": [0],
                "frame_carriers_norm_paths": [],
                "path_base_mode": "artifact_parent_dir",
            },
            {
                "trajectory_id": "traj-b",
                "clip_id": "11",
                "z_norm_path": "carrier_vectors_traj_b.npz#z_norm[0]",
                "frame_indices": [0],
                "frame_carriers_norm_paths": [],
                "path_base_mode": "artifact_parent_dir",
            },
        ],
    )
    frame_a = np.zeros((1, 4, 768), dtype=np.float16)
    frame_a[0, 0, 4] = 1.0
    frame_b = np.zeros((1, 4, 768), dtype=np.float16)
    frame_b[0, 0, 3] = 1.0
    np.savez(frame_dir / "payload" / "clip_a_feats.npz", slot_0=frame_a[0])
    np.savez(frame_dir / "payload" / "clip_b_feats.npz", slot_0=frame_b[0])
    _write_jsonl(
        frame_dir / "frame_records.jsonl",
        [
            {"clip_id": "10", "frame_index": 0, "feat_path": "payload/clip_a_feats.npz#0", "path_base_mode": "artifact_parent_dir"},
            {"clip_id": "11", "frame_index": 0, "feat_path": "payload/clip_b_feats.npz#0", "path_base_mode": "artifact_parent_dir"},
        ],
    )
    _write_jsonl(
        frame_dir / "frame_geom_records.jsonl",
        [
            {
                "clip_id": "10",
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
            },
            {
                "clip_id": "11",
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
                "valid_token_mask_path": "frame_geom_records.jsonl#1",
                "path_base_mode": "artifact_parent_dir",
            },
        ],
    )
    _write_json(
        root / "weak_labels" / "weak_labels_train.json",
        [
            {"clip_id": "10", "video_id": 10, "observed_raw_ids": [1], "observation_protocol_id": "p1", "completeness_status": "unknown"},
            {"clip_id": "11", "video_id": 11, "observed_raw_ids": [1], "observation_protocol_id": "p1", "completeness_status": "unknown"},
        ],
    )
    _write_jsonl(
        root / "exports" / "lvvis_train_base" / "trajectory_records.jsonl",
        [
            {
                "dataset_name": "lvvis_train_base",
                "split_tag": "train",
                "clip_id": 10,
                "video_id": 10,
                "rank_in_clip": 0,
                "trajectory_id": "traj-a",
                "generator_tag": "videocutler_r50_native",
                "pred_score": 0.9,
                "frame_indices": [0],
                "masks_rle": [{}],
                "boxes_xyxy": [[0, 0, 10, 10]],
                "valid_carrier": True,
                "invalid_reason": None,
                "image_size": [28, 28],
            },
            {
                "dataset_name": "lvvis_train_base",
                "split_tag": "train",
                "clip_id": 11,
                "video_id": 11,
                "rank_in_clip": 0,
                "trajectory_id": "traj-b",
                "generator_tag": "videocutler_r50_native",
                "pred_score": 0.9,
                "frame_indices": [0],
                "masks_rle": [{}],
                "boxes_xyxy": [[0, 0, 10, 10]],
                "valid_carrier": True,
                "invalid_reason": None,
                "image_size": [28, 28],
            },
        ],
    )
    _write_jsonl(
        audit_dir / "trajectory_gt_match_train_mainline.jsonl",
        [
            {"dataset_name": "lvvis_train_base", "trajectory_source_branch": "mainline", "trajectory_id": "traj-a", "clip_id": "10", "gt_class_id": 7, "audit_usable": True},
            {"dataset_name": "lvvis_train_base", "trajectory_source_branch": "mainline", "trajectory_id": "traj-b", "clip_id": "11", "gt_class_id": 3, "audit_usable": True},
        ],
    )

    for stage_name in ("prealign", "softem_base", "softem_aug"):
        stage_dir = root / "train" / stage_name / "checkpoints"
        stage_dir.mkdir(parents=True, exist_ok=True)
        projector = Projector(ProjectorConfig())
        torch.save(
            {
                "stage_id": stage_name,
                "epoch": 1,
                "projector_state_dict": projector.state_dict(),
                "projector_config": {
                    "input_dim": 768,
                    "hidden_dim": 512,
                    "output_dim": 512,
                    "dropout": 0.0,
                    "use_layernorm": True,
                },
                "seed": 0,
            },
            stage_dir / f"{stage_name if stage_name != 'softem_base' else 'softem_base'}_last.pth",
        )
    (root / "train" / "prealign").mkdir(parents=True, exist_ok=True)
    (root / "train" / "softem_base").mkdir(parents=True, exist_ok=True)
    (root / "train" / "softem_aug").mkdir(parents=True, exist_ok=True)
    _write_json(root / "train" / "prealign" / "train_state.json", {"stage_id": "prealign", "selected_for_infer": "prealign_only", "checkpoint_selected": "train/prealign/checkpoints/prealign_last.pth"})
    _write_json(root / "train" / "softem_base" / "train_state.json", {"stage_id": "softem_base", "selected_for_infer": "base_only", "checkpoint_selected": "train/softem_base/checkpoints/softem_base_last.pth"})
    _write_json(root / "train" / "softem_aug" / "train_state.json", {"stage_id": "softem_aug", "selected_for_infer": "augmented", "checkpoint_selected": "train/softem_aug/checkpoints/softem_aug_last.pth"})


def test_stagewise_projector_quality_audit_emits_artifacts_and_transition_summary(tmp_path: Path) -> None:
    _prepare_fixture(tmp_path)
    payload = run_projector_quality_audit(
        output_root=tmp_path,
        dataset_name="lvvis_train_base",
        trajectory_source_branch="mainline",
        smoke=True,
        smoke_max_trajectories=8,
        topk=5,
        gt_sidecar_dir="audit",
        temperature=0.07,
    )
    assert payload["status"] == "PASS"
    assert (tmp_path / "train" / "audit" / "prealign_projector_quality.json").is_file()
    assert (tmp_path / "train" / "audit" / "softem_base_projector_quality.json").is_file()
    assert (tmp_path / "train" / "audit" / "softem_aug_projector_quality.json").is_file()
    assert (tmp_path / "train" / "audit" / "stagewise_projector_quality_transition_summary.json").is_file()
    assert (tmp_path / "codex" / "outputs" / "G7_training" / "g7_stagewise_projector_quality_latest.json").is_file()
    assert payload["stage_summaries"]["prealign"]["gt_row_count"] == 2
    assert payload["transition_summary"]["prealign_to_softem_base"]["compared_trajectory_count"] == 2
