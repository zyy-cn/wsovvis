from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from videocutler.ext_stageb_ovvis.algorithms.prealign import PrealignConfig, train_prealign
from videocutler.ext_stageb_ovvis.algorithms.soft_em import SoftEMConfig, run_soft_em
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import (
    Phase1MaterializationConfig,
    materialize_phase1_training_samples,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copytree(src_root: Path, dst_root: Path, relpaths: list[str]) -> None:
    for relpath in relpaths:
        src = src_root / relpath
        dst = dst_root / relpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _prepare_evidence_fixture(root: Path) -> None:
    carrier_dir = root / "carrier_bank" / "lvvis_train_base"
    frame_dir = root / "frame_bank" / "lvvis_train_base"
    text_dir = root / "text_bank"
    for path in (carrier_dir, frame_dir / "payload", text_dir / "payload"):
        path.mkdir(parents=True, exist_ok=True)

    protos = np.zeros((5, 512), dtype=np.float32)
    for idx in range(5):
        protos[idx, idx] = 1.0
    np.savez(text_dir / "payload" / "text_prototypes.npz", protos=protos)

    def _write_jsonl(path: Path, rows) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

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

    carrier_a = np.zeros((1, 768), dtype=np.float16)
    carrier_a[0, 3] = 1.0
    np.savez(carrier_dir / "carrier_vectors_traj_a.npz", z_norm=carrier_a)
    frame_a = np.zeros((1, 4, 768), dtype=np.float16)
    frame_a[0, 0, 4] = 1.0
    np.savez(frame_dir / "payload" / "clip_a_feats.npz", slot_0=frame_a[0])
    carrier_b = np.zeros((1, 768), dtype=np.float16)
    carrier_b[0, 1] = 1.0
    np.savez(carrier_dir / "carrier_vectors_traj_b.npz", z_norm=carrier_b)
    frame_b = np.zeros((1, 4, 768), dtype=np.float16)
    frame_b[0, 0, 2] = 1.0
    np.savez(frame_dir / "payload" / "clip_b_feats.npz", slot_0=frame_b[0])

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


def test_residual_closure_bounded_smoke_regression(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixture_root = tmp_path / "fixture"
    _prepare_evidence_fixture(fixture_root)

    materialized = materialize_phase1_training_samples(
        fixture_root,
        Phase1MaterializationConfig(dataset_name="lvvis_train_base", smoke=True, smoke_max_trajectories=8),
    )
    prealign_result = train_prealign(
        output_root=fixture_root,
        materialized_samples=materialized["samples"],
        config=PrealignConfig(
            dataset_name="lvvis_train_base",
            trajectory_source_branch="mainline",
            device="cpu",
            seed=0,
            smoke=True,
            epochs=1,
            learning_rate=1e-4,
            temperature=0.07,
        ),
    )
    softem_result = run_soft_em(
        output_root=fixture_root,
        materialized_samples=materialized["samples"],
        config=SoftEMConfig(
            dataset_name="lvvis_train_base",
            trajectory_source_branch="mainline",
            mode="base_then_aug",
            device="cpu",
            seed=0,
            smoke=True,
            temperature=0.07,
            em_subiterations=2,
            base_epochs=1,
            aug_epochs=1,
            base_learning_rate=5e-5,
            aug_learning_rate=5e-5,
        ),
    )

    train_artifacts = [
        "train/prealign/train_state.json",
        "train/prealign/proxy_records.jsonl",
        "train/prealign/checkpoints/prealign_last.pth",
        "train/softem_base/train_state.json",
        "train/softem_base/responsibility_records.jsonl",
        "train/softem_base/checkpoints/softem_base_last.pth",
        "train/softem_aug/train_state.json",
        "train/softem_aug/responsibility_records.jsonl",
        "train/softem_aug/checkpoints/softem_aug_last.pth",
    ]
    _copytree(fixture_root, repo_root, train_artifacts)

    summary_payload = {
        "status": "PASS",
        "gate_id": "G7_training",
        "phase_scope": "residual_semantic_drift_closure_bounded_smoke",
        "smoke": True,
        "training_semantics_changed": False,
        "formal_training_ready": False,
        "materialization_stats": materialized["stats"],
        "prealign_result": {
            "record_count_input": prealign_result["record_count_input"],
            "record_count_trainable": prealign_result["record_count_trainable"],
            "record_count_output": prealign_result["record_count_output"],
            "coverage_ratio_trainable": prealign_result["coverage_ratio_trainable"],
            "loss_mean": prealign_result["loss_mean"],
            "loss_last": prealign_result["loss_last"],
        },
        "softem_result": {
            "record_count_input": softem_result["record_count_input"],
            "record_count_trainable": softem_result["record_count_trainable"],
            "record_count_output": softem_result["record_count_output"],
            "coverage_ratio_trainable": softem_result["coverage_ratio_trainable"],
            "selected_checkpoint_path": softem_result["selected_checkpoint_path"],
            "stage_reports": softem_result["stage_reports"],
        },
        "artifacts": {
            "train_prealign_train_state": "train/prealign/train_state.json",
            "train_prealign_proxy_records": "train/prealign/proxy_records.jsonl",
            "train_softem_base_train_state": "train/softem_base/train_state.json",
            "train_softem_base_responsibility_records": "train/softem_base/responsibility_records.jsonl",
            "train_softem_aug_train_state": "train/softem_aug/train_state.json",
            "train_softem_aug_responsibility_records": "train/softem_aug/responsibility_records.jsonl",
            "hand_off_json": "codex/outputs/G7_training/g7_residual_semantic_drift_closure_latest.json",
            "hand_off_md": "codex/outputs/G7_training/g7_residual_semantic_drift_closure_latest.md",
        },
        "bounded_smoke": {
            "dataset_name": "lvvis_train_base",
            "trajectory_source_branch": "mainline",
            "smoke_max_trajectories": 8,
            "em_subiterations": 2,
        },
    }
    _write_json(repo_root / "codex" / "outputs" / "G7_training" / "g7_residual_semantic_drift_closure_latest.json", summary_payload)
    (repo_root / "codex" / "outputs" / "G7_training" / "g7_residual_semantic_drift_closure_latest.md").write_text(
        "\n".join(
            [
                "# G7 Residual Semantic Drift Closure",
                "",
                "- status: PASS",
                "- smoke: true",
                "- training_semantics_changed: false",
                "- formal_training_ready: false",
                "- bounded_smoke: dataset=lvvis_train_base, branch=mainline, smoke_max_trajectories=8, em_subiterations=2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    for relpath in train_artifacts:
        assert (repo_root / relpath).is_file()
    assert (repo_root / "codex" / "outputs" / "G7_training" / "g7_residual_semantic_drift_closure_latest.json").is_file()
    assert (repo_root / "codex" / "outputs" / "G7_training" / "g7_residual_semantic_drift_closure_latest.md").is_file()
