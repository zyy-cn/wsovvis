from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import (
    fuse_carrier_frame_logits,
    load_combined_evidence,
    observed_mass_loss,
)
from videocutler.ext_stageb_ovvis.algorithms.soft_em import _stage_mass_from_logits, _stage_mass_from_logits_iterative
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import (
    Phase1MaterializationConfig,
    materialize_phase1_training_samples,
)
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prepare_fixture(root: Path) -> dict:
    carrier_dir = root / "carrier_bank" / "lvvis_train_base"
    frame_dir = root / "frame_bank" / "lvvis_train_base"
    text_dir = root / "text_bank"
    for path in (carrier_dir, frame_dir / "payload", text_dir / "payload"):
        path.mkdir(parents=True, exist_ok=True)

    traj = np.zeros((1, 768), dtype=np.float16)
    traj[0, 0] = 1.0
    np.savez(carrier_dir / "carrier_vectors_traj.npz", z_norm=traj)
    _write_jsonl(
        carrier_dir / "carrier_records.jsonl",
        [
            {
                "trajectory_id": "traj-1",
                "clip_id": "1",
                "z_norm_path": "carrier_vectors_traj.npz#z_norm[0]",
                "frame_indices": [0],
                "frame_carriers_norm_paths": [],
                "path_base_mode": "artifact_parent_dir",
            }
        ],
    )

    frame_tokens = np.zeros((1, 4, 768), dtype=np.float16)
    frame_tokens[0, 0, 0] = 1.0
    frame_tokens[0, 1, 1] = 2.0
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

    protos = np.zeros((3, 512), dtype=np.float32)
    protos[0, 0] = 1.0
    protos[1, 1] = 1.0
    protos[2, 2] = 1.0
    np.savez(text_dir / "payload" / "text_prototypes.npz", protos=protos)
    _write_jsonl(
        text_dir / "text_prototype_records.jsonl",
        [
            {"raw_id": 1, "proto_path": "payload/text_prototypes.npz#protos[0]", "path_base_mode": "artifact_parent_dir"},
            {"raw_id": 3, "proto_path": "payload/text_prototypes.npz#protos[1]", "path_base_mode": "artifact_parent_dir"},
            {"raw_id": 7, "proto_path": "payload/text_prototypes.npz#protos[2]", "path_base_mode": "artifact_parent_dir"},
        ],
    )
    _write_json(
        root / "weak_labels" / "weak_labels_train.json",
        [{"clip_id": "1", "video_id": 1, "observed_raw_ids": [1], "observation_protocol_id": "p1", "completeness_status": "unknown"}],
    )
    _write_jsonl(
        root / "exports" / "lvvis_train_base" / "trajectory_records.jsonl",
        [
            {
                "dataset_name": "lvvis_train_base",
                "split_tag": "train",
                "clip_id": 1,
                "video_id": 1,
                "rank_in_clip": 0,
                "trajectory_id": "traj-1",
                "generator_tag": "videocutler_r50_native",
                "pred_score": 0.9,
                "frame_indices": [0],
                "masks_rle": [{}],
                "boxes_xyxy": [[0, 0, 10, 10]],
                "valid_carrier": True,
                "invalid_reason": None,
                "image_size": [28, 28],
            }
        ],
    )
    return {
        "frame_row": {"feat_path": "payload/clip_1_feats.npz#0", "path_base_mode": "artifact_parent_dir"},
        "geom_row": {
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
        },
    }


def test_phase1_candidate_extra_is_real(tmp_path: Path) -> None:
    _prepare_fixture(tmp_path)
    result = materialize_phase1_training_samples(
        tmp_path,
        Phase1MaterializationConfig(dataset_name="lvvis_train_base", smoke=True, smoke_max_trajectories=8),
    )
    sample = next(item for item in result["samples"] if item["trajectory_id"] == "traj-1")
    assert sample["sample_valid"] is True
    assert sample["candidate_ids_extra"]
    assert len(sample["candidate_text_prototypes"]) == len(sample["candidate_ids_known"]) + len(sample["candidate_ids_extra"])


def test_phase1_candidate_extra_is_evidence_driven_and_excludes_observed(tmp_path: Path) -> None:
    root = tmp_path
    carrier_dir = root / "carrier_bank" / "lvvis_train_base"
    frame_dir = root / "frame_bank" / "lvvis_train_base"
    text_dir = root / "text_bank"
    for path in (carrier_dir, frame_dir / "payload", text_dir / "payload"):
        path.mkdir(parents=True, exist_ok=True)

    traj = np.zeros((1, 768), dtype=np.float16)
    traj[0, 2] = 1.0
    np.savez(carrier_dir / "carrier_vectors_traj.npz", z_norm=traj)
    _write_jsonl(
        carrier_dir / "carrier_records.jsonl",
        [
            {
                "trajectory_id": "traj-1",
                "clip_id": "1",
                "z_norm_path": "carrier_vectors_traj.npz#z_norm[0]",
                "frame_indices": [0],
                "frame_carriers_norm_paths": [],
                "path_base_mode": "artifact_parent_dir",
            }
        ],
    )

    frame_tokens = np.zeros((1, 4, 768), dtype=np.float16)
    frame_tokens[0, 2, 2] = 1.0
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

    protos = np.zeros((3, 512), dtype=np.float32)
    protos[0, 0] = 1.0
    protos[1, 1] = 1.0
    protos[2, 2] = 1.0
    np.savez(text_dir / "payload" / "text_prototypes.npz", protos=protos)
    _write_jsonl(
        text_dir / "text_prototype_records.jsonl",
        [
            {"raw_id": 1, "proto_path": "payload/text_prototypes.npz#protos[0]", "path_base_mode": "artifact_parent_dir"},
            {"raw_id": 3, "proto_path": "payload/text_prototypes.npz#protos[1]", "path_base_mode": "artifact_parent_dir"},
            {"raw_id": 7, "proto_path": "payload/text_prototypes.npz#protos[2]", "path_base_mode": "artifact_parent_dir"},
        ],
    )
    _write_json(
        root / "weak_labels" / "weak_labels_train.json",
        [{"clip_id": "1", "video_id": 1, "observed_raw_ids": [1], "observation_protocol_id": "p1", "completeness_status": "unknown"}],
    )
    _write_jsonl(
        root / "exports" / "lvvis_train_base" / "trajectory_records.jsonl",
        [
            {
                "dataset_name": "lvvis_train_base",
                "split_tag": "train",
                "clip_id": 1,
                "video_id": 1,
                "rank_in_clip": 0,
                "trajectory_id": "traj-1",
                "generator_tag": "videocutler_r50_native",
                "pred_score": 0.9,
                "frame_indices": [0],
                "masks_rle": [{}],
                "boxes_xyxy": [[0, 0, 10, 10]],
                "valid_carrier": True,
                "invalid_reason": None,
                "image_size": [28, 28],
            }
        ],
    )

    result = materialize_phase1_training_samples(
        root,
        Phase1MaterializationConfig(dataset_name="lvvis_train_base", smoke=True, smoke_max_trajectories=8),
    )
    sample = next(item for item in result["samples"] if item["trajectory_id"] == "traj-1")
    assert sample["sample_valid"] is True
    assert sample["candidate_ids_known"] == [1]
    assert sample["candidate_ids_extra"] == [7]
    assert sample["candidate_ids_extra_provenance"][0]["raw_id"] == 7
    assert sample["candidate_ids_extra_provenance"][0]["admission_reason"] == "topk_non_observed_by_sample_evidence"


def test_prealign_full_vocab_observed_mass_includes_unknown() -> None:
    logits = torch.zeros(3, dtype=torch.float32)
    loss = observed_mass_loss(logits, [1], unknown_logit=torch.zeros((), dtype=torch.float32))
    assert torch.isclose(loss, torch.tensor(np.log(4.0), dtype=torch.float32))


def test_softem_base_and_aug_domains_differ_and_unknown_competes() -> None:
    init_mass = {"unknown": 0.2, "1": 0.5, "3": 0.2, "7": 0.1}
    base_logits = np.zeros(1 + 1, dtype=np.float64)
    aug_logits = np.zeros(1 + 3, dtype=np.float64)
    base_init, base_final, base_bonus = _stage_mass_from_logits(
        stage_id="softem_base",
        candidate_ids_known=[1],
        candidate_ids_extra=[3, 7],
        initial_mass=init_mass,
        stage_logits=base_logits,
    )
    aug_init, aug_final, aug_bonus = _stage_mass_from_logits(
        stage_id="softem_aug",
        candidate_ids_known=[1],
        candidate_ids_extra=[3, 7],
        initial_mass=init_mass,
        stage_logits=aug_logits,
    )
    assert "3" not in base_final and "7" not in base_final
    assert "3" in aug_final and "7" in aug_final
    assert aug_final["1"] > aug_final["3"]
    assert aug_final["unknown"] > 0.0
    assert base_bonus == [1]
    assert aug_bonus == [1]
    assert base_init["unknown"] > 0.0 and aug_init["unknown"] > 0.0

    shifted_init = {"unknown": 0.2, "1": 0.1, "3": 0.6, "7": 0.1}
    _, shifted_final, _ = _stage_mass_from_logits(
        stage_id="softem_aug",
        candidate_ids_known=[1],
        candidate_ids_extra=[3, 7],
        initial_mass=shifted_init,
        stage_logits=aug_logits,
    )
    assert shifted_final != aug_final
    assert shifted_final["3"] != aug_final["3"]


def test_softem_em_subiterations_are_real() -> None:
    init_mass = {"unknown": 0.2, "1": 0.4, "3": 0.3, "7": 0.1}
    logits = np.zeros(1 + 3, dtype=np.float64)
    init1, final1, bonus1, trace1 = _stage_mass_from_logits_iterative(
        stage_id="softem_aug",
        candidate_ids_known=[1],
        candidate_ids_extra=[3, 7],
        initial_mass=init_mass,
        stage_logits=logits,
        em_subiterations=1,
    )
    init2, final2, bonus2, trace2 = _stage_mass_from_logits_iterative(
        stage_id="softem_aug",
        candidate_ids_known=[1],
        candidate_ids_extra=[3, 7],
        initial_mass=init_mass,
        stage_logits=logits,
        em_subiterations=2,
    )
    assert len(trace1) == 1
    assert len(trace2) == 2
    assert final1 != final2
    assert trace2[0]["r_final"] != trace2[1]["r_final"]
    assert bonus1 == bonus2 == [1]


def test_frame_and_carrier_evidence_combine(tmp_path: Path) -> None:
    fixture = _prepare_fixture(tmp_path)
    sample = {
        "carrier_record": {"z_norm_path": "carrier_vectors_traj.npz#z_norm[0]"},
        "frame_feature_rows": [fixture["frame_row"]],
        "frame_geometry_rows": [fixture["geom_row"]],
    }
    carrier_vec, frame_vec, combined_vec = load_combined_evidence(
        sample,
        output_root=tmp_path,
        dataset_name="lvvis_train_base",
        trajectory_source_branch="mainline",
    )
    _, _, fused_logits = fuse_carrier_frame_logits(
        projector=Projector(ProjectorConfig()),
        carrier_vec=carrier_vec,
        frame_vec=frame_vec,
        candidate_matrix=np.asarray([[1.0, 0.0] + [0.0] * 510, [0.0, 1.0] + [0.0] * 510], dtype=np.float32),
        temperature=0.07,
    )
    assert carrier_vec[0] > 0.0
    assert frame_vec[1] > 0.0
    assert combined_vec[0] > 0.0 and combined_vec[1] > 0.0
    assert not np.allclose(combined_vec, carrier_vec)
    assert not np.allclose(combined_vec, frame_vec)
    assert fused_logits.shape == (2,)
