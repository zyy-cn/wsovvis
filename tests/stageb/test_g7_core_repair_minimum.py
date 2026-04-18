from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import (
    build_stage_domain_indices,
    fuse_carrier_frame_logits,
    load_combined_evidence,
    observed_mass_loss,
    refine_responsibilities,
)
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


def _prepare_fixture(root: Path) -> None:
    carrier_dir = root / "carrier_bank" / "lvvis_train_base"
    frame_dir = root / "frame_bank" / "lvvis_train_base"
    text_dir = root / "text_bank"
    for path in (carrier_dir, frame_dir / "payload", text_dir / "payload"):
        path.mkdir(parents=True, exist_ok=True)

    traj = np.zeros((1, 768), dtype=np.float16)
    traj[0, 0] = 1.0
    np.savez(carrier_dir / "carrier_vectors_traj.npz", z_norm=traj)
    frame_a = np.zeros((1, 768), dtype=np.float16)
    frame_a[0, 6] = 1.0
    frame_a[0, 7] = 0.5
    np.savez(carrier_dir / "carrier_vectors_frame_a.npz", z_norm=frame_a)
    _write_jsonl(
        carrier_dir / "carrier_records.jsonl",
        [{"trajectory_id": "traj-1", "clip_id": "1", "z_norm_path": "carrier_vectors_traj.npz#z_norm[0]", "frame_indices": [0], "frame_carriers_norm_paths": ["carrier_vectors_frame_a.npz#z_norm[0]"], "path_base_mode": "artifact_parent_dir"}],
    )


    _write_jsonl(
        frame_dir / "frame_records.jsonl",
        [{"clip_id": "1", "frame_index": 0, "feat_path": "payload/clip_1_feats.npz#0", "path_base_mode": "artifact_parent_dir"}],
    )
    _write_jsonl(
        frame_dir / "frame_geom_records.jsonl",
        [{"clip_id": "1", "frame_index": 0, "orig_h": 28, "orig_w": 28, "resized_h": 28, "resized_w": 28, "padded_h": 28, "padded_w": 28, "scale_y": 1.0, "scale_x": 1.0, "pad_left": 0, "pad_top": 0, "pad_right": 0, "pad_bottom": 0, "patch_size": 14, "grid_h": 2, "grid_w": 2, "valid_token_mask_path": "frame_geom_records.jsonl#0", "path_base_mode": "artifact_parent_dir"}],
    )
    np.savez(frame_dir / "payload" / "clip_1_feats.npz", slot_0=np.ones((4, 768), dtype=np.float16))
    pooled_vec = np.zeros((1, 768), dtype=np.float16)
    pooled_vec[0, 1] = 1.0
    np.savez(frame_dir / "payload" / "clip_1_pooled.npz", frame_pooled=pooled_vec)
    _write_jsonl(
        frame_dir / "pooled_frame_records.jsonl",
        [{"trajectory_id": "traj-1", "clip_id": "1", "trajectory_source_branch": "mainline", "frame_count": 1, "frame_pooled_path": "payload/clip_1_pooled.npz#frame_pooled[0]", "path_base_mode": "artifact_parent_dir"}],
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

    _write_jsonl(
        root / "exports" / "lvvis_train_base" / "trajectory_records.jsonl",
        [{"trajectory_id": "traj-1", "video_id": 1, "clip_id": 1, "frame_count": 1, "trajectory_source_branch": "mainline"}],
    )
    _write_json(
        root / "weak_labels" / "weak_labels_train.json",
        [{"trajectory_id": "traj-1", "video_id": 1, "clip_id": 1, "observed_raw_ids": [1], "observed_contiguous_ids": [0], "observed_class_names": ["cls-1"], "completeness_status": "unknown", "label_source_type": "simulated_from_gt", "observation_protocol_id": "keep60_seed42"}],
    )


def test_phase1_materialization_uses_runtime_frame_bank_and_marks_placeholder(tmp_path: Path) -> None:
    _prepare_fixture(tmp_path)
    result = materialize_phase1_training_samples(
        tmp_path,
        Phase1MaterializationConfig(dataset_name="lvvis_train_base", trajectory_source_branch="mainline", smoke=True, smoke_max_trajectories=8),
    )
    sample = next(item for item in result["samples"] if item["trajectory_id"] == "traj-1")
    assert sample["sample_valid"] is True
    assert sample["candidate_ids_known"] == [1]
    assert sample["candidate_ids_extra"] == []
    assert sample["candidate_ids_extra_provenance"] == []
    assert sample["candidate_proposal_source"] == "phase1_extra_superseded_runtime_only"


def test_stage_domains_follow_current_contract() -> None:
    base_domain, base_known, base_extra = build_stage_domain_indices([1], [3, 7], stage_id="softem_base")
    aug_domain, aug_known, aug_extra = build_stage_domain_indices([1], [3, 7], stage_id="softem_aug")
    assert base_domain == [1]
    assert base_known == [1]
    assert base_extra == []
    assert aug_domain == [1, 3, 7]
    assert aug_known == [1]
    assert aug_extra == [3, 7]


def test_refine_responsibilities_trace_schema_and_chaining() -> None:
    init_mass = {"unknown": 0.2, "1": 0.5, "3": 0.2, "7": 0.1}
    r_init, r_final, trace = refine_responsibilities(
        initial_mass=init_mass,
        model_logits=[0.5, 0.3, 0.2],
        candidate_ids_known=[1],
        candidate_ids_extra=[3, 7],
        stage_id="softem_aug",
        coverage_bonus=0.1,
        coverage_epsilon=1.0,
        extra_penalty=0.1,
        b_u_value=0.0,
    )
    assert trace["init_mass"] == r_init
    assert trace["final_mass"] == r_final
    r_init2, r_final2, trace2 = refine_responsibilities(
        initial_mass=r_final,
        model_logits=[0.5, 0.3, 0.2],
        candidate_ids_known=[1],
        candidate_ids_extra=[3, 7],
        stage_id="softem_aug",
        coverage_bonus=0.1,
        coverage_epsilon=1.0,
        extra_penalty=0.1,
        b_u_value=0.0,
        coverage_context={k: v for k, v in r_final.items() if k != "unknown"},
    )
    assert r_init2 == r_final
    assert trace2["init_mass"] == r_init2
    assert r_final2 != r_final


def test_observed_mass_loss_includes_unknown() -> None:
    logits = torch.zeros(3, dtype=torch.float32)
    loss = observed_mass_loss(logits, [1], unknown_logit=torch.zeros((), dtype=torch.float32))
    assert torch.isclose(loss, torch.tensor(np.log(4.0), dtype=torch.float32))


def test_frame_and_carrier_evidence_combine_with_runtime_frame_logits_average(tmp_path: Path) -> None:
    _prepare_fixture(tmp_path)
    sample = {
        "clip_id": "1",
        "trajectory_record": {"clip_id": 1, "frame_indices": [0]},
        "carrier_record": {"clip_id": "1", "z_norm_path": "carrier_vectors_traj.npz#z_norm[0]", "frame_indices": [0], "frame_carriers_norm_paths": ["carrier_vectors_frame_a.npz#z_norm[0]"]},
    }
    carrier_vec, frame_vectors, frame_vec, combined_vec = load_combined_evidence(
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
    assert len(frame_vectors) == 1
    assert carrier_vec[0] > 0.0
    assert frame_vec[6] > 0.0 and frame_vec[7] > 0.0
    assert combined_vec[0] > 0.0 and combined_vec[6] > 0.0
    assert fused_logits.shape == (2,)


def test_runtime_frame_evidence_comes_from_carrier_locators_not_frame_average(tmp_path: Path) -> None:
    _prepare_fixture(tmp_path)
    sample_a = {
        "clip_id": "1",
        "trajectory_record": {"clip_id": 1, "frame_indices": [0]},
        "carrier_record": {
            "clip_id": "1",
            "z_norm_path": "carrier_vectors_traj.npz#z_norm[0]",
            "frame_indices": [0],
            "frame_carriers_norm_paths": ["carrier_vectors_frame_a.npz#z_norm[0]"],
        },
    }
    sample_b = {
        "clip_id": "1",
        "trajectory_record": {"clip_id": 1, "frame_indices": [0]},
        "carrier_record": {
            "clip_id": "1",
            "z_norm_path": "carrier_vectors_traj.npz#z_norm[0]",
            "frame_indices": [0],
            "frame_carriers_norm_paths": ["carrier_vectors_frame_b.npz#z_norm[0]"],
        },
    }
    frame_b = np.zeros((1, 768), dtype=np.float16)
    frame_b[0, 8] = 1.0
    frame_b[0, 9] = 0.5
    np.savez(tmp_path / "carrier_bank" / "lvvis_train_base" / "carrier_vectors_frame_b.npz", z_norm=frame_b)
    _, frame_vectors_a, frame_vec_a, _ = load_combined_evidence(
        sample_a,
        output_root=tmp_path,
        dataset_name="lvvis_train_base",
        trajectory_source_branch="mainline",
    )
    _, frame_vectors_b, frame_vec_b, _ = load_combined_evidence(
        sample_b,
        output_root=tmp_path,
        dataset_name="lvvis_train_base",
        trajectory_source_branch="mainline",
    )
    assert len(frame_vectors_a) == 1
    assert len(frame_vectors_b) == 1
    assert not np.allclose(frame_vectors_a[0], frame_vectors_b[0])
    assert not np.allclose(frame_vec_a, frame_vec_b)
