from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import (
    fuse_carrier_frame_logits_torch,
    load_combined_evidence,
    observed_mass_loss,
    refine_responsibilities,
)
from videocutler.ext_stageb_ovvis.algorithms.soft_em import _stage_mass_from_logits_iterative
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import (
    Phase1MaterializationConfig,
    materialize_phase1_training_samples,
)
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig
from videocutler.run_stageb_train_softem import resolve_em_subiterations


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prepare_evidence_fixture(root: Path) -> None:
    carrier_dir = root / "carrier_bank" / "lvvis_train_base"
    frame_dir = root / "frame_bank" / "lvvis_train_base"
    text_dir = root / "text_bank"
    for path in (carrier_dir, frame_dir / "payload", text_dir / "payload"):
        path.mkdir(parents=True, exist_ok=True)

    # Five text prototypes: one observed + four non-observed extras.
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

    # Trajectory A should prefer extras 7 and 9.
    carrier_a = np.zeros((1, 768), dtype=np.float16)
    carrier_a[0, 3] = 1.0
    np.savez(carrier_dir / "carrier_vectors_traj_a.npz", z_norm=carrier_a)
    frame_a = np.zeros((1, 4, 768), dtype=np.float16)
    frame_a[0, 0, 4] = 1.0
    np.savez(frame_dir / "payload" / "clip_a_feats.npz", slot_0=frame_a[0])

    # Trajectory B should prefer extras 3 and 5.
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


def test_oracle_prealign_observed_mass_full_vocab() -> None:
    logits = torch.zeros(3, dtype=torch.float32)
    loss = observed_mass_loss(logits, [1], unknown_logit=torch.zeros((), dtype=torch.float32))
    assert torch.isclose(loss, torch.tensor(np.log(4.0), dtype=torch.float32))


def test_oracle_extra_proposal_evidence_driven(tmp_path: Path) -> None:
    _prepare_evidence_fixture(tmp_path)
    result = materialize_phase1_training_samples(
        tmp_path,
        Phase1MaterializationConfig(dataset_name="lvvis_train_base", smoke=True, smoke_max_trajectories=8),
    )
    by_tid = {item["trajectory_id"]: item for item in result["samples"]}
    assert by_tid["traj-a"]["candidate_ids_extra"] == [7]
    assert by_tid["traj-b"]["candidate_ids_extra"] == [3]
    assert set(by_tid["traj-a"]["candidate_ids_extra"]).isdisjoint(set(by_tid["traj-a"]["observed_raw_ids"]))
    assert by_tid["traj-a"]["candidate_ids_extra_provenance"][0]["admission_reason"] == "topk_non_observed_by_clip_evidence"
    assert by_tid["traj-a"]["candidate_ids_extra_provenance"][0]["raw_id"] == 7
    assert 1 not in by_tid["traj-a"]["candidate_ids_extra"]


def test_oracle_coverage_refinement_mass_driven() -> None:
    init_mass_a = {"unknown": 0.2, "1": 0.6, "3": 0.1, "7": 0.1}
    init_mass_b = {"unknown": 0.2, "1": 0.2, "3": 0.5, "7": 0.1}
    model_probs = [0.4, 0.35, 0.25]
    final_a, init_a, bonus_a = refine_responsibilities(
        initial_mass=init_mass_a,
        model_probs=model_probs,
        candidate_ids_known=[1],
        candidate_ids_extra=[3, 7],
        stage_id="softem_aug",
        coverage_bonus=0.1,
        coverage_epsilon=1.0,
        extra_penalty=0.1,
    )
    final_b, init_b, bonus_b = refine_responsibilities(
        initial_mass=init_mass_b,
        model_probs=model_probs,
        candidate_ids_known=[1],
        candidate_ids_extra=[3, 7],
        stage_id="softem_aug",
        coverage_bonus=0.1,
        coverage_epsilon=1.0,
        extra_penalty=0.1,
    )
    assert bonus_a == [1]
    assert bonus_b == [1]
    assert init_a["1"] != init_b["1"]
    assert final_a["1"] != final_b["1"]
    assert final_a["3"] != final_b["3"]
    assert "3" not in bonus_a
    assert "7" not in bonus_a


def test_oracle_em_subiterations_real() -> None:
    init_mass = {"unknown": 0.2, "1": 0.4, "3": 0.3, "7": 0.1}
    logits = np.array([0.0, 1.25, -0.75, 0.25], dtype=np.float64)
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
    assert trace2[1]["r_init"] != init_mass
    assert trace2[1]["r_init"] != trace2[0]["r_init"]
    assert bonus1 == bonus2 == [1]
    assert init1.keys() == init2.keys()
    assert all(abs(float(init1[key]) - float(init2[key])) < 1e-10 for key in init1)


def test_oracle_unknown_competes() -> None:
    final_mass, init_mass, bonus = refine_responsibilities(
        initial_mass={"unknown": 0.7, "1": 0.2, "3": 0.1},
        model_probs=[0.34],
        candidate_ids_known=[1],
        candidate_ids_extra=[],
        stage_id="softem_base",
        coverage_bonus=0.1,
        coverage_epsilon=1.0,
        extra_penalty=0.1,
    )
    assert bonus == [1]
    assert "unknown" in final_mass and "1" in final_mass
    assert 0.0 < final_mass["unknown"] < 1.0
    assert 0.0 < final_mass["1"] < 1.0
    assert init_mass["unknown"] > 0.0


def test_oracle_frame_fusion_formula_check(tmp_path: Path) -> None:
    _prepare_evidence_fixture(tmp_path)
    projector = Projector(ProjectorConfig())
    carrier = np.zeros(768, dtype=np.float32)
    carrier[0] = 1.0
    frame_a = np.zeros(768, dtype=np.float32)
    frame_a[0] = 1.0
    frame_b = np.zeros(768, dtype=np.float32)
    frame_b[1] = 1.0
    candidate_matrix = np.zeros((3, 512), dtype=np.float32)
    candidate_matrix[0, 0] = 1.0
    candidate_matrix[1, 1] = 1.0
    candidate_matrix[2, 2] = 1.0
    carrier_logits, frame_logits, fused_logits = fuse_carrier_frame_logits_torch(
        projector=projector,
        carrier_vec=carrier,
        frame_vec=(frame_a + frame_b) / 2.0,
        frame_vectors=[frame_a, frame_b],
        candidate_matrix=candidate_matrix,
        temperature=0.07,
    )
    device = next(projector.parameters()).device
    carrier_tensor = torch.from_numpy(carrier).to(device=device, dtype=torch.float32).unsqueeze(0)
    frame_tensor = torch.from_numpy(np.stack([frame_a, frame_b], axis=0)).to(device=device, dtype=torch.float32)
    cand_tensor = torch.from_numpy(candidate_matrix).to(device=device, dtype=torch.float32)
    cand_tensor = torch.nn.functional.normalize(cand_tensor, p=2.0, dim=-1)
    expected_carrier = projector(carrier_tensor)
    expected_frame = projector(frame_tensor)
    expected_carrier_logits = torch.matmul(expected_carrier, cand_tensor.t()).squeeze(0) / 0.07
    expected_frame_logits = torch.matmul(expected_frame, cand_tensor.t()) / 0.07
    expected_frame_logits = expected_frame_logits.mean(dim=0)
    expected_fused_logits = 0.75 * expected_carrier_logits + 0.25 * expected_frame_logits
    assert torch.allclose(carrier_logits, expected_carrier_logits, atol=1e-6)
    assert torch.allclose(frame_logits, expected_frame_logits, atol=1e-6)
    assert torch.allclose(fused_logits, expected_fused_logits, atol=1e-6)
    _, _, fallback_logits = fuse_carrier_frame_logits_torch(
        projector=projector,
        carrier_vec=carrier,
        frame_vec=(frame_a + frame_b) / 2.0,
        candidate_matrix=candidate_matrix,
        temperature=0.07,
    )
    assert not torch.allclose(fused_logits, fallback_logits)


def test_oracle_runner_default_em_subiterations_coherent() -> None:
    assert resolve_em_subiterations(smoke=True, explicit=None) == 2
    assert resolve_em_subiterations(smoke=False, explicit=None) == 3
    assert resolve_em_subiterations(smoke=True, explicit=5) == 5
