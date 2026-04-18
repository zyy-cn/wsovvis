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
from videocutler.ext_stageb_ovvis.algorithms.soft_em import (
    _build_runtime_extra_cache,
    _compute_clip_refinement_rows,
    _prepare_examples,
)
from videocutler.ext_stageb_ovvis.banks.responsibility_cache import ResponsibilityCache
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import (
    Phase1MaterializationConfig,
    materialize_phase1_training_samples,
)


class _OracleTextProjector(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        arr = torch.as_tensor(
            inputs,
            dtype=torch.float32,
            device=inputs.device if isinstance(inputs, torch.Tensor) else None,
        )
        out = torch.zeros((arr.shape[0], 768), dtype=torch.float32, device=arr.device)
        width = min(arr.shape[1], 768)
        out[:, :width] = arr[:, :width]
        return torch.nn.functional.normalize(out, p=2.0, dim=-1)


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

    traj_a = np.zeros((1, 768), dtype=np.float16)
    traj_a[0, 3] = 1.0
    traj_a[0, 4] = 0.9
    traj_b = np.zeros((1, 768), dtype=np.float16)
    traj_b[0, 2] = 1.0
    traj_b[0, 3] = 0.7
    np.savez(carrier_dir / "carrier_vectors_a.npz", z_norm=traj_a)
    np.savez(carrier_dir / "carrier_vectors_b.npz", z_norm=traj_b)
    frame_a = np.zeros((1, 768), dtype=np.float16)
    frame_a[0, 6] = 1.0
    frame_a[0, 7] = 0.5
    frame_b = np.zeros((1, 768), dtype=np.float16)
    frame_b[0, 8] = 1.0
    frame_b[0, 9] = 0.5
    np.savez(carrier_dir / "carrier_vectors_frame_a.npz", z_norm=frame_a)
    np.savez(carrier_dir / "carrier_vectors_frame_b.npz", z_norm=frame_b)
    _write_jsonl(
        carrier_dir / "carrier_records.jsonl",
        [
            {"trajectory_id": "traj-a", "clip_id": "10", "z_norm_path": "carrier_vectors_a.npz#z_norm[0]", "frame_indices": [0], "frame_carriers_norm_paths": ["carrier_vectors_frame_a.npz#z_norm[0]"], "path_base_mode": "artifact_parent_dir"},
            {"trajectory_id": "traj-b", "clip_id": "11", "z_norm_path": "carrier_vectors_b.npz#z_norm[0]", "frame_indices": [0], "frame_carriers_norm_paths": ["carrier_vectors_frame_b.npz#z_norm[0]"], "path_base_mode": "artifact_parent_dir"},
        ],
    )


    _write_jsonl(
        frame_dir / "frame_records.jsonl",
        [
            {"clip_id": "10", "frame_index": 0, "feat_path": "payload/clip_10_feats.npz#0", "path_base_mode": "artifact_parent_dir"},
            {"clip_id": "11", "frame_index": 0, "feat_path": "payload/clip_11_feats.npz#0", "path_base_mode": "artifact_parent_dir"},
        ],
    )
    _write_jsonl(
        frame_dir / "frame_geom_records.jsonl",
        [
            {"clip_id": "10", "frame_index": 0, "orig_h": 28, "orig_w": 28, "resized_h": 28, "resized_w": 28, "padded_h": 28, "padded_w": 28, "scale_y": 1.0, "scale_x": 1.0, "pad_left": 0, "pad_top": 0, "pad_right": 0, "pad_bottom": 0, "patch_size": 14, "grid_h": 2, "grid_w": 2, "valid_token_mask_path": "frame_geom_records.jsonl#0", "path_base_mode": "artifact_parent_dir"},
            {"clip_id": "11", "frame_index": 0, "orig_h": 28, "orig_w": 28, "resized_h": 28, "resized_w": 28, "padded_h": 28, "padded_w": 28, "scale_y": 1.0, "scale_x": 1.0, "pad_left": 0, "pad_top": 0, "pad_right": 0, "pad_bottom": 0, "patch_size": 14, "grid_h": 2, "grid_w": 2, "valid_token_mask_path": "frame_geom_records.jsonl#1", "path_base_mode": "artifact_parent_dir"},
        ],
    )
    np.savez(frame_dir / "payload" / "clip_10_feats.npz", slot_0=np.ones((4, 768), dtype=np.float16))
    np.savez(frame_dir / "payload" / "clip_11_feats.npz", slot_0=np.ones((4, 768), dtype=np.float16))

    pooled_a = np.zeros((1, 768), dtype=np.float16)
    pooled_a[0, 3] = 0.8
    pooled_a[0, 4] = 1.0
    pooled_b = np.zeros((1, 768), dtype=np.float16)
    pooled_b[0, 2] = 0.8
    pooled_b[0, 1] = 1.0
    np.savez(frame_dir / "payload" / "clip_10_pooled.npz", frame_pooled=pooled_a)
    np.savez(frame_dir / "payload" / "clip_11_pooled.npz", frame_pooled=pooled_b)
    _write_jsonl(
        frame_dir / "pooled_frame_records.jsonl",
        [
            {"trajectory_id": "traj-a", "clip_id": "10", "trajectory_source_branch": "mainline", "frame_count": 1, "frame_pooled_path": "payload/clip_10_pooled.npz#frame_pooled[0]", "path_base_mode": "artifact_parent_dir"},
            {"trajectory_id": "traj-b", "clip_id": "11", "trajectory_source_branch": "mainline", "frame_count": 1, "frame_pooled_path": "payload/clip_11_pooled.npz#frame_pooled[0]", "path_base_mode": "artifact_parent_dir"},
        ],
    )

    _write_jsonl(
        root / "exports" / "lvvis_train_base" / "trajectory_records.jsonl",
        [
            {"trajectory_id": "traj-a", "video_id": 10, "clip_id": 10, "frame_count": 1, "frame_indices": [0], "trajectory_source_branch": "mainline"},
            {"trajectory_id": "traj-b", "video_id": 11, "clip_id": 11, "frame_count": 1, "frame_indices": [0], "trajectory_source_branch": "mainline"},
        ],
    )
    _write_json(
        root / "weak_labels" / "weak_labels_train.json",
        [
            {"trajectory_id": "traj-a", "video_id": 10, "clip_id": 10, "observed_raw_ids": [1], "observed_contiguous_ids": [0], "observed_class_names": ["cls-1"], "completeness_status": "unknown", "label_source_type": "simulated_from_gt", "observation_protocol_id": "keep60_seed42"},
            {"trajectory_id": "traj-b", "video_id": 11, "clip_id": 11, "observed_raw_ids": [1], "observed_contiguous_ids": [0], "observed_class_names": ["cls-1"], "completeness_status": "unknown", "label_source_type": "simulated_from_gt", "observation_protocol_id": "keep60_seed42"},
        ],
    )


def test_runtime_extra_cache_uses_fused_unique_class_topk(tmp_path: Path) -> None:
    _prepare_evidence_fixture(tmp_path)
    result = materialize_phase1_training_samples(
        tmp_path,
        Phase1MaterializationConfig(dataset_name="lvvis_train_base", trajectory_source_branch="mainline", smoke=True, smoke_max_trajectories=8),
    )
    examples = _prepare_examples(
        result["samples"],
        output_root=tmp_path,
        dataset_name="lvvis_train_base",
        trajectory_source_branch="mainline",
    )["examples"]
    theta_t = torch.nn.Parameter(torch.tensor(np.log(np.exp(0.07) - 1.0), dtype=torch.float32))
    cache = _build_runtime_extra_cache(
        examples=examples,
        text_projector=_OracleTextProjector(),
        theta_t=theta_t,
        output_root=tmp_path,
        k_extra=2,
        alpha=0.25,
        lambda_frame=0.25,
        device=torch.device("cpu"),
    )
    assert set(cache[10]["candidate_ids_extra"]) == {7, 9}
    assert len(set(cache[10]["candidate_ids_extra"])) == 2
    assert cache[10]["candidate_ids_extra_provenance"][0]["admission_reason"] == "fused_score_class_level_max_with_observed_neighbor_penalty"
    assert 1 not in cache[10]["candidate_ids_extra"]


def test_refine_responsibilities_returns_trace_contract_and_allows_chaining() -> None:
    init_mass = {"unknown": 0.2, "1": 0.4, "3": 0.3, "7": 0.1}
    model_logits = [0.4, 0.35, 0.25]
    r_init, r_final, trace = refine_responsibilities(
        initial_mass=init_mass,
        model_logits=model_logits,
        candidate_ids_known=[1],
        candidate_ids_extra=[3, 7],
        stage_id="softem_aug",
        coverage_bonus=0.1,
        coverage_epsilon=1.0,
        extra_penalty=0.1,
        b_u_value=0.0,
    )
    assert set(trace.keys()) >= {"domain_ids", "known_ids", "extra_ids", "coverage_bonus_applied_to", "extra_penalty_applied_to", "b_u", "init_mass", "final_mass"}
    assert trace["init_mass"] == r_init
    assert trace["final_mass"] == r_final
    r_init2, r_final2, trace2 = refine_responsibilities(
        initial_mass=r_final,
        model_logits=model_logits,
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


def test_phase1_materialization_marks_extra_as_placeholder(tmp_path: Path) -> None:
    _prepare_evidence_fixture(tmp_path)
    result = materialize_phase1_training_samples(
        tmp_path,
        Phase1MaterializationConfig(dataset_name="lvvis_train_base", trajectory_source_branch="mainline", smoke=True, smoke_max_trajectories=8),
    )
    by_tid = {item["trajectory_id"]: item for item in result["samples"]}
    assert by_tid["traj-a"]["candidate_ids_extra"] == []
    assert by_tid["traj-a"]["candidate_ids_extra_provenance"] == []
    assert by_tid["traj-a"]["candidate_proposal_source"] == "phase1_extra_superseded_runtime_only"


def test_oracle_frame_fusion_formula_uses_temperature_scaling() -> None:
    projector = _OracleTextProjector()
    carrier = np.zeros(768, dtype=np.float32)
    carrier[0] = 1.0
    frame = np.zeros(768, dtype=np.float32)
    frame[1] = 1.0
    candidate_matrix = np.zeros((3, 512), dtype=np.float32)
    candidate_matrix[0, 0] = 1.0
    candidate_matrix[1, 1] = 1.0
    candidate_matrix[2, 2] = 1.0
    carrier_logits, frame_logits, fused_logits = fuse_carrier_frame_logits_torch(
        projector=projector,
        carrier_vec=carrier,
        frame_vec=frame,
        candidate_matrix=candidate_matrix,
        temperature=torch.tensor(0.07, dtype=torch.float32),
    )
    assert carrier_logits.shape == (3,)
    assert frame_logits.shape == (3,)
    assert fused_logits.shape == (3,)
    assert carrier_logits[0] > carrier_logits[1]
    assert frame_logits[1] > frame_logits[0]


def test_observed_mass_loss_keeps_unknown_slot() -> None:
    logits = torch.zeros(3, dtype=torch.float32)
    loss = observed_mass_loss(logits, [1], unknown_logit=torch.zeros((), dtype=torch.float32))
    assert torch.isclose(loss, torch.tensor(np.log(4.0), dtype=torch.float32))

def test_prealign_contract_uses_fused_logits_with_unknown_competitor() -> None:
    projector = _OracleTextProjector()
    carrier = np.zeros(768, dtype=np.float32)
    carrier[0] = 1.0
    frame = np.zeros(768, dtype=np.float32)
    frame[1] = 1.0
    candidate_matrix = np.zeros((2, 512), dtype=np.float32)
    candidate_matrix[0, 0] = 1.0
    candidate_matrix[1, 1] = 1.0
    carrier_logits, frame_logits, fused_logits = fuse_carrier_frame_logits_torch(
        projector=projector,
        carrier_vec=carrier,
        frame_vec=frame,
        candidate_matrix=candidate_matrix,
        temperature=torch.tensor(0.07, dtype=torch.float32),
    )
    assert carrier_logits[0] > carrier_logits[1]
    assert frame_logits[1] > frame_logits[0]
    expected_fused = 0.75 * carrier_logits + 0.25 * frame_logits
    assert torch.allclose(fused_logits, expected_fused)
    loss = observed_mass_loss(fused_logits, [1], unknown_logit=torch.zeros((), dtype=torch.float32))
    expected = -torch.log(torch.exp(fused_logits[1]) / (torch.tensor(1.0, dtype=torch.float32) + torch.exp(fused_logits[0]) + torch.exp(fused_logits[1])))
    assert torch.isclose(loss, expected, atol=1e-6)


def test_runtime_extra_cache_debias_uses_raw_cosine_not_temperature(tmp_path: Path) -> None:
    text_dir = tmp_path / "text_bank"
    (text_dir / "payload").mkdir(parents=True, exist_ok=True)
    protos = np.zeros((3, 512), dtype=np.float32)
    protos[0, 0] = 1.0
    protos[1, 0] = 0.8
    protos[1, 1] = 0.6
    protos[2, 1] = 1.0
    np.savez(text_dir / "payload" / "text_prototypes.npz", protos=protos)
    _write_jsonl(
        text_dir / "text_prototype_records.jsonl",
        [
            {"raw_id": 1, "proto_path": "payload/text_prototypes.npz#protos[0]", "path_base_mode": "artifact_parent_dir"},
            {"raw_id": 3, "proto_path": "payload/text_prototypes.npz#protos[1]", "path_base_mode": "artifact_parent_dir"},
            {"raw_id": 5, "proto_path": "payload/text_prototypes.npz#protos[2]", "path_base_mode": "artifact_parent_dir"},
        ],
    )
    examples = [
        {
            "trajectory_id": "traj-zero",
            "clip_id": 1,
            "observed_raw_ids": [1],
            "carrier_vec": np.zeros(768, dtype=np.float32),
            "frame_vec": np.zeros(768, dtype=np.float32),
        }
    ]
    theta_small = torch.nn.Parameter(torch.tensor(np.log(np.exp(0.07) - 1.0), dtype=torch.float32))
    theta_large = torch.nn.Parameter(torch.tensor(np.log(np.exp(0.70) - 1.0), dtype=torch.float32))
    cache_small = _build_runtime_extra_cache(
        examples=examples,
        text_projector=_OracleTextProjector(),
        theta_t=theta_small,
        output_root=tmp_path,
        k_extra=2,
        alpha=0.25,
        lambda_frame=0.25,
        device=torch.device("cpu"),
    )
    cache_large = _build_runtime_extra_cache(
        examples=examples,
        text_projector=_OracleTextProjector(),
        theta_t=theta_large,
        output_root=tmp_path,
        k_extra=2,
        alpha=0.25,
        lambda_frame=0.25,
        device=torch.device("cpu"),
    )
    score_small = {int(item["raw_id"]): float(item["score"]) for item in cache_small[1]["candidate_ids_extra_provenance"]}
    score_large = {int(item["raw_id"]): float(item["score"]) for item in cache_large[1]["candidate_ids_extra_provenance"]}
    assert score_small[3] == score_large[3]
    assert score_small[5] == score_large[5]
    assert np.isclose(score_small[3], -0.25 * 0.8, atol=1e-6)
    assert np.isclose(score_small[5], 0.0, atol=1e-6)


def test_runtime_extra_cache_marks_runtime_authority_enum(tmp_path: Path) -> None:
    _prepare_evidence_fixture(tmp_path)
    result = materialize_phase1_training_samples(
        tmp_path,
        Phase1MaterializationConfig(dataset_name="lvvis_train_base", trajectory_source_branch="mainline", smoke=True, smoke_max_trajectories=8),
    )
    examples = _prepare_examples(
        result["samples"],
        output_root=tmp_path,
        dataset_name="lvvis_train_base",
        trajectory_source_branch="mainline",
    )["examples"]
    theta_t = torch.nn.Parameter(torch.tensor(np.log(np.exp(0.07) - 1.0), dtype=torch.float32))
    cache = _build_runtime_extra_cache(
        examples=examples,
        text_projector=_OracleTextProjector(),
        theta_t=theta_t,
        output_root=tmp_path,
        k_extra=2,
        alpha=0.25,
        lambda_frame=0.25,
        device=torch.device("cpu"),
    )
    assert cache[10]["candidate_ids_extra_authority"] == "runtime_refresh_cache_only"
    assert cache[10]["candidate_ids_extra_runtime_authoritative"] == cache[10]["candidate_ids_extra"]


def test_runtime_frame_evidence_comes_from_carrier_locators_not_frame_average(tmp_path: Path) -> None:
    _prepare_evidence_fixture(tmp_path)
    sample_a = {
        "clip_id": "10",
        "trajectory_record": {"clip_id": 10, "frame_indices": [0]},
        "carrier_record": {
            "clip_id": "10",
            "z_norm_path": "carrier_vectors_a.npz#z_norm[0]",
            "frame_indices": [0],
            "frame_carriers_norm_paths": ["carrier_vectors_frame_a.npz#z_norm[0]"],
        },
    }
    sample_b = {
        "clip_id": "11",
        "trajectory_record": {"clip_id": 11, "frame_indices": [0]},
        "carrier_record": {
            "clip_id": "11",
            "z_norm_path": "carrier_vectors_b.npz#z_norm[0]",
            "frame_indices": [0],
            "frame_carriers_norm_paths": ["carrier_vectors_frame_b.npz#z_norm[0]"],
        },
    }
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


def test_refine_responsibilities_uses_raw_coverage_mass_not_normalized_share() -> None:
    init_mass = {"unknown": 0.1, "1": 0.45, "3": 0.45}
    model_logits = [0.6, 0.4]
    coverage_context = {"1": 4.0, "3": 1.0}
    _, r_final, _ = refine_responsibilities(
        initial_mass=init_mass,
        model_logits=model_logits,
        candidate_ids_known=[1, 3],
        candidate_ids_extra=[],
        stage_id="softem_base",
        coverage_bonus=0.1,
        coverage_epsilon=1.0,
        extra_penalty=0.1,
        b_u_value=0.0,
        coverage_context=coverage_context,
    )
    scores = np.asarray([
        0.0,
        0.6 + 0.1 * np.log(1.0 + 4.0),
        0.4 + 0.1 * np.log(1.0 + 1.0),
    ], dtype=np.float64)
    expected = np.exp(scores - scores.max())
    expected = expected / expected.sum()
    assert np.isclose(r_final["unknown"], expected[0], atol=1e-8)
    assert np.isclose(r_final["1"], expected[1], atol=1e-8)
    assert np.isclose(r_final["3"], expected[2], atol=1e-8)




def test_refine_responsibilities_ignores_per_entry_prior_mass_when_model_and_coverage_match() -> None:
    common_kwargs = dict(
        model_logits=[0.5, 0.5],
        candidate_ids_known=[1, 3],
        candidate_ids_extra=[],
        stage_id="softem_base",
        coverage_bonus=0.1,
        coverage_epsilon=1.0,
        extra_penalty=0.1,
        b_u_value=0.0,
        coverage_context={"1": 2.0, "3": 2.0},
    )
    _, r_final_a, _ = refine_responsibilities(
        initial_mass={"unknown": 0.1, "1": 0.8, "3": 0.1},
        **common_kwargs,
    )
    _, r_final_b, _ = refine_responsibilities(
        initial_mass={"unknown": 0.1, "1": 0.1, "3": 0.8},
        **common_kwargs,
    )
    assert np.isclose(r_final_a["1"], r_final_b["1"], atol=1e-8)
    assert np.isclose(r_final_a["3"], r_final_b["3"], atol=1e-8)
    assert np.isclose(r_final_a["unknown"], r_final_b["unknown"], atol=1e-8)

def test_softem_aug_initializes_new_extra_from_explicit_logits(tmp_path: Path) -> None:
    candidate_matrix = np.zeros((2, 512), dtype=np.float32)
    candidate_matrix[0, 0] = 1.0
    candidate_matrix[1, 1] = 1.0
    clip_examples = [
        {
            "trajectory_id": "traj-1",
            "clip_id": 1,
            "video_id": 1,
            "candidate_ids_known": [1],
            "candidate_ids_extra": [3],
            "candidate_matrix": candidate_matrix,
            "candidate_records": [{"raw_id": 1}, {"raw_id": 3}],
            "carrier_vec": np.asarray([1.0] + [0.0] * 767, dtype=np.float32),
            "frame_vec": np.asarray([0.0, 1.0] + [0.0] * 766, dtype=np.float32),
            "frame_vectors": None,
        }
    ]
    base_cache = ResponsibilityCache.from_records(
        stage_id="softem_base",
        records=[{"trajectory_id": "traj-1", "r_final": {"unknown": 0.1, "1": 0.9}}],
    )
    theta_t = torch.nn.Parameter(torch.tensor(np.log(np.exp(0.07) - 1.0), dtype=torch.float32))
    b_u = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    rows, _ = _compute_clip_refinement_rows(
        stage_id="softem_aug",
        clip_examples=clip_examples,
        base_cache=base_cache,
        text_projector=_OracleTextProjector(),
        theta_t=theta_t,
        b_u=b_u,
        em_subiterations=1,
        lambda_frame=0.25,
        device=torch.device("cpu"),
    )
    row = rows[0]
    t_dis = torch.nn.functional.softplus(theta_t.detach()) + 1e-4
    _, _, logits_known_extra = fuse_carrier_frame_logits_torch(
        projector=_OracleTextProjector(),
        carrier_vec=clip_examples[0]["carrier_vec"],
        frame_vec=clip_examples[0]["frame_vec"],
        candidate_matrix=clip_examples[0]["candidate_matrix"],
        temperature=t_dis,
        lambda_frame=0.25,
        frame_vectors=None,
    )
    scores = torch.tensor([0.0, float(logits_known_extra[0]), float(logits_known_extra[1] - 0.1)], dtype=torch.float64)
    expected = torch.softmax(scores, dim=0).cpu().numpy()
    assert np.isclose(row["r_init"]["unknown"], float(expected[0]), atol=1e-8)
    assert np.isclose(row["r_init"]["1"], float(expected[1]), atol=1e-8)
    assert np.isclose(row["r_init"]["3"], float(expected[2]), atol=1e-8)
    assert row["r_init"]["3"] > 0.0
