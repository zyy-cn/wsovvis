from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from videocutler.ext_stageb_ovvis.algorithms.prealign import PrealignConfig, train_prealign
from videocutler.ext_stageb_ovvis.algorithms.soft_em import SoftEMConfig, run_soft_em
from videocutler.ext_stageb_ovvis.audit.attribution_ledger import AttributionLedgerBuffer
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_fixture(root: Path) -> list[dict]:
    carrier_dir = root / "carrier_bank" / "lvvis_train_base"
    carrier_dir.mkdir(parents=True, exist_ok=True)
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
                "frame_indices": [1],
                "frame_carriers_norm_paths": [],
                "path_base_mode": "artifact_parent_dir",
            }
        ],
    )

    text_dir = root / "text_bank"
    (text_dir / "payload").mkdir(parents=True, exist_ok=True)
    protos = np.zeros((2, 512), dtype=np.float32)
    protos[0, 0] = 1.0
    protos[1, 1] = 1.0
    np.savez(text_dir / "payload" / "text_prototypes.npz", protos=protos)
    _write_jsonl(
        text_dir / "text_prototype_records.jsonl",
        [
            {"raw_id": 3, "proto_path": "payload/text_prototypes.npz#protos[0]", "path_base_mode": "artifact_parent_dir"},
            {"raw_id": 7, "proto_path": "payload/text_prototypes.npz#protos[1]", "path_base_mode": "artifact_parent_dir"},
        ],
    )
    frame_dir = root / "frame_bank" / "lvvis_train_base"
    (frame_dir / "payload").mkdir(parents=True, exist_ok=True)
    frame_tokens = np.zeros((1, 4, 768), dtype=np.float16)
    frame_tokens[0, 0, 0] = 1.0
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

    prealign_dir = root / "train" / "prealign"
    (prealign_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    projector = Projector(ProjectorConfig())
    torch.save(
        {
            "stage_id": "prealign",
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
        prealign_dir / "checkpoints" / "prealign_last.pth",
    )
    _write_jsonl(
        prealign_dir / "proxy_records.jsonl",
        [
            {
                "dataset_name": "lvvis_train_base",
                "clip_id": 1,
                "video_id": 1,
                "trajectory_id": "traj-1",
                "observed_raw_ids": [3],
                "proxy_mass": {"unknown": 0.2, "3": 0.8},
                "join_key": "traj-1",
            }
        ],
    )

    sample = {
        "trajectory_id": "traj-1",
        "clip_id": "1",
        "trajectory_record": {"video_id": 1},
        "carrier_record": {"z_norm_path": "carrier_vectors_traj.npz#z_norm[0]"},
        "weak_label_record": {"observed_raw_ids": [3]},
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
        "candidate_ids_known": [3],
        "candidate_ids_extra": [7],
        "missing_views": [],
        "invalid_reasons": [],
        "sample_valid": True,
    }
    return [sample]


def test_audit_ledger_schema_and_write_path(tmp_path: Path) -> None:
    samples = _prepare_fixture(tmp_path)
    audit = AttributionLedgerBuffer(
        output_root=tmp_path,
        dataset_name="lvvis_train_base",
        trajectory_source_branch="mainline",
        topk=5,
        gt_sidecar_dir="audit",
    )
    audit(
        {
            "stage_id": "prealign",
            "snapshot_id": "stage_start",
            "phase": "stage_start",
            "materialized_samples": samples,
            "projector": Projector(ProjectorConfig()),
            "temperature": 0.07,
        }
    )
    audit(
        {
            "stage_id": "prealign",
            "snapshot_id": "stage_end",
            "phase": "stage_end",
            "materialized_samples": samples,
            "projector": Projector(ProjectorConfig()),
            "temperature": 0.07,
        }
    )
    summary = audit.finalize()
    ledger_path = tmp_path / "train" / "prealign" / "attribution_ledger.jsonl"
    assert ledger_path.is_file()
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    row = rows[0]
    assert row["dataset_name"] == "lvvis_train_base"
    assert row["trajectory_id"] == "traj-1"
    assert "topk_ids" in row
    assert "gt_rank" in row
    assert summary["transition_matrix"]["wrong_to_wrong"] == 0
    assert (tmp_path / "train" / "audit" / "attribution_summary.json").is_file()


def test_audit_gracefully_handles_absent_gt_sidecar(tmp_path: Path) -> None:
    samples = _prepare_fixture(tmp_path)
    audit = AttributionLedgerBuffer(
        output_root=tmp_path,
        dataset_name="lvvis_train_base",
        trajectory_source_branch="mainline",
        topk=5,
        gt_sidecar_dir="audit",
    )
    rows = audit.record_snapshot(
        {
            "stage_id": "prealign",
            "snapshot_id": "stage_end",
            "phase": "stage_end",
            "materialized_samples": samples,
            "projector": Projector(ProjectorConfig()),
            "temperature": 0.07,
        }
    )
    assert rows[0]["gt_available_for_audit"] is False
    assert rows[0]["gt_class_id"] is None


def test_summary_transition_matrix_and_monotonic_metrics() -> None:
    stage_rows = {
        "prealign": [
            {"trajectory_id": "t1", "stage_id": "prealign", "snapshot_id": "stage_start", "gt_available_for_audit": True, "gt_rank": 3, "gt_score": 0.1, "is_gt_top1": False, "gt_in_union_domain": True},
            {"trajectory_id": "t1", "stage_id": "prealign", "snapshot_id": "stage_end", "gt_available_for_audit": True, "gt_rank": 1, "gt_score": 0.6, "is_gt_top1": True, "gt_in_union_domain": True},
        ],
        "softem_base": [
            {"trajectory_id": "t1", "stage_id": "softem_base", "snapshot_id": "stage_end", "gt_available_for_audit": True, "gt_rank": 1, "gt_score": 0.7, "is_gt_top1": True, "gt_in_union_domain": True},
        ],
    }
    from videocutler.ext_stageb_ovvis.audit.trajectory_gt_audit import summarize_attribution_rows

    summary = summarize_attribution_rows(stage_rows)
    assert summary["transition_matrix"]["wrong_to_right"] == 1
    assert summary["transition_matrix"]["right_to_right"] == 1
    assert summary["monotonic_gt_rank_improve_rate"] == 1.0
    assert summary["monotonic_gt_score_improve_rate"] == 1.0


def test_audit_hook_does_not_change_training_semantics(tmp_path: Path) -> None:
    control_root = tmp_path / "control"
    audit_root = tmp_path / "audit"
    control_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)
    samples = _prepare_fixture(control_root)
    _prepare_fixture(audit_root)

    control_prealign = train_prealign(
        output_root=control_root,
        materialized_samples=samples,
        config=PrealignConfig(dataset_name="lvvis_train_base", smoke=True, epochs=1, seed=0, device="cpu"),
    )
    control_softem = run_soft_em(
        output_root=control_root,
        materialized_samples=samples,
        config=SoftEMConfig(dataset_name="lvvis_train_base", smoke=True, mode="base_then_aug", seed=0, device="cpu", base_epochs=1, aug_epochs=1),
    )

    audit = AttributionLedgerBuffer(output_root=audit_root, dataset_name="lvvis_train_base", trajectory_source_branch="mainline", topk=5)
    audit_prealign = train_prealign(
        output_root=audit_root,
        materialized_samples=samples,
        config=PrealignConfig(dataset_name="lvvis_train_base", smoke=True, epochs=1, seed=0, device="cpu"),
        audit_callback=audit,
    )
    audit_softem = run_soft_em(
        output_root=audit_root,
        materialized_samples=samples,
        config=SoftEMConfig(dataset_name="lvvis_train_base", smoke=True, mode="base_then_aug", seed=0, device="cpu", base_epochs=1, aug_epochs=1),
        audit_callback=audit,
    )

    assert control_prealign["record_count_output"] == audit_prealign["record_count_output"]
    assert control_softem["record_count_output"] == audit_softem["record_count_output"]
    assert (control_root / "train" / "prealign" / "proxy_records.jsonl").read_text(encoding="utf-8") == (
        audit_root / "train" / "prealign" / "proxy_records.jsonl"
    ).read_text(encoding="utf-8")
    assert (control_root / "train" / "softem_base" / "responsibility_records.jsonl").read_text(encoding="utf-8") == (
        audit_root / "train" / "softem_base" / "responsibility_records.jsonl"
    ).read_text(encoding="utf-8")
