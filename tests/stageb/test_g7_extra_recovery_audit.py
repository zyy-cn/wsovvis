from __future__ import annotations

import json
from pathlib import Path

import pytest

from videocutler.ext_stageb_ovvis.audit.extra_recovery_audit import (
    _clip_summary_from_rows,
    _load_or_generate_gt_sidecar_lookup,
    _sidecar_match_rows,
    build_extra_recovery_rows,
    run_extra_recovery_audit,
)


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_sidecar_fixture(root: Path) -> tuple[dict, dict]:
    exports = root / "exports" / "lvvis_train_base"
    exports_gt = root / "exports_gt" / "lvvis_train_base"
    exports.mkdir(parents=True, exist_ok=True)
    exports_gt.mkdir(parents=True, exist_ok=True)
    main_record = {
        "dataset_name": "lvvis_train_base",
        "split_tag": "train",
        "clip_id": 1,
        "video_id": 11,
        "rank_in_clip": 0,
        "trajectory_id": "main-traj",
        "generator_tag": "main",
        "pred_score": 0.9,
        "frame_indices": [0],
        "masks_rle": [{"size": [2, 2], "counts": "A"}],
        "boxes_xyxy": [[0, 0, 1, 1]],
        "valid_carrier": True,
        "invalid_reason": None,
        "image_size": [2, 2],
        "pred_label_raw": 3,
    }
    gt_record = dict(main_record, trajectory_id="gt-traj", generator_tag="gt", pred_label_raw=7)
    _write_jsonl(exports / "trajectory_records.jsonl", [main_record])
    _write_jsonl(exports_gt / "trajectory_records.jsonl", [gt_record])
    return main_record, gt_record


def test_gt_sidecar_generation_and_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    main_record, gt_record = _prepare_sidecar_fixture(tmp_path)
    monkeypatch.setattr(
        "videocutler.ext_stageb_ovvis.audit.extra_recovery_audit._mean_iou_over_overlapping_frames",
        lambda *_args, **_kwargs: (0.8, 0.8, 1),
    )

    main_sidecars = _sidecar_match_rows(
        output_root=tmp_path,
        dataset_name="lvvis_train_base",
        clip_ids=[1],
        trajectory_source_branch="mainline",
    )
    gt_sidecars = _sidecar_match_rows(
        output_root=tmp_path,
        dataset_name="lvvis_train_base",
        clip_ids=[1],
        trajectory_source_branch="gt_upper_bound",
    )
    assert main_sidecars["match_path"].is_file()
    assert gt_sidecars["identity_path"].is_file()

    lookup = _load_or_generate_gt_sidecar_lookup(
        output_root=tmp_path,
        dataset_name="lvvis_train_base",
        clip_ids=[1],
        generate_sidecars=False,
    )
    assert lookup["main-traj"]["matched_gt_class_id"] == 7
    assert lookup["gt-traj"]["matched_gt_class_id"] == 7
    assert lookup["main-traj"]["audit_usable"] is True
    assert lookup["gt-traj"]["audit_usable"] is True


def test_extra_recovery_row_schema_and_recovery_logic(tmp_path: Path) -> None:
    rows, summary = build_extra_recovery_rows(
        output_root=tmp_path,
        dataset_name="lvvis_train_base",
        trajectory_source_branch="mainline",
        stage_id="softem_aug",
        snapshot_id="stage_end",
        materialized_samples=[
            {
                "trajectory_id": "traj-1",
                "clip_id": 7,
                "video_id": 22,
                "observed_raw_ids": [1],
                "sample_valid": True,
                "missing_views": [],
                "invalid_reasons": [],
            }
        ],
        responsibility_records=[
            {
                "trajectory_id": "traj-1",
                "candidate_ids_known": [1],
                "candidate_ids_extra": [7],
                "r_final": {"1": 0.2, "7": 0.7, "unknown": 0.1},
            }
        ],
        gt_sidecar_lookup={"traj-1": {"audit_usable": True, "matched_gt_class_id": 7}},
        topk=5,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["gt_available_for_audit"] is True
    assert row["gt_missing_from_observed"] is True
    assert row["gt_in_extra_domain"] is True
    assert row["extra_top1_id"] == 7
    assert row["extra_top1_is_gt"] is True
    assert row["gt_recovered_via_extra"] is True
    assert row["gt_rank_within_extra_union"] == 1
    assert row["unknown_score"] == pytest.approx(0.1)
    assert summary["status"] == "PASS"
    assert summary["row_count"] == 1
    assert summary["extra_gt_recall@K"] == pytest.approx(1.0)
    assert summary["extra_precision@K"] == pytest.approx(1.0)
    assert len(summary["clip_summaries"]) == 1


def test_clip_summary_aggregation_counts() -> None:
    clip_rows = _clip_summary_from_rows(
        [
            {
                "clip_id": 5,
                "gt_available_for_audit": True,
                "gt_class_id": 7,
                "gt_missing_from_observed": True,
                "gt_recovered_via_extra": True,
                "candidate_ids_extra": [7],
            },
            {
                "clip_id": 5,
                "gt_available_for_audit": True,
                "gt_class_id": 9,
                "gt_missing_from_observed": True,
                "gt_recovered_via_extra": False,
                "candidate_ids_extra": [11],
            },
        ]
    )
    assert len(clip_rows) == 1
    clip = clip_rows[0]
    assert clip["clip_missing_class_count"] == 2
    assert clip["clip_recovered_missing_class_count"] == 1
    assert clip["clip_extra_class_count"] == 2
    assert clip["clip_missing_recall"] == pytest.approx(0.5)
    assert clip["clip_extra_precision"] == pytest.approx(0.5)
    assert clip["clip_extra_false_discovery_rate"] == pytest.approx(0.5)


def test_extra_recovery_audit_is_audit_only_and_preserves_train_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train_state = tmp_path / "train" / "softem_aug" / "train_state.json"
    train_state.parent.mkdir(parents=True, exist_ok=True)
    train_state.write_text('{"run_scope":"sentinel","stage_id":"softem_aug"}\n', encoding="utf-8")
    resp_path = tmp_path / "train" / "softem_aug" / "responsibility_records.jsonl"
    _write_jsonl(
        resp_path,
        [
            {
                "trajectory_id": "traj-1",
                "candidate_ids_known": [1],
                "candidate_ids_extra": [7],
                "r_final": {"1": 0.2, "7": 0.7, "unknown": 0.1},
            }
        ],
    )
    sample = {
        "trajectory_id": "traj-1",
        "clip_id": 7,
        "video_id": 22,
        "observed_raw_ids": [1],
        "sample_valid": True,
        "missing_views": [],
        "invalid_reasons": [],
    }
    monkeypatch.setattr(
        "videocutler.ext_stageb_ovvis.audit.extra_recovery_audit.materialize_phase1_training_samples",
        lambda *args, **kwargs: {"samples": [sample], "stats": {"sample_count": 1}},
    )
    monkeypatch.setattr(
        "videocutler.ext_stageb_ovvis.audit.extra_recovery_audit._load_or_generate_gt_sidecar_lookup",
        lambda **kwargs: {"traj-1": {"audit_usable": True, "matched_gt_class_id": 7}},
    )

    result = run_extra_recovery_audit(
        output_root=tmp_path,
        dataset_name="lvvis_train_base",
        trajectory_source_branch="mainline",
        smoke=True,
        smoke_max_trajectories=1,
        topk=5,
        generate_val_sidecars=False,
        gt_sidecar_dir="audit",
    )
    assert result["status"] == "PASS"
    assert result["generate_val_sidecars"] is False
    assert (tmp_path / "train" / "softem_aug" / "extra_recovery_ledger.jsonl").is_file()
    assert (tmp_path / "train" / "audit" / "extra_recovery_summary.json").is_file()
    assert train_state.read_text(encoding="utf-8") == '{"run_scope":"sentinel","stage_id":"softem_aug"}\n'
