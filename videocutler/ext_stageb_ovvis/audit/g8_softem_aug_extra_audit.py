from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from videocutler.ext_stageb_ovvis.audit.extra_recovery_audit import (
    _load_or_generate_gt_sidecar_lookup,
    _resolve_responsibility_records,
    build_extra_recovery_rows,
)
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import (
    Phase1MaterializationConfig,
    materialize_phase1_training_samples,
)
from videocutler.ext_stageb_ovvis.eval.g8_bridge import write_json


@dataclass(frozen=True)
class SoftemAugExtraAuditConfig:
    dataset_name: str
    output_root: Path
    trajectory_source_branch: str = "mainline"
    topk: int = 1
    generate_sidecars_if_missing: bool = False


def _summary_path(output_root: Path, dataset_name: str) -> Path:
    return output_root / "audit" / "softem_aug_extra" / dataset_name / "softem_aug_extra_audit_summary.json"


def run_softem_aug_extra_audit(config: SoftemAugExtraAuditConfig) -> Dict[str, Any]:
    checkpoint_path = config.output_root / "train" / "softem_aug" / "train_state.json"
    summary_path = _summary_path(config.output_root, config.dataset_name)
    if not checkpoint_path.is_file():
        summary = {
            "dataset_name": config.dataset_name,
            "stage": "softem_aug",
            "status": "STAGE_NOT_PRESENT",
            "row_count": 0,
            "gt_available_row_count": 0,
            "extra_selected_count": 0,
            "extra_correct_count": 0,
            "extra_precision": None,
            "gt_recovered_by_extra_count": 0,
            "gt_recovered_by_extra_rate": None,
            "summary_path": str(summary_path),
            "note": "softem_aug not present for this output_root",
        }
        write_json(summary_path, summary)
        return summary

    materialized = materialize_phase1_training_samples(
        config.output_root,
        Phase1MaterializationConfig(
            dataset_name=config.dataset_name,
            trajectory_source_branch=config.trajectory_source_branch,
            smoke=False,
        ),
    )
    samples = [dict(x) for x in materialized.get("samples", []) if bool(x.get("sample_valid", False))]
    clip_ids = sorted({int(sample.get("trajectory_record", {}).get("clip_id")) for sample in samples if sample.get("trajectory_record", {}).get("clip_id") is not None})
    gt_sidecar_lookup = _load_or_generate_gt_sidecar_lookup(
        output_root=config.output_root,
        dataset_name=config.dataset_name,
        clip_ids=clip_ids,
        generate_sidecars=bool(config.generate_sidecars_if_missing),
    )
    if samples and not gt_sidecar_lookup:
        raise RuntimeError(
            f"AUG_EXTRA_GT_SIDECAR_MISSING_OR_EMPTY for dataset={config.dataset_name} output_root={config.output_root}"
        )
    responsibility_records = list(_resolve_responsibility_records(config.output_root).values())
    rows, _summary = build_extra_recovery_rows(
        output_root=config.output_root,
        dataset_name=config.dataset_name,
        trajectory_source_branch=config.trajectory_source_branch,
        stage_id="softem_aug",
        snapshot_id="softem_aug_extra_minimal",
        materialized_samples=samples,
        responsibility_records=responsibility_records,
        gt_sidecar_lookup=gt_sidecar_lookup,
        topk=max(1, int(config.topk)),
    )
    gt_available_rows = [row for row in rows if bool(row.get("gt_available_for_audit"))]
    extra_selected = [row for row in gt_available_rows if row.get("extra_top1_id") is not None]
    extra_correct = [row for row in extra_selected if bool(row.get("extra_top1_is_gt"))]
    gt_missing_rows = [row for row in gt_available_rows if bool(row.get("gt_missing_from_observed"))]
    gt_recovered = [row for row in gt_missing_rows if bool(row.get("gt_recovered_via_extra"))]
    summary = {
        "dataset_name": config.dataset_name,
        "stage": "softem_aug",
        "status": "PASS" if rows else "EMPTY",
        "row_count": int(len(rows)),
        "gt_available_row_count": int(len(gt_available_rows)),
        "extra_selected_count": int(len(extra_selected)),
        "extra_correct_count": int(len(extra_correct)),
        "extra_precision": float(len(extra_correct) / len(extra_selected)) if extra_selected else None,
        "gt_recovered_by_extra_count": int(len(gt_recovered)),
        "gt_recovered_by_extra_rate": float(len(gt_recovered) / len(gt_missing_rows)) if gt_missing_rows else None,
        "summary_path": str(summary_path),
    }
    write_json(summary_path, summary)
    return summary
