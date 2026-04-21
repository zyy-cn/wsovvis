from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from videocutler.ext_stageb_ovvis.audit.extra_recovery_audit import (
    _dataset_split,
    _load_jsonl,
    _sidecar_match_rows,
)
from videocutler.ext_stageb_ovvis.banks.carrier_bank import read_carrier_records

Record = Dict[str, Any]


@dataclass(frozen=True)
class G8GTSidecarGenerationConfig:
    output_root: Path
    dataset_name: str
    gt_sidecar_dir: str = "audit"
    rewrite_existing: bool = False


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _count_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _collect_clip_ids(output_root: Path, dataset_name: str) -> List[int]:
    clip_ids: set[int] = set()
    for rel in (
        Path("exports") / dataset_name / "trajectory_records.jsonl",
        Path("exports_gt") / dataset_name / "trajectory_records.jsonl",
    ):
        path = output_root / rel
        if not path.is_file():
            continue
        for rec in read_carrier_records(path):
            try:
                clip_ids.add(int(rec.get("clip_id", -1)))
            except Exception:
                continue
    return sorted(x for x in clip_ids if x >= 0)


def _expected_paths(output_root: Path, dataset_name: str, gt_sidecar_dir: str) -> Dict[str, Path]:
    split = _dataset_split(dataset_name)
    sidecar_root = output_root / gt_sidecar_dir
    return {
        "match": sidecar_root / f"trajectory_gt_match_{split}_mainline.jsonl",
        "identity": sidecar_root / f"trajectory_gt_identity_{split}_gt.jsonl",
    }


def _summarize_sidecars(output_root: Path, dataset_name: str, gt_sidecar_dir: str) -> Dict[str, Any]:
    paths = _expected_paths(output_root, dataset_name, gt_sidecar_dir)
    match_rows = _load_jsonl(paths["match"])
    identity_rows = _load_jsonl(paths["identity"])
    usable_match = sum(1 for row in match_rows if bool(row.get("audit_usable", False)))
    usable_identity = sum(1 for row in identity_rows if bool(row.get("audit_usable", False)))
    sample_match = match_rows[0] if match_rows else None
    sample_identity = identity_rows[0] if identity_rows else None
    return {
        "dataset_name": dataset_name,
        "split": _dataset_split(dataset_name),
        "sidecar_dir": gt_sidecar_dir,
        "artifacts": {
            "match": str(paths["match"]),
            "identity": str(paths["identity"]),
        },
        "row_counts": {
            "match": len(match_rows),
            "identity": len(identity_rows),
        },
        "usable_row_counts": {
            "match": int(usable_match),
            "identity": int(usable_identity),
        },
        "sample_fields": {
            "match_keys": sorted(sample_match.keys()) if isinstance(sample_match, dict) else [],
            "identity_keys": sorted(sample_identity.keys()) if isinstance(sample_identity, dict) else [],
        },
    }


def run_g8_gt_sidecar_generation(config: G8GTSidecarGenerationConfig) -> Dict[str, Any]:
    output_root = config.output_root.expanduser().resolve()
    clip_ids = _collect_clip_ids(output_root, config.dataset_name)
    if not clip_ids:
        raise RuntimeError(
            f"NO_CLIP_IDS_FOR_DATASET:{config.dataset_name}: expected exports/{{dataset}}/trajectory_records.jsonl and/or exports_gt/{{dataset}}/trajectory_records.jsonl under {output_root}"
        )
    paths = _expected_paths(output_root, config.dataset_name, config.gt_sidecar_dir)
    preexisting = {name: path.is_file() for name, path in paths.items()}
    if (not config.rewrite_existing) and all(preexisting.values()):
        status = "REUSED_EXISTING"
    else:
        _sidecar_match_rows(
            output_root=output_root,
            dataset_name=config.dataset_name,
            clip_ids=clip_ids,
            trajectory_source_branch="mainline",
        )
        _sidecar_match_rows(
            output_root=output_root,
            dataset_name=config.dataset_name,
            clip_ids=clip_ids,
            trajectory_source_branch="gt_upper_bound",
        )
        status = "GENERATED"
    summary = _summarize_sidecars(output_root, config.dataset_name, config.gt_sidecar_dir)
    summary.update(
        {
            "status": status,
            "audit_pipeline_id": "g8_gt_sidecar_only_v1",
            "audit_entrypoint": "videocutler/run_stageb_audit_g8_gt_sidecar.py",
            "phase_scope": "gt_sidecar_generation_only",
            "generated_by_new_chain": True,
            "clip_count": len(clip_ids),
            "clip_id_sample": clip_ids[:8],
            "rewrite_existing": bool(config.rewrite_existing),
            "preexisting": preexisting,
            "current_asset_mode_behavior": "sidecar_only_generation_no_metric_audit",
        }
    )
    summary_root = output_root / config.gt_sidecar_dir / "gt_sidecar_generation"
    _write_json(summary_root / f"{config.dataset_name}_summary.json", summary)
    _write_md(
        summary_root / f"{config.dataset_name}_summary.md",
        [
            "# G8 GT Sidecar Generation",
            "",
            f"- dataset_name: `{config.dataset_name}`",
            f"- status: `{status}`",
            f"- audit_pipeline_id: `{summary['audit_pipeline_id']}`",
            f"- generated_by_new_chain: `{summary['generated_by_new_chain']}`",
            f"- clip_count: `{summary['clip_count']}`",
            f"- match_rows: `{summary['row_counts']['match']}`",
            f"- identity_rows: `{summary['row_counts']['identity']}`",
            f"- usable_match_rows: `{summary['usable_row_counts']['match']}`",
            f"- usable_identity_rows: `{summary['usable_row_counts']['identity']}`",
            f"- match_artifact: `{summary['artifacts']['match']}`",
            f"- identity_artifact: `{summary['artifacts']['identity']}`",
        ],
    )
    return summary
