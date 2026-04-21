from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from videocutler.ext_stageb_ovvis.audit.extra_recovery_audit import _dataset_split, _load_jsonl
from videocutler.ext_stageb_ovvis.audit.gt_attribution_rank_audit import _video_iou
from videocutler.ext_stageb_ovvis.banks.carrier_bank import read_carrier_records
from videocutler.ext_stageb_ovvis.eval.g8_bridge import densify_segmentations

Record = Dict[str, Any]
MATCH_POLICY_ID = "videocutler_video_iou_050_v1"
MATCH_IOU_THRESHOLD = 0.5


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


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _valid_carrier_records(records: Sequence[Mapping[str, Any]]) -> List[Record]:
    out: List[Record] = []
    for record in records:
        if not bool(record.get("valid_carrier", False)):
            continue
        trajectory_id = str(record.get("trajectory_id", "")).strip()
        if not trajectory_id:
            continue
        out.append(dict(record))
    return out


def _row_video_length(record: Mapping[str, Any]) -> int:
    frame_indices = [int(x) for x in list(record.get("frame_indices", []))]
    return (max(frame_indices) + 1) if frame_indices else 0


def _row_segmentations(record: Mapping[str, Any]) -> List[Any]:
    return densify_segmentations(record, video_length=_row_video_length(record))


def _write_jsonl(path: Path, rows: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_match_rows(*, output_root: Path, dataset_name: str, clip_ids: Sequence[int]) -> List[Record]:
    clip_id_set = {int(x) for x in clip_ids}
    main_records = _valid_carrier_records([
        rec
        for rec in read_carrier_records(output_root / "exports" / dataset_name / "trajectory_records.jsonl")
        if int(rec.get("clip_id", -1)) in clip_id_set
    ])
    gt_records = _valid_carrier_records([
        rec
        for rec in read_carrier_records(output_root / "exports_gt" / dataset_name / "trajectory_records.jsonl")
        if int(rec.get("clip_id", -1)) in clip_id_set
    ])
    main_by_clip: Dict[int, List[Record]] = {}
    gt_by_clip: Dict[int, List[Record]] = {}
    for rec in main_records:
        main_by_clip.setdefault(int(rec["clip_id"]), []).append(dict(rec))
    for rec in gt_records:
        gt_by_clip.setdefault(int(rec["clip_id"]), []).append(dict(rec))

    match_rows: List[Record] = []
    for clip_id in sorted(clip_id_set):
        gt_clip_records = gt_by_clip.get(clip_id, [])
        gt_dense_cache = {str(rec.get("trajectory_id", "")): _row_segmentations(rec) for rec in gt_clip_records}
        for main_rec in main_by_clip.get(clip_id, []):
            pred_dense = _row_segmentations(main_rec)
            best_row: Optional[Record] = None
            best_key = (-1.0, "")
            image_size = list(main_rec.get("image_size") or [0, 0])
            h = int(image_size[0] or 0) if len(image_size) >= 1 else 0
            w = int(image_size[1] or 0) if len(image_size) >= 2 else 0
            for gt_rec in gt_clip_records:
                gt_tid = str(gt_rec.get("trajectory_id", ""))
                iou = _video_iou(pred_dense, gt_dense_cache[gt_tid], h=h, w=w)
                key = (float(iou), gt_tid)
                if key > best_key:
                    best_key = key
                    gt_class_id = int(gt_rec.get("pred_label_raw")) if gt_rec.get("pred_label_raw") is not None else None
                    best_row = {
                        "dataset_name": dataset_name,
                        "trajectory_source_branch": "mainline",
                        "split_tag": main_rec.get("split_tag"),
                        "trajectory_id": str(main_rec.get("trajectory_id", "")),
                        "clip_id": int(main_rec.get("clip_id", -1)),
                        "video_id": int(main_rec.get("video_id", -1)) if main_rec.get("video_id") is not None else None,
                        "matched_gt_track_id": gt_tid,
                        "matched_gt_raw_id": gt_class_id,
                        "matched_gt_class_id": gt_class_id,
                        "match_iou_video": float(iou),
                        "match_policy_id": MATCH_POLICY_ID,
                        "match_iou_threshold": float(MATCH_IOU_THRESHOLD),
                        "audit_usable": bool(gt_class_id is not None and float(iou) >= float(MATCH_IOU_THRESHOLD)),
                    }
            if best_row is None:
                best_row = {
                    "dataset_name": dataset_name,
                    "trajectory_source_branch": "mainline",
                    "split_tag": main_rec.get("split_tag"),
                    "trajectory_id": str(main_rec.get("trajectory_id", "")),
                    "clip_id": int(main_rec.get("clip_id", -1)),
                    "video_id": int(main_rec.get("video_id", -1)) if main_rec.get("video_id") is not None else None,
                    "matched_gt_track_id": None,
                    "matched_gt_raw_id": None,
                    "matched_gt_class_id": None,
                    "match_iou_video": 0.0,
                    "match_policy_id": MATCH_POLICY_ID,
                    "match_iou_threshold": float(MATCH_IOU_THRESHOLD),
                    "audit_usable": False,
                }
            match_rows.append(best_row)
    return sorted(match_rows, key=lambda row: str(row.get("trajectory_id", "")))


def _build_identity_rows(*, output_root: Path, dataset_name: str, clip_ids: Sequence[int]) -> List[Record]:
    clip_id_set = {int(x) for x in clip_ids}
    gt_records = _valid_carrier_records([
        rec
        for rec in read_carrier_records(output_root / "exports_gt" / dataset_name / "trajectory_records.jsonl")
        if int(rec.get("clip_id", -1)) in clip_id_set
    ])
    rows: List[Record] = []
    for gt_rec in gt_records:
        gt_class_id = int(gt_rec.get("pred_label_raw")) if gt_rec.get("pred_label_raw") is not None else None
        rows.append(
            {
                "dataset_name": dataset_name,
                "trajectory_source_branch": "gt_upper_bound",
                "split_tag": gt_rec.get("split_tag"),
                "trajectory_id": str(gt_rec.get("trajectory_id", "")),
                "clip_id": int(gt_rec.get("clip_id", -1)),
                "video_id": int(gt_rec.get("video_id", -1)) if gt_rec.get("video_id") is not None else None,
                "matched_gt_track_id": str(gt_rec.get("trajectory_id", "")),
                "matched_gt_raw_id": gt_class_id,
                "matched_gt_class_id": gt_class_id,
                "match_iou_video": 1.0,
                "match_policy_id": "gt_identity",
                "match_iou_threshold": 1.0,
                "audit_usable": bool(gt_class_id is not None),
            }
        )
    return sorted(rows, key=lambda row: str(row.get("trajectory_id", "")))


def _summarize_sidecars(output_root: Path, dataset_name: str, gt_sidecar_dir: str) -> Dict[str, Any]:
    paths = _expected_paths(output_root, dataset_name, gt_sidecar_dir)
    match_rows = _load_jsonl(paths["match"])
    identity_rows = _load_jsonl(paths["identity"])
    usable_match = sum(1 for row in match_rows if bool(row.get("audit_usable", False)))
    usable_identity = sum(1 for row in identity_rows if bool(row.get("audit_usable", False)))
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
        "unusable_row_counts": {
            "match": int(len(match_rows) - usable_match),
            "identity": int(len(identity_rows) - usable_identity),
        },
        "unique_counts": {
            "trajectory_id": int(len({str(row.get("trajectory_id", "")) for row in match_rows if str(row.get("trajectory_id", "")).strip()})),
            "matched_gt_raw_id": int(len({int(row.get("matched_gt_raw_id")) for row in match_rows if row.get("matched_gt_raw_id") is not None})),
            "matched_gt_class_id": int(len({int(row.get("matched_gt_class_id")) for row in match_rows if row.get("matched_gt_class_id") is not None})),
        },
        "sample_fields": {
            "match_keys": sorted(match_rows[0].keys()) if match_rows else [],
            "identity_keys": sorted(identity_rows[0].keys()) if identity_rows else [],
        },
        "sha256": {
            "match": _sha256(paths["match"]),
            "identity": _sha256(paths["identity"]),
        },
        "match_policy_id": MATCH_POLICY_ID,
        "match_iou_threshold": float(MATCH_IOU_THRESHOLD),
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
        _write_jsonl(paths["match"], _build_match_rows(output_root=output_root, dataset_name=config.dataset_name, clip_ids=clip_ids))
        _write_jsonl(paths["identity"], _build_identity_rows(output_root=output_root, dataset_name=config.dataset_name, clip_ids=clip_ids))
        status = "GENERATED"
    summary = _summarize_sidecars(output_root, config.dataset_name, config.gt_sidecar_dir)
    summary.update(
        {
            "status": status,
            "audit_pipeline_id": "g8_gt_sidecar_only_v2",
            "audit_entrypoint": "videocutler/run_stageb_audit_g8_gt_sidecar.py",
            "phase_scope": "gt_sidecar_generation_only",
            "generated_by_new_chain": True,
            "clip_count": len(clip_ids),
            "clip_id_sample": clip_ids[:8],
            "rewrite_existing": bool(config.rewrite_existing),
            "preexisting": preexisting,
            "current_asset_mode_behavior": "sidecar_only_generation_video_iou_050_only",
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
            f"- match_policy_id: `{summary['match_policy_id']}`",
            f"- match_iou_threshold: `{summary['match_iou_threshold']}`",
            f"- match_rows: `{summary['row_counts']['match']}`",
            f"- identity_rows: `{summary['row_counts']['identity']}`",
            f"- usable_match_rows: `{summary['usable_row_counts']['match']}`",
            f"- usable_identity_rows: `{summary['usable_row_counts']['identity']}`",
            f"- unique_trajectory_id_count: `{summary['unique_counts']['trajectory_id']}`",
            f"- unique_matched_gt_raw_id_count: `{summary['unique_counts']['matched_gt_raw_id']}`",
            f"- unique_matched_gt_class_id_count: `{summary['unique_counts']['matched_gt_class_id']}`",
            f"- match_artifact: `{summary['artifacts']['match']}`",
            f"- identity_artifact: `{summary['artifacts']['identity']}`",
        ],
    )
    return summary
