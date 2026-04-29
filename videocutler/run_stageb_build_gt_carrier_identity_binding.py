#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


Record = Dict[str, Any]


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def _iter_jsonl(path: Path) -> Iterator[Record]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _jsonl_schema_summary(path: Path, important_keys: Sequence[str], *, sample_limit: int = 5, scan_limit: Optional[int] = None) -> Record:
    out: Record = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": int(path.stat().st_size) if path.is_file() else None,
        "row_count": 0,
        "sampled_count": 0,
        "sampled_key_sets": [],
        "important_key_presence_counts": {key: 0 for key in important_keys},
        "compact_examples": [],
    }
    if not path.is_file():
        return out
    for row in _iter_jsonl(path):
        if scan_limit is not None and int(out["row_count"]) >= int(scan_limit):
            break
        out["row_count"] += 1
        for key in important_keys:
            if key in row and row.get(key) is not None:
                out["important_key_presence_counts"][key] += 1
        if int(out["sampled_count"]) < sample_limit:
            out["sampled_count"] += 1
            out["sampled_key_sets"].append(sorted(str(k) for k in row.keys()))
            out["compact_examples"].append({key: row.get(key) for key in important_keys if key in row})
    return out


def _sample_json_array(path: Path, key: str, important_keys: Sequence[str], *, sample_limit: int = 3) -> Record:
    out: Record = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": int(path.stat().st_size) if path.is_file() else None,
        "array_key": key,
        "sampled_count": 0,
        "sampled_key_sets": [],
        "important_key_presence_counts": {k: 0 for k in important_keys},
        "compact_examples": [],
    }
    if not path.is_file():
        return out
    decoder = json.JSONDecoder()
    token = json.dumps(key)
    buffer = ""
    found = False
    with path.open("r", encoding="utf-8") as handle:
        while int(out["sampled_count"]) < sample_limit:
            chunk = handle.read(1 << 20)
            if not chunk and not buffer:
                break
            buffer += chunk
            if not found:
                idx = buffer.find(token)
                if idx < 0:
                    if not chunk:
                        break
                    buffer = buffer[-4096:]
                    continue
                arr_start = buffer.find("[", idx)
                if arr_start < 0:
                    if not chunk:
                        break
                    buffer = buffer[idx:]
                    continue
                buffer = buffer[arr_start + 1 :]
                found = True
            while int(out["sampled_count"]) < sample_limit:
                buffer = buffer.lstrip()
                if not buffer or buffer[0] == "]":
                    break
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                try:
                    obj, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                if isinstance(obj, dict):
                    out["sampled_count"] += 1
                    out["sampled_key_sets"].append(sorted(str(k) for k in obj.keys()))
                    compact = {k: obj.get(k) for k in important_keys if k in obj}
                    out["compact_examples"].append(compact)
                    for item_key in important_keys:
                        if item_key in obj and obj.get(item_key) is not None:
                            out["important_key_presence_counts"][item_key] += 1
                buffer = buffer[end:]
            if not chunk:
                break
    return out


def _load_full_y_by_clip(path: Path) -> Tuple[Dict[int, set[int]], Record]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", []) if isinstance(payload, dict) else []
    by_clip: Dict[int, set[int]] = {}
    union: set[int] = set()
    for rec in records:
        if not isinstance(rec, Mapping):
            continue
        clip_id = _safe_int(rec.get("clip_id"), _safe_int(rec.get("video_id"), None))
        if clip_id is None:
            continue
        vals = {_safe_int(x) for x in rec.get("full_y_raw_ids", [])}
        cleaned = {int(x) for x in vals if x is not None}
        by_clip[int(clip_id)] = cleaned
        union.update(cleaned)
    return by_clip, {"full_y_record_count": len(by_clip), "full_y_union_count": len(union)}


def _load_gt_trajectory_identity(path: Path) -> Tuple[Dict[str, Record], Record]:
    lookup: Dict[str, Record] = {}
    raw_union: set[int] = set()
    duplicate_count = 0
    row_count = 0
    usable_count = 0
    for row in _iter_jsonl(path):
        row_count += 1
        tid = str(row.get("trajectory_id", "")).strip()
        legacy_id = _safe_int(row.get("pred_label_raw"), None)
        raw_id = int(legacy_id) + 1 if legacy_id is not None else _safe_int(row.get("raw_category_id"), None)
        if not tid:
            continue
        if tid in lookup:
            duplicate_count += 1
        if raw_id is not None:
            usable_count += 1
            raw_union.add(int(raw_id))
        lookup[tid] = {
            "trajectory_id": tid,
            "clip_id": _safe_int(row.get("clip_id"), _safe_int(row.get("video_id"), None)),
            "video_id": _safe_int(row.get("video_id"), _safe_int(row.get("clip_id"), None)),
            "raw_category_id": int(raw_id) if raw_id is not None else None,
            "legacy_pred_label_raw": int(legacy_id) if legacy_id is not None else None,
            "gt_track_id": tid,
        }
    return lookup, {
        "gt_trajectory_path": str(path),
        "gt_trajectory_row_count": row_count,
        "gt_trajectory_lookup_count": len(lookup),
        "gt_trajectory_usable_raw_id_count": usable_count,
        "gt_trajectory_duplicate_trajectory_id_count": duplicate_count,
        "gt_trajectory_raw_category_union_count": len(raw_union),
    }


def _build_binding(
    *,
    carrier_path: Path,
    gt_lookup: Mapping[str, Mapping[str, Any]],
    full_y_by_clip: Mapping[int, set[int]],
    output_jsonl: Path,
) -> Record:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    bound_count = 0
    full_y_checked = 0
    full_y_ok = 0
    raw_union: set[int] = set()
    clip_ids: set[int] = set()
    unbound_examples: List[Record] = []
    not_full_y_examples: List[Record] = []
    source_counts: Counter[str] = Counter()
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in _iter_jsonl(carrier_path):
            carrier_idx = row_count
            row_count += 1
            tid = str(row.get("trajectory_id", "")).strip()
            clip_id = _safe_int(row.get("clip_id"), _safe_int(row.get("video_id"), None))
            video_id = _safe_int(row.get("video_id"), clip_id)
            gt = gt_lookup.get(tid, {})
            raw_id = _safe_int(gt.get("raw_category_id"), None)
            if raw_id is None:
                if len(unbound_examples) < 16:
                    unbound_examples.append(
                        {
                            "carrier_row_index": carrier_idx,
                            "trajectory_id": tid,
                            "clip_id": clip_id,
                            "available_keys": sorted(str(k) for k in row.keys()),
                        }
                    )
                continue
            bound_count += 1
            raw_union.add(int(raw_id))
            if clip_id is not None:
                clip_ids.add(int(clip_id))
                full_set = full_y_by_clip.get(int(clip_id), set())
                full_y_checked += 1
                if int(raw_id) in full_set:
                    full_y_ok += 1
                elif len(not_full_y_examples) < 16:
                    not_full_y_examples.append(
                        {
                            "carrier_row_index": carrier_idx,
                            "trajectory_id": tid,
                            "clip_id": int(clip_id),
                            "raw_category_id": int(raw_id),
                            "full_y_raw_ids_sample": sorted(full_set)[:32],
                        }
                    )
            source_counts["exports_gt_trajectory_records_pred_label_raw_plus_one"] += 1
            out = {
                "carrier_row_index": int(carrier_idx),
                "carrier_id": None,
                "trajectory_id": tid or None,
                "clip_id": int(clip_id) if clip_id is not None else gt.get("clip_id"),
                "video_id": int(video_id) if video_id is not None else gt.get("video_id"),
                "gt_track_id": gt.get("gt_track_id", tid or None),
                "gt_instance_id": None,
                "raw_category_id": int(raw_id),
                "matched_gt_raw_id": int(raw_id),
                "matched_gt_class_id": int(raw_id),
                "match_iou_mean": 1.0,
                "match_iou_p50": 1.0,
                "match_iou_video": 1.0,
                "audit_usable": True,
                "binding_source": "exports_gt_trajectory_records.pred_label_raw_plus_one",
                "binding_key": "trajectory_id",
                "binding_confidence": "deterministic",
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
    return {
        "status": "PASS" if row_count and bound_count / row_count >= 0.95 else "FAIL_LOW_COVERAGE",
        "gt_carrier_row_count": int(row_count),
        "bound_row_count": int(bound_count),
        "unbound_row_count": int(row_count - bound_count),
        "identity_coverage_rate": float(bound_count / row_count) if row_count else None,
        "raw_category_union_count": int(len(raw_union)),
        "clip_count": int(len(clip_ids)),
        "binding_source": "exports_gt_trajectory_records.pred_label_raw_plus_one",
        "binding_key": "trajectory_id",
        "raw_id_field_used": "pred_label_raw + 1",
        "join_key_used": "trajectory_id",
        "source_counts": dict(source_counts),
        "unbound_examples_compact": unbound_examples,
        "full_y_consistency_checked": bool(full_y_checked > 0),
        "gt_carrier_class_in_full_y_rate": float(full_y_ok / full_y_checked) if full_y_checked else None,
        "gt_carrier_class_not_in_full_y_count": int(full_y_checked - full_y_ok),
        "gt_carrier_class_not_in_full_y_examples_compact": not_full_y_examples,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic GT-carrier identity binding.")
    parser.add_argument("--dataset_name", default="lvvis_train_base")
    parser.add_argument("--gt_carrier_path", required=True)
    parser.add_argument("--gt_trajectory_path", default=None)
    parser.add_argument("--sparse_gt_identity_path", required=True)
    parser.add_argument("--train_instances_json", required=True)
    parser.add_argument("--full_y_path", required=True)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--output_summary", default=None)
    parser.add_argument("--schema_audit_dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    carrier_path = Path(args.gt_carrier_path).expanduser().resolve()
    gt_trajectory_path = (
        Path(args.gt_trajectory_path).expanduser().resolve()
        if args.gt_trajectory_path
        else Path("/home/zyy/code/wsovvis_asserts/exports_gt") / args.dataset_name / "trajectory_records.jsonl"
    )
    output_jsonl = (
        Path(args.output_jsonl).expanduser().resolve()
        if args.output_jsonl
        else carrier_path.parent / "gt_carrier_identity_binding.jsonl"
    )
    output_summary = (
        Path(args.output_summary).expanduser().resolve()
        if args.output_summary
        else carrier_path.parent / "gt_carrier_identity_binding_summary.json"
    )
    schema_dir = Path(args.schema_audit_dir).expanduser().resolve()
    schema_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        schema_dir / "gt_carrier_schema_summary.json",
        _jsonl_schema_summary(
            carrier_path,
            ("trajectory_id", "carrier_id", "clip_id", "video_id", "raw_id", "category_id", "gt_raw_id", "matched_gt_raw_id", "gt_category_id", "track_id", "gt_track_id", "instance_id", "z_norm_path"),
            scan_limit=5000,
        ),
    )
    _write_json(
        schema_dir / "sparse_gt_identity_schema_summary.json",
        _jsonl_schema_summary(
            Path(args.sparse_gt_identity_path).expanduser().resolve(),
            ("trajectory_id", "clip_id", "video_id", "matched_gt_raw_id", "matched_gt_class_id", "matched_gt_track_id", "match_iou_mean", "match_iou_p50", "audit_usable"),
        ),
    )
    _write_json(
        schema_dir / "train_instances_schema_summary.json",
        {
            "path": str(Path(args.train_instances_json).expanduser().resolve()),
            "exists": Path(args.train_instances_json).expanduser().is_file(),
            "size_bytes": int(Path(args.train_instances_json).expanduser().stat().st_size) if Path(args.train_instances_json).expanduser().is_file() else None,
            "videos": _sample_json_array(Path(args.train_instances_json).expanduser().resolve(), "videos", ("id", "file_names", "width", "height")),
            "annotations": _sample_json_array(Path(args.train_instances_json).expanduser().resolve(), "annotations", ("id", "video_id", "category_id", "segmentations")),
        },
    )
    _write_json(
        schema_dir / "candidate_join_key_summary.json",
        {
            "selected_strategy": "join GT carrier records to exports_gt trajectory_records by trajectory_id",
            "priority_rule_used": "GT carrier row does not contain direct raw id; trajectory_id deterministically joins to exports_gt pred_label_raw and converts to raw category id via +1",
            "carrier_path": str(carrier_path),
            "gt_trajectory_path": str(gt_trajectory_path),
            "join_key": "trajectory_id",
            "raw_id_field": "pred_label_raw + 1",
        },
    )

    full_y_by_clip, full_y_meta = _load_full_y_by_clip(Path(args.full_y_path).expanduser().resolve())
    gt_lookup, gt_meta = _load_gt_trajectory_identity(gt_trajectory_path)
    binding_summary = _build_binding(
        carrier_path=carrier_path,
        gt_lookup=gt_lookup,
        full_y_by_clip=full_y_by_clip,
        output_jsonl=output_jsonl,
    )
    summary = {
        "dataset_name": args.dataset_name,
        "output_jsonl": str(output_jsonl),
        **binding_summary,
        **full_y_meta,
        **gt_meta,
    }
    _write_json(output_summary, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
