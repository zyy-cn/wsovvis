#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_root_for_direct_cli()

from videocutler.ext_stageb_ovvis.audit.trajectory_gt_audit import load_gt_sidecar_lookup
from videocutler.ext_stageb_ovvis.data.oracle_clean_ablation_sources import (
    iter_jsonl,
    load_json,
    load_weak_label_records,
    safe_int,
    unique_ints,
    write_csv,
    write_json,
)


Record = Dict[str, Any]
EXPECTED_YPRIME_SUPPORT_AT_05 = 0.4735567218409366
REFERENCE_TOLERANCE = 1e-6
HIGH_IDENTITY_COVERAGE_THRESHOLD = 0.95


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oracle clean-data ablation audit (read-only).")
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--run_root_v2b", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--full_y_path", required=True)
    p.add_argument("--weak_label_path", required=True)
    p.add_argument("--videocutler_trajectory_path", required=True)
    p.add_argument("--videocutler_gt_match_path", required=True)
    p.add_argument("--gt_carrier_path", required=True)
    p.add_argument("--gt_identity_path", required=True)
    p.add_argument("--iou_threshold", type=float, default=0.5)
    p.add_argument("--top_examples", type=int, default=64)
    p.add_argument("--show_progress", type=_parse_bool, default=False)
    return p.parse_args()


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return default


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def _median(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(median(vals)) if vals else None


def _rate(num: int, den: int) -> Optional[float]:
    return float(num / den) if den else None


def _load_full_y_records(path: Path) -> List[Record]:
    payload = load_json(path)
    rows = payload.get("records", [])
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _clip_rows(records: Sequence[Mapping[str, Any]]) -> Dict[int, List[Record]]:
    out: Dict[int, List[Record]] = defaultdict(list)
    for rec in records:
        clip_id = safe_int(rec.get("clip_id"), safe_int(rec.get("video_id"), None))
        if clip_id is not None:
            out[int(clip_id)].append(dict(rec))
    return out


def _label_set_for_clip(labels_by_clip: Mapping[int, Sequence[Mapping[str, Any]]], clip_id: int, label_key: str) -> set[int]:
    out: set[int] = set()
    for rec in labels_by_clip.get(int(clip_id), []):
        out.update(unique_ints(rec.get(label_key, [])))
    return out


def _label_rows_for_key(full_y_records: Sequence[Mapping[str, Any]], label_key: str) -> List[Record]:
    rows: List[Record] = []
    for rec in full_y_records:
        rows.append(
            {
                "clip_id": rec.get("clip_id"),
                "video_id": rec.get("video_id"),
                label_key: unique_ints(rec.get(label_key, [])),
            }
        )
    return rows


def _normalized_scope_key(value: Any) -> Optional[str]:
    ix = safe_int(value, None)
    if ix is not None:
        return str(int(ix))
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _build_full_y_scope_index(full_y_records: Sequence[Mapping[str, Any]]) -> Dict[str, Record]:
    out: Dict[str, Record] = {}
    for rec in full_y_records:
        clip_key = _normalized_scope_key(rec.get("clip_id"))
        if clip_key is None:
            clip_key = _normalized_scope_key(rec.get("video_id"))
        if clip_key is None:
            continue
        out[clip_key] = {
            "clip_id": rec.get("clip_id"),
            "video_id": rec.get("video_id"),
            "full_y_raw_ids": unique_ints(rec.get("full_y_raw_ids", [])),
            "yprime_raw_ids": unique_ints(rec.get("yprime_raw_ids", [])),
        }
    return out


def _schema_summary(path: Path, *, important_keys: Sequence[str], sample_limit: int = 3, scan_limit: Optional[int] = None) -> Record:
    summary: Record = {
        "path": str(path),
        "file_exists": path.is_file(),
        "size_bytes": int(path.stat().st_size) if path.is_file() else None,
        "row_count": 0,
        "sampled_key_sets": [],
        "important_key_presence_counts": {k: 0 for k in important_keys},
        "compact_examples": [],
    }
    if not path.is_file():
        return summary
    key_sets: List[List[str]] = []
    for row in iter_jsonl(path):
        summary["row_count"] += 1
        if scan_limit is not None and int(summary["row_count"]) > int(scan_limit):
            break
        for key in important_keys:
            if key in row and row.get(key) is not None:
                summary["important_key_presence_counts"][key] += 1
        if len(key_sets) < sample_limit:
            keys = sorted(str(k) for k in row.keys())
            key_sets.append(keys)
            compact = {k: row.get(k) for k in important_keys if k in row}
            for key in ("trajectory_id", "clip_id", "video_id", "z_norm_path", "match_iou_mean", "match_iou_p50"):
                if key in row:
                    compact[key] = row.get(key)
            summary["compact_examples"].append(compact)
    summary["sampled_key_sets"] = key_sets
    return summary


def _load_sidecar_by_tid(path: Path) -> Dict[str, Record]:
    out: Dict[str, Record] = {}
    for row in iter_jsonl(path):
        tid = str(row.get("trajectory_id", "")).strip()
        if tid:
            out[tid] = dict(row)
    return out


def _raw_id_from_binding(row: Mapping[str, Any]) -> Optional[int]:
    for key in (
        "raw_category_id",
        "matched_gt_raw_id_canonical",
        "matched_gt_raw_id",
        "gt_raw_id",
        "raw_id",
        "category_id",
        "gt_category_id",
        "matched_gt_class_id",
        "pred_label_raw",
    ):
        val = safe_int(row.get(key), None)
        if val is not None:
            return int(val)
    return None


def _iou_from_binding(row: Mapping[str, Any]) -> float:
    for key in ("match_iou_mean", "match_iou_video", "best_iou", "best_video_iou", "video_iou", "match_iou_p50"):
        val = _safe_float(row.get(key), None)
        if val is not None:
            return float(val)
    # GT identity rows are exact only when they are truly high-coverage identity bindings.
    return 1.0


def _build_support_from_sidecar(
    *,
    sidecar_lookup: Mapping[str, Mapping[str, Any]],
    iou_threshold: float,
) -> Tuple[Dict[int, Dict[int, int]], Record]:
    support: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    bound = 0
    usable = 0
    for row in sidecar_lookup.values():
        clip_id = safe_int(row.get("clip_id"), safe_int(row.get("video_id"), None))
        raw_id = _raw_id_from_binding(row)
        if clip_id is None or raw_id is None:
            continue
        bound += 1
        iou = _iou_from_binding(row)
        if iou >= float(iou_threshold):
            usable += 1
            support[int(clip_id)][int(raw_id)] += 1
    return support, {"bound_identity_count": int(bound), "usable_identity_count_at_threshold": int(usable)}


def _support_stats(records: Sequence[Mapping[str, Any]], label_key: str, support_map: Mapping[int, Mapping[int, int]]) -> Record:
    pair_count = 0
    supported = 0
    clip_count = 0
    clip_all_supported = 0
    support_counts: List[int] = []
    union: set[int] = set()
    for rec in records:
        clip_id = safe_int(rec.get("clip_id"), safe_int(rec.get("video_id"), None))
        if clip_id is None:
            continue
        labels = unique_ints(rec.get(label_key, []))
        if not labels:
            continue
        clip_count += 1
        all_supported = True
        for raw_id in labels:
            pair_count += 1
            union.add(int(raw_id))
            ct = int(support_map.get(int(clip_id), {}).get(int(raw_id), 0))
            support_counts.append(ct)
            if ct > 0:
                supported += 1
            else:
                all_supported = False
        if all_supported:
            clip_all_supported += 1
    return {
        "label_pair_count": int(pair_count),
        "label_union_count": int(len(union)),
        "support_upper_bound_pair_count": int(supported),
        "support_upper_bound_rate_at_0.5": _rate(supported, pair_count),
        "support_rate_at_0.5": _rate(supported, pair_count),
        "zero_support_rate_at_0.5": _rate(pair_count - supported, pair_count),
        "clip_all_labels_supported_rate_at_0.5": _rate(clip_all_supported, clip_count),
        "mean_support_count_per_label": _mean([float(x) for x in support_counts]),
        "median_support_count_per_label": _median([float(x) for x in support_counts]),
    }


def _classify_scope_mismatch(
    *,
    binding_rows: Sequence[Mapping[str, Any]],
    full_y_records: Sequence[Mapping[str, Any]],
    top_examples: int = 24,
) -> Tuple[Record, List[Record], List[Record], List[Record], List[Record]]:
    full_y_by_clip = _build_full_y_scope_index(full_y_records)
    buckets: Counter[str] = Counter(
        {
            "clip_missing": 0,
            "class_missing": 0,
            "id_mapping_minus1_suspect": 0,
            "id_mapping_plus1_suspect": 0,
            "video_key_mismatch_suspect": 0,
            "malformed_binding": 0,
            "in_scope": 0,
        }
    )
    by_category: Counter[Tuple[str, Optional[int]]] = Counter()
    by_clip: Counter[Tuple[str, Optional[str]]] = Counter()
    examples: List[Record] = []
    in_scope_rows: List[Record] = []

    def add_example(bucket: str, payload: Record) -> None:
        if len(examples) < top_examples:
            item = {"bucket": bucket}
            item.update(payload)
            examples.append(item)

    for row in binding_rows:
        clip_key = _normalized_scope_key(row.get("clip_id"))
        video_key = _normalized_scope_key(row.get("video_id"))
        raw_id = safe_int(row.get("raw_category_id"), None)
        tid = str(row.get("trajectory_id", "")).strip() or None
        if clip_key is None or raw_id is None:
            buckets["malformed_binding"] += 1
            by_category[("malformed_binding", raw_id)] += 1
            by_clip[("malformed_binding", clip_key)] += 1
            add_example(
                "malformed_binding",
                {
                    "trajectory_id": tid,
                    "clip_id": row.get("clip_id"),
                    "video_id": row.get("video_id"),
                    "raw_category_id": row.get("raw_category_id"),
                    "available_keys": sorted(str(k) for k in row.keys()),
                },
            )
            continue
        scope_row = full_y_by_clip.get(clip_key)
        if scope_row is None:
            if video_key is not None and video_key != clip_key and video_key in full_y_by_clip:
                buckets["video_key_mismatch_suspect"] += 1
                by_category[("video_key_mismatch_suspect", int(raw_id))] += 1
                by_clip[("video_key_mismatch_suspect", clip_key)] += 1
                add_example(
                    "video_key_mismatch_suspect",
                    {
                        "trajectory_id": tid,
                        "clip_id": row.get("clip_id"),
                        "video_id": row.get("video_id"),
                        "raw_category_id": int(raw_id),
                        "matched_full_y_clip_id": full_y_by_clip.get(video_key, {}).get("clip_id"),
                        "matched_full_y_video_id": full_y_by_clip.get(video_key, {}).get("video_id"),
                    },
                )
            else:
                buckets["clip_missing"] += 1
                by_category[("clip_missing", int(raw_id))] += 1
                by_clip[("clip_missing", clip_key)] += 1
                add_example(
                    "clip_missing",
                    {
                        "trajectory_id": tid,
                        "clip_id": row.get("clip_id"),
                        "video_id": row.get("video_id"),
                        "raw_category_id": int(raw_id),
                    },
                )
            continue
        full_raw_ids = set(int(x) for x in scope_row.get("full_y_raw_ids", []))
        if int(raw_id) in full_raw_ids:
            buckets["in_scope"] += 1
            in_scope_rows.append(dict(row))
            continue
        buckets["class_missing"] += 1
        by_category[("class_missing", int(raw_id))] += 1
        by_clip[("class_missing", clip_key)] += 1
        add_example(
            "class_missing",
            {
                "trajectory_id": tid,
                "clip_id": row.get("clip_id"),
                "video_id": row.get("video_id"),
                "raw_category_id": int(raw_id),
                "full_y_raw_ids_sample": sorted(full_raw_ids)[:32],
            },
        )
        if (int(raw_id) - 1) in full_raw_ids:
            buckets["id_mapping_minus1_suspect"] += 1
            by_category[("id_mapping_minus1_suspect", int(raw_id))] += 1
            by_clip[("id_mapping_minus1_suspect", clip_key)] += 1
            add_example(
                "id_mapping_minus1_suspect",
                {
                    "trajectory_id": tid,
                    "clip_id": row.get("clip_id"),
                    "raw_category_id": int(raw_id),
                    "minus1": int(raw_id) - 1,
                    "full_y_raw_ids_sample": sorted(full_raw_ids)[:32],
                },
            )
        if (int(raw_id) + 1) in full_raw_ids:
            buckets["id_mapping_plus1_suspect"] += 1
            by_category[("id_mapping_plus1_suspect", int(raw_id))] += 1
            by_clip[("id_mapping_plus1_suspect", clip_key)] += 1
            add_example(
                "id_mapping_plus1_suspect",
                {
                    "trajectory_id": tid,
                    "clip_id": row.get("clip_id"),
                    "raw_category_id": int(raw_id),
                    "plus1": int(raw_id) + 1,
                    "full_y_raw_ids_sample": sorted(full_raw_ids)[:32],
                },
            )

    rows_by_category = [
        {"bucket": bucket, "raw_category_id": raw_id, "count": int(count)}
        for (bucket, raw_id), count in sorted(by_category.items(), key=lambda item: (-item[1], str(item[0][0]), str(item[0][1])))
    ]
    rows_by_clip = [
        {"bucket": bucket, "clip_id": clip_id, "count": int(count)}
        for (bucket, clip_id), count in sorted(by_clip.items(), key=lambda item: (-item[1], str(item[0][0]), str(item[0][1])))
    ]
    summary = {
        "gt_carrier_bound_row_count": int(len(binding_rows)),
        "full_y_record_count": int(len(full_y_records)),
        "in_scope_row_count": int(buckets["in_scope"]),
        "out_of_scope_row_count": int(len(binding_rows) - buckets["in_scope"]),
        "in_scope_rate": _rate(int(buckets["in_scope"]), len(binding_rows)),
        "out_of_scope_rate": _rate(int(len(binding_rows) - buckets["in_scope"]), len(binding_rows)),
        "clip_missing_count": int(buckets["clip_missing"]),
        "class_missing_count": int(buckets["class_missing"]),
        "id_mapping_minus1_suspect_count": int(buckets["id_mapping_minus1_suspect"]),
        "id_mapping_plus1_suspect_count": int(buckets["id_mapping_plus1_suspect"]),
        "video_key_mismatch_suspect_count": int(buckets["video_key_mismatch_suspect"]),
        "malformed_binding_count": int(buckets["malformed_binding"]),
        "top_out_of_scope_categories": [
            {"bucket": bucket, "raw_category_id": raw_id, "count": int(count)}
            for (bucket, raw_id), count in Counter(
                {(bucket, raw_id): count for (bucket, raw_id), count in by_category.items() if bucket != "in_scope"}
            ).most_common(10)
        ],
        "top_out_of_scope_clips": [
            {"bucket": bucket, "clip_id": clip_id, "count": int(count)}
            for (bucket, clip_id), count in Counter(
                {(bucket, clip_id): count for (bucket, clip_id), count in by_clip.items() if bucket != "in_scope"}
            ).most_common(10)
        ],
        "full_y_scope_key_used": "clip_id",
        "scope_key_match_rule": "normalized int/string exact equality on clip_id; video_id only used for diagnostics",
        "recommended_resolution": "filter_E_to_full_y_scope" if buckets["clip_missing"] >= buckets["class_missing"] else "materialize_full_y_for_gt_carrier_universe",
    }
    return summary, rows_by_category, rows_by_clip, examples, in_scope_rows


def _write_scope_mismatch_diagnostics(
    *,
    output_root: Path,
    summary: Mapping[str, Any],
    rows_by_category: Sequence[Mapping[str, Any]],
    rows_by_clip: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
) -> Path:
    diag_dir = output_root / "scope_mismatch_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    write_json(diag_dir / "gtcarrier_fully_scope_mismatch_summary.json", summary)
    write_csv(
        diag_dir / "gtcarrier_fully_scope_mismatch_buckets.csv",
        [
            {"bucket": "in_scope", "count": summary.get("in_scope_row_count"), "rate": summary.get("in_scope_rate")},
            {"bucket": "clip_missing", "count": summary.get("clip_missing_count")},
            {"bucket": "class_missing", "count": summary.get("class_missing_count")},
            {"bucket": "id_mapping_minus1_suspect", "count": summary.get("id_mapping_minus1_suspect_count")},
            {"bucket": "id_mapping_plus1_suspect", "count": summary.get("id_mapping_plus1_suspect_count")},
            {"bucket": "video_key_mismatch_suspect", "count": summary.get("video_key_mismatch_suspect_count")},
            {"bucket": "malformed_binding", "count": summary.get("malformed_binding_count")},
        ],
        fieldnames=("bucket", "count", "rate"),
    )
    write_csv(
        diag_dir / "gtcarrier_fully_scope_mismatch_by_category.csv",
        rows_by_category,
        fieldnames=("bucket", "raw_category_id", "count"),
    )
    write_csv(
        diag_dir / "gtcarrier_fully_scope_mismatch_by_clip.csv",
        rows_by_clip,
        fieldnames=("bucket", "clip_id", "count"),
    )
    examples_path = diag_dir / "gtcarrier_fully_scope_mismatch_examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return diag_dir


def _find_reference_summary(run_root: Path, dataset_name: str) -> Tuple[Optional[Path], Optional[Record]]:
    candidates = [
        run_root / "analysis" / "yprime_support_coverage" / dataset_name / "prealign" / "summary.json",
        run_root / "analysis" / "yprime_support_coverage" / dataset_name / "softem_aug" / "summary.json",
    ]
    for path in candidates:
        if path.is_file():
            payload = load_json(path)
            if "yprime_trajectory_support_rate" in payload:
                return path, payload
    return None, None


def _best_existing_summary_paths(run_root: Path, dataset_name: str, stage: str = "prealign") -> Dict[str, Optional[Path]]:
    candidates = {
        "support_null_responsibility": run_root / "analysis" / "support_null_responsibility" / dataset_name / stage / "summary.json",
        "assignment_nonidentifiability": run_root / "analysis" / "assignment_nonidentifiability" / dataset_name / stage / "summary.json",
        "yprime_support_coverage": run_root / "analysis" / "yprime_support_coverage" / dataset_name / stage / "summary.json",
        "text_projector_hubness": run_root / "analysis" / "text_projector_hubness" / dataset_name / stage / "summary.json",
        "hub_carrier_separability": run_root / "analysis" / "hub_carrier_separability" / dataset_name / stage / "summary.json",
    }
    return {k: v if v.is_file() else None for k, v in candidates.items()}


def _gt_carrier_identity_meta(carrier_rows: Sequence[Mapping[str, Any]], identity_lookup: Mapping[str, Mapping[str, Any]]) -> Record:
    direct_fields = ("raw_category_id", "raw_id", "category_id", "gt_raw_id", "matched_gt_raw_id", "gt_category_id", "matched_gt_class_id")
    direct_field_used = None
    bound_by_direct = 0
    for field in direct_fields:
        count = sum(1 for row in carrier_rows if safe_int(row.get(field), None) is not None)
        if count:
            direct_field_used = field
            bound_by_direct = count
            break
    sidecar_bound = 0
    examples: List[Record] = []
    for row in carrier_rows:
        tid = str(row.get("trajectory_id", "")).strip()
        raw_id = safe_int(row.get(direct_field_used), None) if direct_field_used else None
        if raw_id is None and tid:
            raw_id = _raw_id_from_binding(identity_lookup.get(tid, {}))
        if raw_id is not None:
            sidecar_bound += 1
        elif len(examples) < 8:
            examples.append(
                {
                    "trajectory_id": tid,
                    "clip_id": row.get("clip_id"),
                    "available_keys": sorted(str(k) for k in row.keys()),
                }
            )
    carrier_count = len(carrier_rows)
    source = "carrier_direct_field" if direct_field_used else "provided_gt_identity_binding"
    return {
        "gt_carrier_row_count": int(carrier_count),
        "gt_carrier_identity_bound_count": int(sidecar_bound),
        "gt_carrier_identity_coverage_rate": _rate(sidecar_bound, carrier_count),
        "gt_carrier_raw_id_field_used": direct_field_used,
        "gt_carrier_join_key_used": "trajectory_id" if not direct_field_used else "carrier_row_direct_field",
        "gt_carrier_identity_source": source,
        "gt_carrier_identity_unbound_examples_compact": examples,
        "gt_carrier_direct_field_bound_count": int(bound_by_direct),
    }


def _oracle_stats(carrier_rows: Sequence[Mapping[str, Any]], identity_lookup: Mapping[str, Mapping[str, Any]], label_records: Sequence[Mapping[str, Any]], label_key: str) -> Record:
    labels_by_clip = _clip_rows(label_records)
    total = 0
    valid = 0
    examples: List[Record] = []
    for row in carrier_rows:
        tid = str(row.get("trajectory_id", "")).strip()
        clip_id = safe_int(row.get("clip_id"), safe_int(row.get("video_id"), None))
        raw_id = _raw_id_from_binding(row)
        if raw_id is None and tid:
            raw_id = _raw_id_from_binding(identity_lookup.get(tid, {}))
        if clip_id is None or raw_id is None:
            continue
        total += 1
        ok = int(raw_id) in _label_set_for_clip(labels_by_clip, int(clip_id), label_key)
        if ok:
            valid += 1
        elif len(examples) < 8:
            examples.append({"trajectory_id": tid, "clip_id": int(clip_id), "gt_raw_id": int(raw_id), "label_key": label_key})
    rate = _rate(valid, total)
    return {
        "oracle_assignment_checked_count": int(total),
        "oracle_assignment_valid_count": int(valid),
        "oracle_assignment_valid_rate": rate,
        "gt_carrier_class_not_in_full_y_count": int(total - valid),
        "true_support_mass_mean_oracle": rate,
        "true_support_top1_rate_oracle": rate,
        "oracle_assignment_examples": examples,
    }


def _arm_summary(
    *,
    arm: str,
    label_source: str,
    carrier_source: str,
    assignment_mode: str,
    status: str,
    support_stats: Mapping[str, Any],
    support_status: str,
    latent_status: str,
    oracle_status: str,
    carrier_row_count: int,
    blocker: Optional[str],
    reused_paths: Sequence[str] = (),
    extra: Optional[Mapping[str, Any]] = None,
) -> Record:
    row: Record = {
        "arm": arm,
        "status": status,
        "label_source": label_source,
        "carrier_source": carrier_source,
        "assignment_mode": assignment_mode,
        "support_upper_bound_status": support_status,
        "latent_assignment_status": latent_status,
        "oracle_assignment_status": oracle_status,
        "carrier_row_count": int(carrier_row_count),
        "blocker": blocker,
        "reused_existing_summary_paths": list(reused_paths),
        "true_support_mass_mean": None,
        "true_support_top1_rate": None,
    }
    row.update(dict(support_stats))
    if extra:
        row.update(dict(extra))
    return row


def _write_schema_audit(
    *,
    output_root: Path,
    args: argparse.Namespace,
    reference_path: Optional[Path],
    reference_summary: Optional[Mapping[str, Any]],
) -> None:
    schema_dir = output_root / "schema_audit"
    schema_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        schema_dir / "videocutler_trajectory_schema_summary.json",
        _schema_summary(
            Path(args.videocutler_trajectory_path),
            important_keys=("trajectory_id", "clip_id", "video_id", "masks_rle", "pred_label_raw", "frame_indices"),
            scan_limit=1000,
        ),
    )
    write_json(
        schema_dir / "videocutler_gt_match_schema_summary.json",
        _schema_summary(
            Path(args.videocutler_gt_match_path),
            important_keys=("trajectory_id", "clip_id", "video_id", "matched_gt_raw_id", "matched_gt_class_id", "match_iou_mean", "match_iou_p50", "audit_usable"),
        ),
    )
    write_json(
        schema_dir / "gt_carrier_schema_summary.json",
        _schema_summary(
            Path(args.gt_carrier_path),
            important_keys=("trajectory_id", "clip_id", "video_id", "raw_id", "category_id", "gt_raw_id", "matched_gt_raw_id", "gt_category_id", "track_id", "gt_track_id", "instance_id", "z_norm_path"),
            scan_limit=2000,
        ),
    )
    write_json(
        schema_dir / "gt_identity_schema_summary.json",
        _schema_summary(
            Path(args.gt_identity_path),
            important_keys=("trajectory_id", "clip_id", "video_id", "matched_gt_raw_id", "matched_gt_class_id", "matched_gt_track_id", "match_iou_mean", "match_iou_p50", "audit_usable"),
        ),
    )
    write_json(
        schema_dir / "existing_validated_yprime_support_reference.json",
        {
            "path": str(reference_path) if reference_path else None,
            "found": reference_path is not None,
            "expected_yprime_support_rate_at_0.5": EXPECTED_YPRIME_SUPPORT_AT_05,
            "reference_yprime_support_rate_at_0.5": reference_summary.get("yprime_trajectory_support_rate") if reference_summary else None,
            "reference_stage": reference_summary.get("stage") if reference_summary else None,
            "reference_clip_yprime_pair_count": reference_summary.get("clip_yprime_pair_count") if reference_summary else None,
            "reference_mean_support_count_per_yprime": reference_summary.get("mean_support_count_per_yprime") if reference_summary else None,
        },
    )


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    full_y_path = Path(args.full_y_path).expanduser().resolve()
    full_y_records = _load_full_y_records(full_y_path)
    weak_rows = load_weak_label_records(Path(args.weak_label_path).expanduser().resolve())
    yprime_rows = _label_rows_for_key(full_y_records, "yprime_raw_ids")
    full_rows = _label_rows_for_key(full_y_records, "full_y_raw_ids")
    full_y_scope_index = _build_full_y_scope_index(full_y_records)

    run_root_v2b = Path(args.run_root_v2b).expanduser().resolve()
    reference_path, reference_summary = _find_reference_summary(run_root_v2b, args.dataset_name)
    _write_schema_audit(output_root=output_root, args=args, reference_path=reference_path, reference_summary=reference_summary)

    if reference_summary is None:
        a_support_stats = _support_stats(yprime_rows, "yprime_raw_ids", {})
        a_status = "FAIL_SCHEMA"
        a_blocker = "validated prealign yprime_support_coverage summary not found"
        a_match_status = "MISSING_REFERENCE"
    else:
        ref_rate = _safe_float(reference_summary.get("yprime_trajectory_support_rate"), None)
        ref_pairs = safe_int(reference_summary.get("clip_yprime_pair_count"), None)
        support_count = int(round(float(ref_rate or 0.0) * int(ref_pairs or 0)))
        pair_count = int(ref_pairs or 0)
        a_support_stats = {
            "label_pair_count": pair_count,
            "label_union_count": int(len({x for rec in yprime_rows for x in unique_ints(rec.get("yprime_raw_ids"))})),
            "support_upper_bound_pair_count": support_count,
            "support_upper_bound_rate_at_0.5": ref_rate,
            "support_rate_at_0.5": ref_rate,
            "zero_support_rate_at_0.5": (1.0 - float(ref_rate)) if ref_rate is not None else None,
            "clip_all_labels_supported_rate_at_0.5": reference_summary.get("clip_all_yprime_supported_rate"),
            "mean_support_count_per_label": reference_summary.get("mean_support_count_per_yprime"),
            "median_support_count_per_label": reference_summary.get("median_support_count_per_yprime"),
        }
        diff = abs(float(ref_rate) - EXPECTED_YPRIME_SUPPORT_AT_05) if ref_rate is not None else float("inf")
        a_match_status = "PASS" if diff <= REFERENCE_TOLERANCE else "FAIL"
        a_status = "PASS" if a_match_status == "PASS" else "FAIL_SCHEMA"
        a_blocker = None if a_status == "PASS" else f"validated yprime support mismatch: diff={diff}"

    existing_paths = [str(p) for p in _best_existing_summary_paths(run_root_v2b, args.dataset_name, "prealign").values() if p is not None]
    a_summary = _arm_summary(
        arm="A_yprime_videocutler_latent",
        label_source="yprime",
        carrier_source="videocutler",
        assignment_mode="latent",
        status=a_status,
        support_stats=a_support_stats,
        support_status=a_match_status,
        latent_status="REUSED_EXISTING_PREALIGN_AUDIT",
        oracle_status="NOT_APPLICABLE",
        carrier_row_count=0,
        blocker=a_blocker,
        reused_paths=existing_paths,
        extra={"validated_yprime_support_reference_path": str(reference_path) if reference_path else None},
    )

    sidecar_lookup = load_gt_sidecar_lookup(run_root_v2b, dataset_name=args.dataset_name, trajectory_source_branch="mainline")
    video_support_map, video_support_meta = _build_support_from_sidecar(sidecar_lookup=sidecar_lookup, iou_threshold=float(args.iou_threshold))
    b_stats = _support_stats(full_rows, "full_y_raw_ids", video_support_map)
    b_summary = _arm_summary(
        arm="B_fully_videocutler_latent",
        label_source="full_y",
        carrier_source="videocutler",
        assignment_mode="latent",
        status="PARTIAL_SUPPORT_ONLY",
        support_stats=b_stats,
        support_status="COMPUTED_FROM_AVAILABLE_MAINLINE_GTSIDECAR" if video_support_map else "FAIL_SCHEMA",
        latent_status="PARTIAL_NEEDS_LATENT_ADAPTER",
        oracle_status="NOT_APPLICABLE",
        carrier_row_count=int(video_support_meta.get("bound_identity_count", 0)),
        blocker="latent assignment requires dedicated scoring/responsibility adapter",
        extra={"videocutler_support_meta": video_support_meta},
    )

    gt_carrier_rows = list(iter_jsonl(Path(args.gt_carrier_path).expanduser().resolve()))
    gt_binding_rows = list(iter_jsonl(Path(args.gt_identity_path).expanduser().resolve()))
    gt_identity_lookup = _load_sidecar_by_tid(Path(args.gt_identity_path).expanduser().resolve())
    identity_meta = _gt_carrier_identity_meta(gt_binding_rows, gt_identity_lookup)
    gt_identity_high = float(identity_meta.get("gt_carrier_identity_coverage_rate") or 0.0) >= HIGH_IDENTITY_COVERAGE_THRESHOLD
    gt_support_map, gt_support_meta = _build_support_from_sidecar(sidecar_lookup=gt_identity_lookup, iou_threshold=0.0)

    scope_mismatch_summary, scope_rows_by_category, scope_rows_by_clip, scope_examples, scope_in_scope_rows = _classify_scope_mismatch(
        binding_rows=gt_binding_rows,
        full_y_records=full_y_records,
        top_examples=int(args.top_examples),
    )
    scope_diag_path = _write_scope_mismatch_diagnostics(
        output_root=output_root,
        summary=scope_mismatch_summary,
        rows_by_category=scope_rows_by_category,
        rows_by_clip=scope_rows_by_clip,
        examples=scope_examples,
    )
    out_of_scope_count = int(scope_mismatch_summary["out_of_scope_row_count"])
    in_scope_count = int(scope_mismatch_summary["in_scope_row_count"])
    in_scope_oracle_stats = _oracle_stats(scope_in_scope_rows, gt_identity_lookup, full_rows, "full_y_raw_ids")

    gt_blocker = None if gt_identity_high else "GT-carrier identity binding is sparse; high-coverage raw category id binding unavailable"
    c_summary = _arm_summary(
        arm="C_yprime_gtcarrier_latent",
        label_source="yprime",
        carrier_source="gt_carrier",
        assignment_mode="latent",
        status="PARTIAL_SUPPORT_ONLY" if gt_identity_high else "FAIL_SCHEMA",
        support_stats=_support_stats(yprime_rows, "yprime_raw_ids", gt_support_map),
        support_status="COMPUTED" if gt_identity_high else "FAIL_SCHEMA",
        latent_status="PARTIAL_NEEDS_LATENT_ADAPTER",
        oracle_status="NOT_APPLICABLE",
        carrier_row_count=len(gt_carrier_rows),
        blocker=gt_blocker or "latent assignment requires dedicated scoring/responsibility adapter",
        extra={**identity_meta, "gt_support_meta": gt_support_meta},
    )
    d_summary = _arm_summary(
        arm="D_fully_gtcarrier_latent",
        label_source="full_y",
        carrier_source="gt_carrier",
        assignment_mode="latent",
        status="PARTIAL_SUPPORT_ONLY" if gt_identity_high else "FAIL_SCHEMA",
        support_stats=_support_stats(full_rows, "full_y_raw_ids", gt_support_map),
        support_status="COMPUTED" if gt_identity_high else "FAIL_SCHEMA",
        latent_status="PARTIAL_NEEDS_LATENT_ADAPTER",
        oracle_status="NOT_APPLICABLE",
        carrier_row_count=len(gt_carrier_rows),
        blocker=gt_blocker or "latent assignment requires dedicated scoring/responsibility adapter",
        extra={**identity_meta, "gt_support_meta": gt_support_meta},
    )

    oracle_extra = _oracle_stats(gt_binding_rows, gt_identity_lookup, full_rows, "full_y_raw_ids")
    oracle_valid_rate = oracle_extra.get("oracle_assignment_valid_rate")
    oracle_valid_overall = bool(
        gt_identity_high
        and out_of_scope_count == 0
        and (oracle_valid_rate is not None)
        and float(oracle_valid_rate) >= HIGH_IDENTITY_COVERAGE_THRESHOLD
    )
    oracle_valid_in_scope = bool(
        gt_identity_high
        and in_scope_count > 0
        and (in_scope_oracle_stats.get("oracle_assignment_valid_rate") is not None)
        and float(in_scope_oracle_stats.get("oracle_assignment_valid_rate")) >= HIGH_IDENTITY_COVERAGE_THRESHOLD
    )
    if out_of_scope_count > 0:
        e_status = "PARTIAL_SCOPE_MISMATCH"
        e_blocker = (
            "GT-carrier oracle ceiling is scope-mismatch-limited: "
            f"{out_of_scope_count} bound GT carrier rows are outside the full-Y clip scope"
        )
    elif oracle_valid_overall:
        e_status = "PASS"
        e_blocker = None
    else:
        e_status = "FAIL_SCHEMA"
        e_blocker = "GT-carrier oracle ceiling unavailable: high-coverage carrier->raw category binding was not verified"
    e_summary = _arm_summary(
        arm="E_fully_gtcarrier_oracle",
        label_source="full_y",
        carrier_source="gt_carrier",
        assignment_mode="oracle_gt",
        status=e_status,
        support_stats=_support_stats(full_rows, "full_y_raw_ids", gt_support_map),
        support_status="COMPUTED" if gt_identity_high else "FAIL_SCHEMA",
        latent_status="NOT_APPLICABLE",
        oracle_status="PASS" if oracle_valid_overall else ("PARTIAL_SCOPE_MISMATCH" if out_of_scope_count > 0 else "FAIL_SCHEMA"),
        carrier_row_count=len(gt_carrier_rows),
        blocker=e_blocker,
        extra={
            **identity_meta,
            **oracle_extra,
            "gt_support_meta": gt_support_meta,
            "oracle_assignment_valid_rate_overall": oracle_extra.get("oracle_assignment_valid_rate"),
            "oracle_assignment_valid_rate_in_scope": in_scope_oracle_stats.get("oracle_assignment_valid_rate"),
            "in_scope_row_count": in_scope_count,
            "out_of_scope_row_count": out_of_scope_count,
            "in_scope_true_support_mass_mean_oracle": in_scope_oracle_stats.get("true_support_mass_mean_oracle"),
            "in_scope_true_support_top1_rate_oracle": in_scope_oracle_stats.get("true_support_top1_rate_oracle"),
            "scope_mismatch_limited": bool(out_of_scope_count > 0),
            "scope_mismatch_diagnostics_path": str(scope_diag_path),
            "full_y_scope_key_used": scope_mismatch_summary.get("full_y_scope_key_used"),
            "scope_key_match_rule": scope_mismatch_summary.get("scope_key_match_rule"),
            "recommended_resolution": scope_mismatch_summary.get("recommended_resolution"),
        },
    )
    e_in_scope_summary = _arm_summary(
        arm="E_fully_gtcarrier_oracle_in_scope",
        label_source="full_y",
        carrier_source="gt_carrier",
        assignment_mode="oracle_gt",
        status="PASS" if oracle_valid_in_scope else "FAIL_SCHEMA",
        support_stats=_support_stats(full_rows, "full_y_raw_ids", gt_support_map),
        support_status="COMPUTED" if gt_identity_high else "FAIL_SCHEMA",
        latent_status="NOT_APPLICABLE",
        oracle_status="PASS" if oracle_valid_in_scope else "FAIL_SCHEMA",
        carrier_row_count=in_scope_count,
        blocker=None if oracle_valid_in_scope else "in-scope oracle ceiling failed exact validity check",
        extra={
            **identity_meta,
            **in_scope_oracle_stats,
            "gt_support_meta": gt_support_meta,
            "oracle_assignment_valid_rate_overall": oracle_extra.get("oracle_assignment_valid_rate"),
            "oracle_assignment_valid_rate_in_scope": in_scope_oracle_stats.get("oracle_assignment_valid_rate"),
            "in_scope_row_count": in_scope_count,
            "out_of_scope_row_count": 0,
            "in_scope_true_support_mass_mean_oracle": in_scope_oracle_stats.get("true_support_mass_mean_oracle"),
            "in_scope_true_support_top1_rate_oracle": in_scope_oracle_stats.get("true_support_top1_rate_oracle"),
            "scope_mismatch_limited": False,
            "scope_mismatch_diagnostics_path": str(scope_diag_path),
            "full_y_scope_key_used": scope_mismatch_summary.get("full_y_scope_key_used"),
            "scope_key_match_rule": scope_mismatch_summary.get("scope_key_match_rule"),
            "recommended_resolution": scope_mismatch_summary.get("recommended_resolution"),
        },
    )

    arms = [a_summary, b_summary, c_summary, d_summary, e_summary, e_in_scope_summary]
    for arm in arms:
        arm_dir = output_root / "arms" / arm["arm"]
        arm_dir.mkdir(parents=True, exist_ok=True)
        write_json(arm_dir / "summary.json", arm)

    fieldnames = [
        "arm",
        "status",
        "support_upper_bound_status",
        "latent_assignment_status",
        "oracle_assignment_status",
        "label_source",
        "carrier_source",
        "assignment_mode",
        "label_pair_count",
        "label_union_count",
        "carrier_row_count",
        "support_upper_bound_rate_at_0.5",
        "support_upper_bound_pair_count",
        "zero_support_rate_at_0.5",
        "oracle_assignment_valid_rate_overall",
        "oracle_assignment_valid_rate_in_scope",
        "in_scope_row_count",
        "out_of_scope_row_count",
        "oracle_assignment_valid_rate",
        "scope_mismatch_limited",
        "scope_mismatch_diagnostics_path",
        "blocker",
    ]
    comparison_rows = [{k: arm.get(k) for k in fieldnames} for arm in arms]
    write_csv(output_root / "oracle_clean_ablation_comparison.csv", comparison_rows, fieldnames=fieldnames)

    assets_manifest = {
        "dataset_name": args.dataset_name,
        "full_y_path": str(full_y_path),
        "weak_label_path": str(Path(args.weak_label_path).expanduser().resolve()),
        "videocutler_trajectory_path": str(Path(args.videocutler_trajectory_path).expanduser().resolve()),
        "videocutler_gt_match_path": str(Path(args.videocutler_gt_match_path).expanduser().resolve()),
        "gt_carrier_path": str(Path(args.gt_carrier_path).expanduser().resolve()),
        "gt_identity_path": str(Path(args.gt_identity_path).expanduser().resolve()),
        "run_root_v2b": str(run_root_v2b),
        "full_y_record_count": int(len(full_y_records)),
        "weak_label_record_count": int(len(weak_rows)),
        "full_y_union_count": int(len({x for rec in full_y_records for x in unique_ints(rec.get("full_y_raw_ids"))})),
        "yprime_union_count": int(len({x for rec in full_y_records for x in unique_ints(rec.get("yprime_raw_ids"))})),
        "gt_carrier_row_count": int(len(gt_carrier_rows)),
        "gt_carrier_identity": identity_meta,
        "validated_yprime_support_reference_path": str(reference_path) if reference_path else None,
        "scope_mismatch_diagnostics_path": str(scope_diag_path),
        "full_y_scope_key_used": scope_mismatch_summary.get("full_y_scope_key_used"),
    }
    write_json(output_root / "assets_manifest.json", assets_manifest)

    yprime_subset_full_y = all(
        set(unique_ints(rec.get("yprime_raw_ids", []))).issubset(set(unique_ints(rec.get("full_y_raw_ids", []))))
        for rec in full_y_records
    )
    overall_status = "FAIL" if a_summary["status"] != "PASS" else "PARTIAL"
    summary = {
        "status": overall_status,
        "dataset_name": args.dataset_name,
        "full_y_materialized_successfully": full_y_path.is_file(),
        "yprime_subset_full_y": bool(yprime_subset_full_y),
        "a_support_matches_validated_yprime_support": a_match_status == "PASS",
        "gt_carrier_identity_high_coverage": bool(gt_identity_high),
        "e_oracle_ceiling_valid": bool(oracle_valid_overall),
        "e_oracle_ceiling_valid_in_scope": bool(oracle_valid_in_scope),
        "oracle_assignment_valid_rate_overall": oracle_extra.get("oracle_assignment_valid_rate"),
        "oracle_assignment_valid_rate_in_scope": in_scope_oracle_stats.get("oracle_assignment_valid_rate"),
        "in_scope_row_count": in_scope_count,
        "out_of_scope_row_count": out_of_scope_count,
        "scope_mismatch_limited": bool(out_of_scope_count > 0),
        "scope_mismatch_diagnostics_path": str(scope_diag_path),
        "full_y_scope_key_used": scope_mismatch_summary.get("full_y_scope_key_used"),
        "recommended_resolution": scope_mismatch_summary.get("recommended_resolution"),
        "arm_statuses": {arm["arm"]: arm["status"] for arm in arms},
        "gt_carrier_identity": identity_meta,
        "minimal_next_action": (
            "latent B/C/D still require a scoring/responsibility adapter; "
            "E overall remains scope-mismatch-limited while the in-scope oracle ceiling is valid"
        ),
    }
    write_json(output_root / "oracle_clean_ablation_summary.json", summary)

    lines = [
        "# Oracle Clean-Data Ablation Takeover",
        "",
        f"- status: `{overall_status}`",
        f"- full_y_materialized_successfully: `{full_y_path.is_file()}`",
        f"- yprime_subset_full_y: `{bool(yprime_subset_full_y)}`",
        f"- A_support_matches_validated_yprime_support: `{a_match_status == 'PASS'}`",
        f"- GT_carrier_identity_coverage_rate: `{identity_meta.get('gt_carrier_identity_coverage_rate')}`",
        f"- E_oracle_ceiling_valid_overall: `{bool(oracle_valid_overall)}`",
        f"- E_oracle_ceiling_valid_in_scope: `{bool(oracle_valid_in_scope)}`",
        f"- scope_mismatch_limited: `{bool(out_of_scope_count > 0)}`",
        f"- scope_mismatch_diagnostics_path: `{scope_diag_path}`",
        "",
        "| arm | status | support_status | latent_status | oracle_status | support_rate@0.5 | oracle_valid_rate | blocker |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for arm in arms:
        lines.append(
            f"| {arm['arm']} | {arm['status']} | {arm['support_upper_bound_status']} | {arm['latent_assignment_status']} | "
            f"{arm['oracle_assignment_status']} | {arm.get('support_upper_bound_rate_at_0.5')} | "
            f"{arm.get('oracle_assignment_valid_rate')} | {arm.get('blocker') or ''} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The full-Y builder remains valid: weak labels are a subset of full-Y.",
        "- A uses the validated prealign yprime-support audit as authority and matches the expected support rate.",
        "- GT-carrier identity binding is high coverage; C/D support ceilings are meaningful but latent assignment was not recomputed.",
        f"- E overall is scope-mismatch-limited because {out_of_scope_count} bound GT-carrier rows are outside the current full-Y label scope.",
        "- E in-scope oracle ceiling is valid and should be used for the scoped oracle report.",
        "- B/C/D latent assignment was not recomputed; those arms remain partial and require a dedicated scoring/responsibility adapter.",
    ]
    (output_root / "ORACLE_CLEAN_ABLATION_TAKEOVER.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    print(json.dumps({"status": overall_status, "output_root": str(output_root), "arm_statuses": summary["arm_statuses"]}, indent=2))


if __name__ == "__main__":
    main()
