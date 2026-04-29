#!/usr/bin/env python3
"""
Read-only audit for official-base classes that are absent from the current GT context audit.

Purpose
-------
For base classes with clip_count == 0 in gt_context_identifiability/per_class_context_identifiability.csv,
identify whether they are truly absent from the train annotation, filtered out by the audit universe,
missing due to id/name mapping, or present only in an optional validation/other annotation file.

This script is intentionally standard-library only and does not modify training / inference artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _norm_name(x: Any) -> str:
    s = str(x or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("-", "_")
    s = re.sub(r"[^a-z0-9_()]+", "", s)
    return s


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(str(x)))
        except Exception:
            return None


def _as_str_id(x: Any) -> str:
    i = _as_int(x)
    return str(i) if i is not None else str(x)


def _maybe_list_of_ids(v: Any) -> Optional[List[int]]:
    if isinstance(v, list):
        out: List[int] = []
        ok = True
        for item in v:
            if isinstance(item, dict):
                val = item.get("raw_id", item.get("id", item.get("category_id")))
            else:
                val = item
            ii = _as_int(val)
            if ii is None:
                ok = False
                break
            out.append(ii)
        return out if ok else None
    return None


def _extract_split_ids(obj: Any, split_name: str) -> List[int]:
    """Robustly extract base/novel raw ids from common split-json layouts."""
    keys = {
        "base": [
            "base", "base_ids", "base_raw_ids", "base_category_ids", "base_classes",
            "official_base", "base_raw_id_list", "base_categories",
        ],
        "novel": [
            "novel", "novel_ids", "novel_raw_ids", "novel_category_ids", "novel_classes",
            "official_novel", "novel_raw_id_list", "novel_categories",
        ],
    }[split_name]

    found: List[int] = []

    def walk(x: Any, path: Tuple[str, ...] = ()) -> None:
        nonlocal found
        if found:
            return
        if isinstance(x, dict):
            for k in keys:
                if k in x:
                    ids = _maybe_list_of_ids(x[k])
                    if ids is not None:
                        found = ids
                        return
            # Some files store {"splits": {"base": [...]}}
            for k, v in x.items():
                if str(k).lower() == split_name:
                    ids = _maybe_list_of_ids(v)
                    if ids is not None:
                        found = ids
                        return
            for v in x.values():
                walk(v, path)
                if found:
                    return
        elif isinstance(x, list):
            # If list of records with split field.
            records = [e for e in x if isinstance(e, dict)]
            if records and any(str(r.get("split", "")).lower() == split_name for r in records):
                vals: List[int] = []
                for r in records:
                    if str(r.get("split", "")).lower() == split_name:
                        val = r.get("raw_id", r.get("id", r.get("category_id")))
                        ii = _as_int(val)
                        if ii is not None:
                            vals.append(ii)
                if vals:
                    found = vals
                    return

    walk(obj)
    return sorted(set(found))


def _category_maps(ann: Dict[str, Any]) -> Tuple[Dict[int, str], Dict[str, List[int]]]:
    id_to_name: Dict[int, str] = {}
    name_to_ids: Dict[str, List[int]] = defaultdict(list)
    cats = ann.get("categories") or ann.get("classes") or []
    if isinstance(cats, dict):
        # sometimes id -> name
        iterable = []
        for k, v in cats.items():
            if isinstance(v, dict):
                rec = dict(v)
                rec.setdefault("id", k)
            else:
                rec = {"id": k, "name": v}
            iterable.append(rec)
    else:
        iterable = cats if isinstance(cats, list) else []
    for c in iterable:
        if not isinstance(c, dict):
            continue
        cid = _as_int(c.get("id", c.get("raw_id", c.get("category_id"))))
        if cid is None:
            continue
        name = str(c.get("name", c.get("class_name", cid)))
        id_to_name[cid] = name
        name_to_ids[_norm_name(name)].append(cid)
    return id_to_name, dict(name_to_ids)


def _ann_clip_id(a: Dict[str, Any]) -> str:
    for k in ("clip_id", "video_id", "image_id", "sequence_id", "id"):
        if k in a and a[k] is not None:
            return str(a[k])
    return "__unknown_clip__"


def _ann_video_id(a: Dict[str, Any]) -> str:
    for k in ("video_id", "sequence_id", "clip_id", "image_id", "id"):
        if k in a and a[k] is not None:
            return str(a[k])
    return "__unknown_video__"


def _has_segmentation(a: Dict[str, Any]) -> bool:
    for k in ("segmentations", "segmentation", "segments_info", "masks", "mask", "rle"):
        if k in a:
            v = a.get(k)
            if v is None:
                continue
            if isinstance(v, list):
                return len(v) > 0 and any(x not in (None, [], {}, "") for x in v)
            if isinstance(v, dict):
                return bool(v)
            if isinstance(v, str):
                return bool(v.strip())
            return True
    return False


def _ignored(a: Dict[str, Any]) -> bool:
    for k in ("ignore", "ignored", "iscrowd", "is_crowd"):
        if a.get(k) in (1, True, "1", "true", "True"):
            return True
    return False


def _count_annotation_presence(ann: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    stats: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
        "raw_annotation_count": 0,
        "valid_annotation_count": 0,
        "ignored_annotation_count": 0,
        "with_segmentation_count": 0,
        "clip_ids": set(),
        "video_ids": set(),
    })
    anns = ann.get("annotations") or ann.get("instances") or []
    if isinstance(anns, dict):
        iterable = anns.values()
    else:
        iterable = anns if isinstance(anns, list) else []
    for a in iterable:
        if not isinstance(a, dict):
            continue
        cid = _as_int(a.get("category_id", a.get("raw_id", a.get("class_id"))))
        if cid is None:
            continue
        st = stats[cid]
        st["raw_annotation_count"] += 1
        if _ignored(a):
            st["ignored_annotation_count"] += 1
        else:
            st["valid_annotation_count"] += 1
        if _has_segmentation(a):
            st["with_segmentation_count"] += 1
        st["clip_ids"].add(_ann_clip_id(a))
        st["video_ids"].add(_ann_video_id(a))
    # convert sets later where needed
    return stats


def _load_per_class_absent(per_class_csv: Path) -> Tuple[Set[int], Dict[int, Dict[str, str]]]:
    absent: Set[int] = set()
    rows_by_id: Dict[int, Dict[str, str]] = {}
    if not per_class_csv.exists():
        return absent, rows_by_id
    with per_class_csv.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("context_label") != "base_only":
                continue
            if r.get("split_type") != "base":
                continue
            rid = _as_int(r.get("raw_id"))
            if rid is None:
                continue
            rows_by_id[rid] = r
            clip_count = _as_int(r.get("clip_count")) or 0
            if clip_count == 0:
                absent.add(rid)
    return absent, rows_by_id


def _bucket_for_class(
    rid: int,
    train_cat_name: Optional[str],
    train_name_only_ids: List[int],
    train_stats: Dict[str, Any],
    per_class_clip_count: Optional[int],
    val_stats: Optional[Dict[str, Any]],
) -> str:
    raw = int(train_stats.get("raw_annotation_count", 0))
    valid = int(train_stats.get("valid_annotation_count", 0))
    seg = int(train_stats.get("with_segmentation_count", 0))
    val_raw = int((val_stats or {}).get("raw_annotation_count", 0))
    if train_cat_name is None and train_name_only_ids:
        return "id_mapping_mismatch_name_only_match"
    if train_cat_name is None:
        if val_raw > 0:
            return "not_in_train_categories_but_present_in_other_annotation"
        return "taxonomy_base_id_missing_from_train_categories"
    if raw == 0:
        if val_raw > 0:
            return "present_only_in_other_annotation"
        return "base_but_absent_from_train_annotation"
    if per_class_clip_count == 0 and raw > 0:
        if valid == 0:
            return "raw_present_but_only_ignored_or_invalid"
        if seg == 0:
            return "raw_present_but_no_segmentation_surface"
        return "present_raw_but_filtered_out_or_audit_universe_mismatch"
    return "present_in_train_annotation"


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit origin of official-base classes absent from GT context identifiability.")
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--runtime_output_root", default=".")
    ap.add_argument("--dataset_name", default="lvvis_train_base")
    ap.add_argument("--annotation_json", required=True)
    ap.add_argument("--split_json", default="package/reference/lvvis_official_base_novel_split.json")
    ap.add_argument("--per_class_csv", default=None, help="Existing gt_context_identifiability per_class_context_identifiability.csv")
    ap.add_argument("--other_annotation_json", default=None, help="Optional val/other annotation to test present-only-in-other-split")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--top_examples", type=int, default=128)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    split_path = Path(args.split_json)
    ann_path = Path(args.annotation_json)
    if args.per_class_csv:
        per_class_path = Path(args.per_class_csv)
    else:
        per_class_path = run_root / "analysis" / "gt_context_identifiability" / args.dataset_name / "per_class_context_identifiability.csv"
    out_dir = Path(args.output_dir) if args.output_dir else run_root / "analysis" / "base_absent_class_origin" / args.dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    split_obj = _read_json(split_path)
    base_ids = _extract_split_ids(split_obj, "base")
    novel_ids = _extract_split_ids(split_obj, "novel")
    base_set = set(base_ids)

    ann = _read_json(ann_path)
    train_id_to_name, train_name_to_ids = _category_maps(ann)
    train_presence = _count_annotation_presence(ann)

    other_id_to_name: Dict[int, str] = {}
    other_presence: Dict[int, Dict[str, Any]] = {}
    if args.other_annotation_json:
        other = _read_json(Path(args.other_annotation_json))
        other_id_to_name, _ = _category_maps(other)
        other_presence = _count_annotation_presence(other)

    absent_from_per_class, per_rows = _load_per_class_absent(per_class_path)
    if absent_from_per_class:
        absent_ids = sorted(absent_from_per_class)
        absent_source = "per_class_context_identifiability_clip_count_0"
    else:
        # fallback: official base not appearing in train annotation at all
        absent_ids = sorted([rid for rid in base_ids if int(train_presence.get(rid, {}).get("raw_annotation_count", 0)) == 0])
        absent_source = "fallback_base_ids_not_in_train_annotations"

    rows: List[Dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter()
    examples_by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for rid in absent_ids:
        train_cat_name = train_id_to_name.get(rid)
        name_norm = _norm_name(train_cat_name) if train_cat_name else ""
        # If id not found, try split/name from per_class row.
        pc = per_rows.get(rid, {})
        class_name = pc.get("class_name") or train_cat_name or other_id_to_name.get(rid) or f"raw_{rid}"
        name_only_ids = train_name_to_ids.get(_norm_name(class_name), []) if train_cat_name is None else []
        st = train_presence.get(rid, {})
        oth = other_presence.get(rid, {}) if other_presence else None
        per_clip = _as_int(pc.get("clip_count")) if pc else None
        bucket = _bucket_for_class(rid, train_cat_name, name_only_ids, st, per_clip, oth)
        bucket_counts[bucket] += 1
        clip_ids = st.get("clip_ids", set()) if st else set()
        video_ids = st.get("video_ids", set()) if st else set()
        other_clip_ids = (oth or {}).get("clip_ids", set()) if oth else set()
        row = {
            "raw_id": rid,
            "class_name": class_name,
            "bucket": bucket,
            "in_official_base_split": True,
            "in_official_novel_split": rid in set(novel_ids),
            "in_train_categories_by_id": train_cat_name is not None,
            "train_category_name_by_id": train_cat_name or "",
            "name_only_train_category_ids": ";".join(map(str, name_only_ids)),
            "per_class_clip_count": per_clip if per_clip is not None else "",
            "per_class_instance_count": pc.get("instance_count", "") if pc else "",
            "train_raw_annotation_count": int(st.get("raw_annotation_count", 0)) if st else 0,
            "train_valid_annotation_count": int(st.get("valid_annotation_count", 0)) if st else 0,
            "train_ignored_annotation_count": int(st.get("ignored_annotation_count", 0)) if st else 0,
            "train_with_segmentation_count": int(st.get("with_segmentation_count", 0)) if st else 0,
            "train_clip_count_raw": len(clip_ids),
            "train_video_count_raw": len(video_ids),
            "other_raw_annotation_count": int((oth or {}).get("raw_annotation_count", 0)) if oth else 0,
            "other_clip_count_raw": len(other_clip_ids) if oth else 0,
            "mapping_status": (
                "exact_id_found" if train_cat_name is not None else
                "name_only_found" if name_only_ids else
                "id_missing"
            ),
            "evidence_summary": "",
        }
        row["evidence_summary"] = (
            f"bucket={bucket}; train_raw={row['train_raw_annotation_count']}; "
            f"per_class_clip={row['per_class_clip_count']}; mapping={row['mapping_status']}"
        )
        rows.append(row)
        if len(examples_by_bucket[bucket]) < args.top_examples:
            examples_by_bucket[bucket].append(row)

    # Write CSV.
    csv_path = out_dir / "absent_class_origin.csv"
    fieldnames = [
        "raw_id", "class_name", "bucket", "in_official_base_split", "in_official_novel_split",
        "in_train_categories_by_id", "train_category_name_by_id", "name_only_train_category_ids",
        "per_class_clip_count", "per_class_instance_count", "train_raw_annotation_count",
        "train_valid_annotation_count", "train_ignored_annotation_count", "train_with_segmentation_count",
        "train_clip_count_raw", "train_video_count_raw", "other_raw_annotation_count", "other_clip_count_raw",
        "mapping_status", "evidence_summary",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    bucket_path = out_dir / "bucket_summary.csv"
    with bucket_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bucket", "count", "rate"])
        w.writeheader()
        total = len(rows)
        for b, c in bucket_counts.most_common():
            w.writerow({"bucket": b, "count": c, "rate": (c / total if total else 0.0)})

    examples_path = out_dir / "absent_class_origin_examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as f:
        for b, exs in examples_by_bucket.items():
            for r in exs:
                f.write(json.dumps({"bucket": b, **r}, ensure_ascii=False) + "\n")

    summary = {
        "status": "PASS",
        "dataset_name": args.dataset_name,
        "official_split_path": str(split_path),
        "base_count": len(base_ids),
        "novel_count": len(novel_ids),
        "annotation_json": str(ann_path),
        "per_class_csv": str(per_class_path),
        "absent_source": absent_source,
        "absent_class_count": len(rows),
        "bucket_counts": dict(bucket_counts),
        "train_category_count": len(train_id_to_name),
        "train_annotation_class_count": len(train_presence),
        "other_annotation_json": args.other_annotation_json or "",
        "outputs": {
            "summary_json": str(out_dir / "summary.json"),
            "absent_class_origin_csv": str(csv_path),
            "bucket_summary_csv": str(bucket_path),
            "examples_jsonl": str(examples_path),
            "takeover_md": str(out_dir / "BASE_ABSENT_CLASS_ORIGIN_TAKEOVER.md"),
        },
        "interpretation": {
            "purpose": "Classify why official-base classes have clip_count=0 in the GT context identifiability audit.",
            "valid_scientific_claim": "This audit identifies whether absent classes are absent from train annotations, filtered/mismatched, or only present in optional other annotations; it does not assess scorer quality.",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown takeover.
    lines = [
        "# Base Absent Class Origin Audit",
        "",
        f"status: `{summary['status']}`",
        f"dataset: `{args.dataset_name}`",
        f"base_count: `{len(base_ids)}`",
        f"novel_count: `{len(novel_ids)}`",
        f"absent_class_count: `{len(rows)}`",
        "",
        "## Bucket summary",
        "",
        "| bucket | count | rate |",
        "|---|---:|---:|",
    ]
    total = len(rows)
    for b, c in bucket_counts.most_common():
        lines.append(f"| `{b}` | {c} | {c/total if total else 0.0:.6f} |")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `base_but_absent_from_train_annotation`: official-base taxonomy class exists, but no raw train annotation instance was found.",
        "- `present_raw_but_filtered_out_or_audit_universe_mismatch`: raw train annotations exist although the prior context audit marked clip_count=0; inspect filtering / id universe.",
        "- `id_mapping_mismatch_name_only_match`: split raw id did not match train category id, but the class name matched another id.",
        "- `present_only_in_other_annotation`: absent from train but present in optional other annotation, if provided.",
        "",
        "This is a read-only audit. It does not modify training, inference, checkpoints, or predictions.",
    ])
    (out_dir / "BASE_ABSENT_CLASS_ORIGIN_TAKEOVER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
