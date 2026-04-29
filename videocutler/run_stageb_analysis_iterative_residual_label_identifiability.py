#!/usr/bin/env python3
"""
Read-only iterative residual label identifiability audit for WS-OVVIS / LV-VIS.

This script answers a label-only theoretical question:
Given train-observed official-base classes and an initial set of already identifiable
anchor classes, how many additional classes can be uniquely identified by iteratively
removing known classes from clip-level GT label contexts?

It does not read trajectory masks/features, train models, run inference, or modify
existing artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_PERSON_RAW_ID = 773
DEFAULT_MIN_AUDITABLE_CLIPS = 3


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x)
        s = str(x).strip()
        if not s:
            return None
        return int(float(s)) if "." in s else int(s)
    except Exception:
        return None


def _as_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def _safe_json_loads(s: Any, default: Any) -> Any:
    if s in (None, ""):
        return default
    try:
        return json.loads(str(s))
    except Exception:
        return default


def _json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path, max_bytes: Optional[int] = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(f.read(max_bytes))
    return h.hexdigest()


def _canonical_key(vals: Iterable[int]) -> str:
    return ";".join(str(v) for v in sorted(vals))


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _parse_split(split_json: Path) -> Tuple[Set[int], Set[int], Dict[str, Any]]:
    obj = _json_load(split_json)
    base = obj.get("base_raw_ids") or obj.get("base_ids") or obj.get("base") or []
    novel = obj.get("novel_raw_ids") or obj.get("novel_ids") or obj.get("novel") or []
    base_set = {int(x) for x in base}
    novel_set = {int(x) for x in novel}
    meta = {
        "split_path": str(split_json),
        "base_count": len(base_set),
        "novel_count": len(novel_set),
        "expected_base_count": obj.get("base_category_count") or obj.get("base_count"),
        "expected_novel_count": obj.get("novel_category_count") or obj.get("novel_count"),
        "authority_type": obj.get("authority_type"),
        "source": obj.get("source"),
        "sha256_prefix": _sha256_file(split_json, max_bytes=1 << 20)[:16],
    }
    return base_set, novel_set, meta


def _load_annotation_contexts(annotation_json: Path) -> Tuple[Dict[int, Set[int]], Dict[int, str], Counter, Counter, Dict[str, Any]]:
    obj = _json_load(annotation_json)
    cat_names: Dict[int, str] = {}
    for c in obj.get("categories") or []:
        cid = _as_int(c.get("id") or c.get("category_id") or c.get("raw_id"))
        if cid is not None:
            cat_names[cid] = str(c.get("name") or cid)

    contexts: Dict[int, Set[int]] = defaultdict(set)
    instance_count: Counter = Counter()
    clip_pair_count: Counter = Counter()
    ann_count = 0
    ann_no_class = 0
    ann_no_clip = 0

    for ann in obj.get("annotations") or []:
        ann_count += 1
        cid = _as_int(ann.get("category_id") or ann.get("raw_id") or ann.get("class_id"))
        # In LV-VIS train_instances.json, video_id is the relevant unit for clip/video-level label context.
        vid = _as_int(ann.get("video_id") or ann.get("clip_id") or ann.get("image_id"))
        if cid is None:
            ann_no_class += 1
            continue
        if vid is None:
            ann_no_clip += 1
            continue
        contexts[vid].add(cid)
        instance_count[cid] += 1

    for _vid, classes in contexts.items():
        for cid in classes:
            clip_pair_count[cid] += 1

    meta = {
        "annotation_path": str(annotation_json),
        "annotation_sha256_prefix": _sha256_file(annotation_json, max_bytes=1 << 20)[:16],
        "annotation_count": ann_count,
        "annotation_without_category_id": ann_no_class,
        "annotation_without_clip_or_video_id": ann_no_clip,
        "clip_or_video_context_count": len(contexts),
        "category_name_count": len(cat_names),
    }
    return dict(contexts), cat_names, instance_count, clip_pair_count, meta


def _read_per_class_csv(path: Optional[Path]) -> Dict[int, Dict[str, str]]:
    if not path or not path.exists():
        return {}
    out: Dict[int, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = _as_int(row.get("raw_id") or row.get("class_id") or row.get("category_id"))
            if rid is None:
                continue
            # Prefer base_only/base rows if the file contains multiple context labels.
            if row.get("context_label") and row.get("context_label") != "base_only":
                continue
            if row.get("split_type") and row.get("split_type") != "base":
                continue
            out[rid] = row
    return out


def _compute_context_stats(
    class_id: int,
    class_to_clips: Dict[int, List[int]],
    clip_contexts: Dict[int, Set[int]],
) -> Dict[str, Any]:
    clips = class_to_clips.get(class_id, [])
    contexts = [set(clip_contexts[v]) for v in clips]
    if not contexts:
        return {
            "clip_count": 0,
            "intersection": set(),
            "intersection_size": 0,
            "is_identifiable": False,
        }
    inter = set(contexts[0])
    for ctx in contexts[1:]:
        inter &= ctx
    return {
        "clip_count": len(clips),
        "intersection": inter,
        "intersection_size": len(inter),
        "is_identifiable": inter == {class_id},
    }


def _initial_anchors_from_csv(
    per_class: Dict[int, Dict[str, str]],
    train_observed_base: Set[int],
    min_clips: int,
) -> Set[int]:
    anchors: Set[int] = set()
    for rid, row in per_class.items():
        if rid not in train_observed_base:
            continue
        cc = _as_int(row.get("clip_count") or row.get("base_clip_count")) or 0
        if cc >= min_clips and _as_bool(row.get("is_identifiable")):
            anchors.add(rid)
    return anchors


def _initial_anchors_computed(
    train_observed_base: Set[int],
    class_to_clips: Dict[int, List[int]],
    clip_contexts: Dict[int, Set[int]],
    min_clips: int,
) -> Set[int]:
    anchors: Set[int] = set()
    for rid in train_observed_base:
        st = _compute_context_stats(rid, class_to_clips, clip_contexts)
        if st["clip_count"] >= min_clips and st["is_identifiable"]:
            anchors.add(rid)
    return anchors


def _certificate_type(clip_count: int, confounders_removed: Set[int], person_id: int) -> str:
    if clip_count <= 0:
        return "absent"
    if clip_count == 1:
        base = "single_clip_residual"
    elif clip_count == 2:
        base = "two_clip_residual"
    else:
        base = "multi_clip_residual"
    if person_id in confounders_removed:
        return "person_conditioned_" + base
    return "anchor_conditioned_" + base


def _run_variant(
    variant_name: str,
    initial_known: Set[int],
    train_observed_base: Set[int],
    class_to_clips: Dict[int, List[int]],
    clip_contexts: Dict[int, Set[int]],
    cat_names: Dict[int, str],
    instance_count: Counter,
    clip_pair_count: Counter,
    person_id: int,
    max_iterations: int,
) -> Dict[str, Any]:
    known: Set[int] = set(initial_known)
    all_rows: List[Dict[str, Any]] = []
    dependency_edges: List[Dict[str, Any]] = []
    iteration_rows: List[Dict[str, Any]] = []

    # Seed initial rows.
    for rid in sorted(train_observed_base):
        st = _compute_context_stats(rid, class_to_clips, clip_contexts)
        if rid in initial_known:
            all_rows.append({
                "variant": variant_name,
                "raw_id": rid,
                "class_name": cat_names.get(rid, str(rid)),
                "clip_count": st["clip_count"],
                "instance_count": int(instance_count.get(rid, 0)),
                "initial_intersection_size": st["intersection_size"],
                "initial_intersection_raw_ids": json.dumps(sorted(st["intersection"]), ensure_ascii=False),
                "initial_intersection_names": json.dumps([cat_names.get(x, str(x)) for x in sorted(st["intersection"])], ensure_ascii=False),
                "resolved": True,
                "resolved_at_iteration": 0,
                "certificate_type": "initial_context_identifiable" if st["is_identifiable"] else "initial_known_injected",
                "residual_intersection_size": st["intersection_size"],
                "residual_intersection_raw_ids": json.dumps(sorted(st["intersection"]), ensure_ascii=False),
                "residual_intersection_names": json.dumps([cat_names.get(x, str(x)) for x in sorted(st["intersection"])], ensure_ascii=False),
                "removed_common_known_raw_ids": json.dumps([], ensure_ascii=False),
                "removed_common_known_names": json.dumps([], ensure_ascii=False),
            })

    # Iteratively resolve from previous known set only.
    resolved_this_variant: Set[int] = set(initial_known)
    for iteration in range(1, max_iterations + 1):
        newly: Dict[int, Dict[str, Any]] = {}
        snapshot_known = set(known)
        for rid in sorted(train_observed_base - snapshot_known):
            clips = class_to_clips.get(rid, [])
            if not clips:
                continue
            raw_contexts = [set(clip_contexts[v]) for v in clips]
            initial_inter = set(raw_contexts[0])
            for ctx in raw_contexts[1:]:
                initial_inter &= ctx

            residual_contexts = [ctx - snapshot_known for ctx in raw_contexts]
            residual_inter = set(residual_contexts[0])
            for ctx in residual_contexts[1:]:
                residual_inter &= ctx

            if residual_inter == {rid}:
                removed_common = initial_inter & snapshot_known
                cert = _certificate_type(len(clips), removed_common, person_id)
                newly[rid] = {
                    "variant": variant_name,
                    "raw_id": rid,
                    "class_name": cat_names.get(rid, str(rid)),
                    "clip_count": len(clips),
                    "instance_count": int(instance_count.get(rid, 0)),
                    "initial_intersection_size": len(initial_inter),
                    "initial_intersection_raw_ids": json.dumps(sorted(initial_inter), ensure_ascii=False),
                    "initial_intersection_names": json.dumps([cat_names.get(x, str(x)) for x in sorted(initial_inter)], ensure_ascii=False),
                    "resolved": True,
                    "resolved_at_iteration": iteration,
                    "certificate_type": cert,
                    "residual_intersection_size": len(residual_inter),
                    "residual_intersection_raw_ids": json.dumps(sorted(residual_inter), ensure_ascii=False),
                    "residual_intersection_names": json.dumps([cat_names.get(x, str(x)) for x in sorted(residual_inter)], ensure_ascii=False),
                    "removed_common_known_raw_ids": json.dumps(sorted(removed_common), ensure_ascii=False),
                    "removed_common_known_names": json.dumps([cat_names.get(x, str(x)) for x in sorted(removed_common)], ensure_ascii=False),
                }
                for dep in sorted(removed_common):
                    dependency_edges.append({
                        "variant": variant_name,
                        "target_raw_id": rid,
                        "target_name": cat_names.get(rid, str(rid)),
                        "removed_known_raw_id": dep,
                        "removed_known_name": cat_names.get(dep, str(dep)),
                        "resolved_at_iteration": iteration,
                    })
        if not newly:
            iteration_rows.append({
                "variant": variant_name,
                "iteration": iteration,
                "newly_resolved_count": 0,
                "cumulative_known_count": len(known),
                "stop_reason": "no_new_classes",
            })
            break
        for rid, row in newly.items():
            known.add(rid)
            resolved_this_variant.add(rid)
            all_rows.append(row)
        iteration_rows.append({
            "variant": variant_name,
            "iteration": iteration,
            "newly_resolved_count": len(newly),
            "cumulative_known_count": len(known),
            "stop_reason": "continue",
        })
    else:
        iteration_rows.append({
            "variant": variant_name,
            "iteration": max_iterations,
            "newly_resolved_count": 0,
            "cumulative_known_count": len(known),
            "stop_reason": "max_iterations_reached",
        })

    # Add unresolved rows and ambiguity groups under final known closure.
    ambiguity_groups: Dict[str, List[int]] = defaultdict(list)
    for rid in sorted(train_observed_base - known):
        clips = class_to_clips.get(rid, [])
        raw_contexts = [set(clip_contexts[v]) for v in clips]
        if raw_contexts:
            initial_inter = set(raw_contexts[0])
            for ctx in raw_contexts[1:]:
                initial_inter &= ctx
            residual_contexts = [ctx - known for ctx in raw_contexts]
            residual_inter = set(residual_contexts[0])
            for ctx in residual_contexts[1:]:
                residual_inter &= ctx
        else:
            initial_inter = set()
            residual_inter = set()
        key = _canonical_key(residual_inter)
        ambiguity_groups[key].append(rid)
        cert = "observed_but_insufficient_context" if len(clips) < DEFAULT_MIN_AUDITABLE_CLIPS else "unresolved_ambiguous"
        all_rows.append({
            "variant": variant_name,
            "raw_id": rid,
            "class_name": cat_names.get(rid, str(rid)),
            "clip_count": len(clips),
            "instance_count": int(instance_count.get(rid, 0)),
            "initial_intersection_size": len(initial_inter),
            "initial_intersection_raw_ids": json.dumps(sorted(initial_inter), ensure_ascii=False),
            "initial_intersection_names": json.dumps([cat_names.get(x, str(x)) for x in sorted(initial_inter)], ensure_ascii=False),
            "resolved": False,
            "resolved_at_iteration": "",
            "certificate_type": cert,
            "residual_intersection_size": len(residual_inter),
            "residual_intersection_raw_ids": json.dumps(sorted(residual_inter), ensure_ascii=False),
            "residual_intersection_names": json.dumps([cat_names.get(x, str(x)) for x in sorted(residual_inter)], ensure_ascii=False),
            "removed_common_known_raw_ids": json.dumps([], ensure_ascii=False),
            "removed_common_known_names": json.dumps([], ensure_ascii=False),
        })

    ambiguity_rows: List[Dict[str, Any]] = []
    for idx, (key, rids) in enumerate(sorted(ambiguity_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])), start=1):
        residual = [int(x) for x in key.split(";") if x != ""]
        ambiguity_rows.append({
            "variant": variant_name,
            "ambiguity_group_id": idx,
            "group_size": len(rids),
            "target_raw_ids": json.dumps(sorted(rids), ensure_ascii=False),
            "target_names": json.dumps([cat_names.get(x, str(x)) for x in sorted(rids)], ensure_ascii=False),
            "residual_intersection_raw_ids": json.dumps(residual, ensure_ascii=False),
            "residual_intersection_names": json.dumps([cat_names.get(x, str(x)) for x in residual], ensure_ascii=False),
        })

    cert_counter = Counter(row["certificate_type"] for row in all_rows)
    resolved_count = sum(1 for row in all_rows if row["resolved"] is True)
    summary = {
        "variant": variant_name,
        "initial_known_count": len(initial_known),
        "final_known_count": len(known),
        "train_observed_base_count": len(train_observed_base),
        "resolved_total_count": resolved_count,
        "resolved_rate_among_train_observed_base": _safe_div(resolved_count, len(train_observed_base)),
        "newly_resolved_count": max(0, resolved_count - len(initial_known)),
        "unresolved_count": len(train_observed_base) - resolved_count,
        "unresolved_rate": _safe_div(len(train_observed_base) - resolved_count, len(train_observed_base)),
        "certificate_counts": dict(cert_counter),
        "newly_resolved_by_iteration": [r for r in iteration_rows if r.get("newly_resolved_count", 0) > 0],
        "max_iteration_with_new_classes": max([int(r["iteration"]) for r in iteration_rows if r.get("newly_resolved_count", 0) > 0] or [0]),
    }
    return {
        "summary": summary,
        "per_class_rows": all_rows,
        "iteration_rows": iteration_rows,
        "dependency_edges": dependency_edges,
        "ambiguity_rows": ambiguity_rows,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_takeover(path: Path, summary: Dict[str, Any]) -> None:
    lines = []
    lines.append("# Iterative Residual Label Identifiability Audit")
    lines.append("")
    lines.append(f"- Status: `{summary.get('status')}`")
    lines.append(f"- Dataset: `{summary.get('dataset_name')}`")
    lines.append(f"- Official base / novel: `{summary.get('base_count')}` / `{summary.get('novel_count')}`")
    lines.append(f"- Train-observed base: `{summary.get('train_observed_base_count')}`")
    lines.append(f"- Train-absent base: `{summary.get('train_absent_base_count')}`")
    lines.append("")
    lines.append("## Variant summaries")
    for v, s in summary.get("variants", {}).items():
        lines.append(f"### {v}")
        lines.append(f"- Initial known: `{s.get('initial_known_count')}`")
        lines.append(f"- Final resolved: `{s.get('resolved_total_count')}` / `{s.get('train_observed_base_count')}` = `{s.get('resolved_rate_among_train_observed_base'):.6f}`")
        lines.append(f"- Newly resolved: `{s.get('newly_resolved_count')}`")
        lines.append(f"- Unresolved: `{s.get('unresolved_count')}`")
        lines.append(f"- Max iteration with new classes: `{s.get('max_iteration_with_new_classes')}`")
        lines.append(f"- Certificate counts: `{json.dumps(s.get('certificate_counts', {}), ensure_ascii=False)}`")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("This is a label-only theoretical closure audit. It does not claim that GT features, projected text anchors, VideoCutLER trajectories, or the current scorer can recognize the resolved classes.")
    lines.append("Classes resolved beyond the initial anchors are conditionally identifiable after removing previously known classes from GT label contexts.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only iterative residual label identifiability audit.")
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--runtime_output_root", default=".")
    ap.add_argument("--dataset_name", default="lvvis_train_base")
    ap.add_argument("--annotation_json", required=True)
    ap.add_argument("--split_json", default="package/reference/lvvis_official_base_novel_split.json")
    ap.add_argument("--per_class_csv", default="", help="Optional GT context identifiability per-class CSV for anchor cross-check/source.")
    ap.add_argument("--anchor_source", choices=["auto", "computed", "per_class_csv"], default="auto")
    ap.add_argument("--person_raw_id", type=int, default=DEFAULT_PERSON_RAW_ID)
    ap.add_argument("--min_auditable_clips", type=int, default=DEFAULT_MIN_AUDITABLE_CLIPS)
    ap.add_argument("--max_iterations", type=int, default=50)
    ap.add_argument("--include_strict_anchor", action="store_true", default=True)
    ap.add_argument("--include_person_aware", action="store_true", default=True)
    ap.add_argument("--top_hub_raw_ids", default="", help="Optional comma-separated additional known hubs for top_hub_aware variant.")
    ap.add_argument("--top_examples", type=int, default=128)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    out_dir = run_root / "analysis" / "iterative_residual_label_identifiability" / args.dataset_name
    _ensure_dir(out_dir)

    split_path = Path(args.split_json)
    ann_path = Path(args.annotation_json)
    per_class_path = Path(args.per_class_csv) if args.per_class_csv else None

    base_set, novel_set, split_meta = _parse_split(split_path)
    contexts_all, cat_names, instance_count, clip_pair_count, ann_meta = _load_annotation_contexts(ann_path)

    # Base-only GT label context.
    base_clip_contexts: Dict[int, Set[int]] = {}
    for vid, cls in contexts_all.items():
        base_ctx = set(cls) & base_set
        if base_ctx:
            base_clip_contexts[vid] = base_ctx

    class_to_clips: Dict[int, List[int]] = defaultdict(list)
    for vid, cls in base_clip_contexts.items():
        for rid in cls:
            class_to_clips[rid].append(vid)

    train_observed_base = set(class_to_clips.keys()) & base_set
    train_absent_base = base_set - train_observed_base

    per_class = _read_per_class_csv(per_class_path)
    anchors_computed = _initial_anchors_computed(train_observed_base, class_to_clips, base_clip_contexts, args.min_auditable_clips)
    anchors_csv = _initial_anchors_from_csv(per_class, train_observed_base, args.min_auditable_clips) if per_class else set()

    if args.anchor_source == "computed":
        anchors = set(anchors_computed)
        anchor_source_used = "computed"
    elif args.anchor_source == "per_class_csv":
        anchors = set(anchors_csv)
        anchor_source_used = "per_class_csv"
    else:
        anchors = set(anchors_csv) if anchors_csv else set(anchors_computed)
        anchor_source_used = "per_class_csv" if anchors_csv else "computed"

    variants: Dict[str, Set[int]] = {}
    variants["strict_anchor"] = set(anchors)
    person_known = set(anchors)
    if args.person_raw_id in train_observed_base:
        person_known.add(args.person_raw_id)
    variants["person_aware"] = person_known

    top_hubs = {int(x) for x in args.top_hub_raw_ids.split(",") if x.strip()} if args.top_hub_raw_ids.strip() else set()
    if top_hubs:
        hub_known = set(person_known) | (top_hubs & train_observed_base)
        variants["top_hub_aware"] = hub_known

    all_per_class_rows: List[Dict[str, Any]] = []
    all_iteration_rows: List[Dict[str, Any]] = []
    all_dependency_edges: List[Dict[str, Any]] = []
    all_ambiguity_rows: List[Dict[str, Any]] = []
    variant_summaries: Dict[str, Any] = {}

    for variant_name, k0 in variants.items():
        res = _run_variant(
            variant_name=variant_name,
            initial_known=k0,
            train_observed_base=train_observed_base,
            class_to_clips=class_to_clips,
            clip_contexts=base_clip_contexts,
            cat_names=cat_names,
            instance_count=instance_count,
            clip_pair_count=clip_pair_count,
            person_id=args.person_raw_id,
            max_iterations=args.max_iterations,
        )
        variant_summaries[variant_name] = res["summary"]
        all_per_class_rows.extend(res["per_class_rows"])
        all_iteration_rows.extend(res["iteration_rows"])
        all_dependency_edges.extend(res["dependency_edges"])
        all_ambiguity_rows.extend(res["ambiguity_rows"])

    # Add absent rows once as audit metadata, not part of 525 train-observed denominator.
    absent_rows = []
    for rid in sorted(train_absent_base):
        absent_rows.append({
            "variant": "absent_metadata",
            "raw_id": rid,
            "class_name": cat_names.get(rid, str(rid)),
            "clip_count": 0,
            "instance_count": int(instance_count.get(rid, 0)),
            "initial_intersection_size": 0,
            "initial_intersection_raw_ids": "[]",
            "initial_intersection_names": "[]",
            "resolved": False,
            "resolved_at_iteration": "",
            "certificate_type": "train_absent_base_not_in_525_main_audit",
            "residual_intersection_size": 0,
            "residual_intersection_raw_ids": "[]",
            "residual_intersection_names": "[]",
            "removed_common_known_raw_ids": "[]",
            "removed_common_known_names": "[]",
        })

    # Summary.
    status = "PASS"
    warnings: List[str] = []
    if split_meta["base_count"] != 641 or split_meta["novel_count"] != 555:
        status = "WARN_SPLIT_COUNTS_UNEXPECTED"
        warnings.append("official split counts differ from expected 641/555")
    if anchors_csv and anchors_csv != anchors_computed:
        warnings.append(f"anchor set mismatch: per_class_csv={len(anchors_csv)} computed={len(anchors_computed)}; using {anchor_source_used}")

    summary = {
        "status": status,
        "dataset_name": args.dataset_name,
        "output_dir": str(out_dir),
        "official_split_path": str(split_path),
        "base_count": split_meta["base_count"],
        "novel_count": split_meta["novel_count"],
        "annotation_json": str(ann_path),
        "per_class_csv": str(per_class_path) if per_class_path else "",
        "anchor_source_used": anchor_source_used,
        "anchor_count_used": len(anchors),
        "anchor_count_computed": len(anchors_computed),
        "anchor_count_from_per_class_csv": len(anchors_csv),
        "person_raw_id": args.person_raw_id,
        "person_in_train_observed_base": args.person_raw_id in train_observed_base,
        "official_base_count": len(base_set),
        "train_observed_base_count": len(train_observed_base),
        "train_absent_base_count": len(train_absent_base),
        "base_context_clip_count": len(base_clip_contexts),
        "annotation_meta": ann_meta,
        "split_meta": split_meta,
        "variants": variant_summaries,
        "warnings": warnings,
        "outputs": {},
        "interpretation": {
            "purpose": "Compute the label-only closure of train-observed official-base classes under iterative residual peeling from known classes.",
            "valid_scientific_claim": "This audit provides a GT label-context theoretical upper bound. It does not assess GT feature separability, scorer quality, or trajectory/proposal support.",
        },
    }

    fields = [
        "variant", "raw_id", "class_name", "clip_count", "instance_count",
        "initial_intersection_size", "initial_intersection_raw_ids", "initial_intersection_names",
        "resolved", "resolved_at_iteration", "certificate_type",
        "residual_intersection_size", "residual_intersection_raw_ids", "residual_intersection_names",
        "removed_common_known_raw_ids", "removed_common_known_names",
    ]
    per_class_path_out = out_dir / "per_class_iterative_residual_identifiability.csv"
    _write_csv(per_class_path_out, all_per_class_rows + absent_rows, fields)

    iter_path = out_dir / "resolved_by_iteration.csv"
    _write_csv(iter_path, all_iteration_rows, ["variant", "iteration", "newly_resolved_count", "cumulative_known_count", "stop_reason"])

    dep_path = out_dir / "removed_class_dependency_edges.csv"
    _write_csv(dep_path, all_dependency_edges, ["variant", "target_raw_id", "target_name", "removed_known_raw_id", "removed_known_name", "resolved_at_iteration"])

    amb_path = out_dir / "unresolved_ambiguity_groups.csv"
    _write_csv(amb_path, all_ambiguity_rows, ["variant", "ambiguity_group_id", "group_size", "target_raw_ids", "target_names", "residual_intersection_raw_ids", "residual_intersection_names"])

    # Compact bucket summary by variant/certificate.
    bucket_rows = []
    for variant_name in variants:
        cc = Counter(r["certificate_type"] for r in all_per_class_rows if r["variant"] == variant_name)
        for cert, count in sorted(cc.items()):
            bucket_rows.append({"variant": variant_name, "certificate_type": cert, "count": count})
    bucket_path = out_dir / "certificate_bucket_summary.csv"
    _write_csv(bucket_path, bucket_rows, ["variant", "certificate_type", "count"])

    summary_path = out_dir / "summary.json"
    takeover_path = out_dir / "ITERATIVE_RESIDUAL_LABEL_IDENTIFIABILITY_TAKEOVER.md"
    summary["outputs"] = {
        "summary_json": str(summary_path),
        "per_class_csv": str(per_class_path_out),
        "resolved_by_iteration_csv": str(iter_path),
        "removed_dependency_edges_csv": str(dep_path),
        "unresolved_ambiguity_groups_csv": str(amb_path),
        "certificate_bucket_summary_csv": str(bucket_path),
        "takeover_md": str(takeover_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_takeover(takeover_path, summary)

    print(json.dumps({
        "status": summary["status"],
        "output_dir": str(out_dir),
        "official_base_count": len(base_set),
        "train_observed_base_count": len(train_observed_base),
        "train_absent_base_count": len(train_absent_base),
        "anchor_source_used": anchor_source_used,
        "anchor_count_used": len(anchors),
        "person_in_train_observed_base": args.person_raw_id in train_observed_base,
        "variants": variant_summaries,
        "outputs": summary["outputs"],
        "warnings": warnings,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
