#!/usr/bin/env python3
"""GT-fullY clean grouped diagnosis v2.

Read-only post-processing / diagnosis overlay for the clean GT trajectory + full Y_base
mechanism experiment.  It consumes the grouped attribution diagnosis outputs and, when
available, the clean training run outputs / GT carrier identity binding, then writes a
compact audit package with:

- top soft-routing improved/degraded classes;
- anchor-conditioned degradation and person-conditioned improvement slices;
- static-residual degradation slices;
- optional static supply collapse summaries from runtime metrics;
- optional skipped gt_not_in_y_base origin audit from GT carrier identity binding;
- a compact TAKEOVER.

The script does not train, does not evaluate LV-VIS mAP, and does not modify checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


NUMERIC_FIELDS = {
    "gt_count",
    "baseline_gt_count",
    "mean_normalized_gt_rank",
    "gt_top1_hit_rate",
    "candidate_size_mean",
    "gt_rank_mean",
    "baseline_mean_normalized_gt_rank",
    "baseline_gt_top1_hit_rate",
    "baseline_gt_rank_mean",
    "baseline_candidate_size_mean",
    "delta_mean_normalized_gt_rank",
    "delta_gt_top1_hit_rate",
    "delta_gt_rank_mean",
    "effective_trajectory_count_epoch",
    "candidate_size_mean_epoch",
    "empty_group_rate_epoch",
    "loss_mean",
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _to_float(v: Any, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _to_int(v: Any, default: int = 0) -> int:
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except Exception:
        return default


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def _sha256_file(path: Path, max_bytes: Optional[int] = None) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    read = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            if max_bytes is not None and read + len(chunk) > max_bytes:
                chunk = chunk[: max_bytes - read]
            h.update(chunk)
            read += len(chunk)
            if max_bytes is not None and read >= max_bytes:
                break
    return h.hexdigest()


def _copy_row_with_score(row: Mapping[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(row)
    out["sort_delta_gt_top1_hit_rate"] = _to_float(row.get("delta_gt_top1_hit_rate"))
    out["sort_delta_mean_normalized_gt_rank"] = _to_float(row.get("delta_mean_normalized_gt_rank"))
    out["sort_gt_count"] = _to_int(row.get("gt_count"))
    return out


def _select_top_classes(
    class_delta_rows: Sequence[Mapping[str, str]],
    run: str,
    top_k: int,
    mode: str,
    min_gt_count: int = 1,
    family: Optional[str] = None,
    person_conditioned: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in class_delta_rows:
        if r.get("run") != run:
            continue
        if _to_int(r.get("gt_count")) < min_gt_count:
            continue
        if family is not None and r.get("certificate_family") != family:
            continue
        if person_conditioned is not None and _truthy(r.get("person_conditioned")) != person_conditioned:
            continue
        rows.append(_copy_row_with_score(r))
    reverse = mode == "improved"
    # Top1 delta is primary; normalized rank delta is secondary with opposite sign.
    rows.sort(
        key=lambda x: (
            _to_float(x.get("delta_gt_top1_hit_rate")),
            -_to_float(x.get("delta_mean_normalized_gt_rank")),
            _to_int(x.get("gt_count")),
        ),
        reverse=reverse,
    )
    return rows[:top_k]


def _group_delta(rows: Sequence[Mapping[str, str]], run: str, group_name: Optional[str] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("run") != run:
            continue
        if group_name is not None and r.get("group_name") != group_name:
            continue
        out.append(_copy_row_with_score(r))
    out.sort(key=lambda x: _to_float(x.get("delta_gt_top1_hit_rate")))
    return out


def _read_runtime_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _static_supply_summary(run_root: Optional[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if run_root is None:
        return [], {"status": "SKIPPED", "reason": "run_root not provided"}
    candidates = [
        run_root / "gt_full_y_static_residual_15ep" / "train" / "prealign" / "runtime_metrics.jsonl",
        run_root / "gt_full_y_static_residual_15ep" / "train" / "prealign" / "clean_mechanism_metrics.jsonl",
    ]
    rows: List[Dict[str, Any]] = []
    source = None
    for p in candidates:
        rows = _read_runtime_jsonl(p)
        if rows:
            source = p
            break
    if not rows:
        return [], {"status": "SKIPPED", "reason": "runtime metrics not found", "candidate_paths": [str(p) for p in candidates]}
    epoch_rows = []
    for r in rows:
        if "epoch" not in r:
            continue
        epoch_rows.append(
            {
                "epoch": _to_int(r.get("epoch")),
                "protocol": r.get("protocol", ""),
                "loss_mean": _to_float(r.get("loss_mean")),
                "effective_trajectory_count_epoch": _to_float(r.get("effective_trajectory_count_epoch")),
                "candidate_size_mean_epoch": _to_float(r.get("candidate_size_mean_epoch")),
                "empty_group_rate_epoch": _to_float(r.get("empty_group_rate_epoch")),
            }
        )
    epoch_rows.sort(key=lambda x: x["epoch"])
    if not epoch_rows:
        return [], {"status": "SKIPPED", "reason": "no epoch rows", "source": str(source)}
    first = epoch_rows[0]
    last = epoch_rows[-1]
    min_eff = min(_to_float(r.get("effective_trajectory_count_epoch")) for r in epoch_rows)
    max_empty = max(_to_float(r.get("empty_group_rate_epoch")) for r in epoch_rows)
    summary = {
        "status": "PASS",
        "source": str(source),
        "epoch_count": len(epoch_rows),
        "first_epoch_effective_trajectory_count": first["effective_trajectory_count_epoch"],
        "last_epoch_effective_trajectory_count": last["effective_trajectory_count_epoch"],
        "min_effective_trajectory_count": min_eff,
        "max_empty_group_rate": max_empty,
        "collapse_ratio_last_vs_first": (last["effective_trajectory_count_epoch"] / first["effective_trajectory_count_epoch"]) if first["effective_trajectory_count_epoch"] else None,
    }
    return epoch_rows, summary


def _load_split_base_ids(path: Path) -> Tuple[set[int], set[int], Dict[str, Any]]:
    obj = _read_json(path)
    base_keys = ["base", "base_raw_ids", "official_base_raw_ids", "base_ids", "base_class_ids"]
    novel_keys = ["novel", "novel_raw_ids", "official_novel_raw_ids", "novel_ids", "novel_class_ids"]

    def collect(keys: Sequence[str]) -> set[int]:
        vals: Any = []
        for k in keys:
            if k in obj:
                vals = obj[k]
                break
        out: set[int] = set()
        if isinstance(vals, dict):
            vals = vals.values()
        for v in vals or []:
            if isinstance(v, dict):
                for kk in ("raw_id", "id", "category_id"):
                    if kk in v:
                        out.add(_to_int(v[kk], -1))
                        break
            else:
                out.add(_to_int(v, -1))
        out.discard(-1)
        return out

    base = collect(base_keys)
    novel = collect(novel_keys)
    meta = {"split_keys": list(obj.keys())[:50], "base_count": len(base), "novel_count": len(novel)}
    return base, novel, meta


def _load_annotation_context(path: Path, base_ids: set[int]) -> Tuple[Dict[Any, set[int]], Dict[int, str], Dict[str, Any]]:
    obj = _read_json(path)
    cats = obj.get("categories", []) or []
    id_to_name: Dict[int, str] = {}
    for c in cats:
        cid = _to_int(c.get("id", c.get("raw_id", c.get("category_id"))), -1)
        if cid != -1:
            id_to_name[cid] = str(c.get("name", c.get("synset", cid)))
    video_to_base: Dict[Any, set[int]] = defaultdict(set)
    for ann in obj.get("annotations", []) or []:
        raw = ann.get("category_id", ann.get("raw_id", ann.get("raw_category_id")))
        raw_id = _to_int(raw, -1)
        if raw_id not in base_ids:
            continue
        vid = ann.get("video_id", ann.get("video", ann.get("id_video")))
        if vid is None:
            # For image-only variants, use image_id as a fallback group id.
            vid = ann.get("image_id")
        if vid is not None:
            video_to_base[vid].add(raw_id)
    meta = {
        "annotation_category_count": len(id_to_name),
        "annotation_video_context_count": len(video_to_base),
        "annotation_count": len(obj.get("annotations", []) or []),
    }
    return dict(video_to_base), id_to_name, meta


def _extract_first_int(obj: Mapping[str, Any], keys: Sequence[str]) -> Optional[int]:
    for k in keys:
        if k in obj and obj[k] not in (None, ""):
            val = _to_int(obj[k], -999999)
            if val != -999999:
                return val
    return None


def _extract_first_value(obj: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in obj and obj[k] not in (None, ""):
            return obj[k]
    return None


def _audit_skipped_origin(
    asset_root: Optional[Path],
    dataset_name: str,
    annotation_json: Optional[Path],
    split_json: Optional[Path],
    output_dir: Path,
    expected_skipped_gt_not_in_y_base: Optional[int],
    max_examples: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if asset_root is None or annotation_json is None or split_json is None:
        reason = "asset_root/annotation_json/split_json not all provided"
        aggregate = []
        if expected_skipped_gt_not_in_y_base is not None:
            aggregate.append({"bucket": "aggregate_only_gt_not_in_y_base", "count": expected_skipped_gt_not_in_y_base, "note": reason})
        return aggregate, [], {"status": "SKIPPED", "reason": reason}

    gt_root = asset_root / "carrier_bank_gt" / dataset_name
    binding_path = gt_root / "gt_carrier_identity_binding.jsonl"
    if not binding_path.exists():
        reason = f"identity binding not found: {binding_path}"
        aggregate = []
        if expected_skipped_gt_not_in_y_base is not None:
            aggregate.append({"bucket": "aggregate_only_gt_not_in_y_base", "count": expected_skipped_gt_not_in_y_base, "note": reason})
        return aggregate, [], {"status": "SKIPPED", "reason": reason, "binding_path": str(binding_path)}

    base_ids, novel_ids, split_meta = _load_split_base_ids(split_json)
    video_to_base, id_to_name, ann_meta = _load_annotation_context(annotation_json, base_ids)
    bucket = Counter()
    per_class = Counter()
    examples: List[Dict[str, Any]] = []
    total = 0
    comparable = 0
    skipped_like = 0
    sample_keys: List[str] = []
    with binding_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bucket["bad_json"] += 1
                continue
            if not sample_keys:
                sample_keys = sorted(obj.keys())[:100]
            total += 1
            raw_id = _extract_first_int(
                obj,
                [
                    "gt_raw_id",
                    "matched_gt_raw_id",
                    "best_gt_raw_id",
                    "raw_id",
                    "category_id",
                    "class_raw_id",
                    "gt_category_id",
                ],
            )
            vid = _extract_first_value(obj, ["video_id", "video", "image_id", "clip_id"])
            traj = _extract_first_value(obj, ["trajectory_id", "carrier_id", "gt_id", "instance_id", "id"])
            if raw_id is None:
                b = "missing_gt_raw_id"
            elif raw_id in novel_ids:
                b = "gt_is_official_novel"
            elif raw_id not in base_ids:
                b = "gt_not_official_base"
            elif vid is None:
                b = "missing_video_or_clip_id"
            elif vid not in video_to_base:
                b = "missing_video_y_base_context"
            elif raw_id not in video_to_base[vid]:
                b = "official_base_but_not_in_video_y_base"
            else:
                b = "comparable_gt_in_y_base"
                comparable += 1
            bucket[b] += 1
            if b != "comparable_gt_in_y_base":
                skipped_like += 1
                if raw_id is not None:
                    per_class[(raw_id, id_to_name.get(raw_id, str(raw_id)), b)] += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "bucket": b,
                            "video_id": vid,
                            "trajectory_or_instance_id": traj,
                            "gt_raw_id": raw_id,
                            "class_name": id_to_name.get(raw_id, str(raw_id)) if raw_id is not None else "",
                            "y_base_size_for_video": len(video_to_base.get(vid, set())) if vid is not None else "",
                        }
                    )
    aggregate_rows = [{"bucket": k, "count": v} for k, v in bucket.most_common()]
    class_rows = [
        {"raw_id": rid, "class_name": name, "bucket": b, "count": count}
        for (rid, name, b), count in per_class.most_common()
    ]
    meta = {
        "status": "PASS",
        "binding_path": str(binding_path),
        "identity_binding_rows_seen": total,
        "comparable_gt_in_y_base_count": comparable,
        "skipped_like_count": skipped_like,
        "expected_skipped_gt_not_in_y_base": expected_skipped_gt_not_in_y_base,
        "binding_sample_keys": sample_keys,
        "split_meta": split_meta,
        "annotation_meta": ann_meta,
    }
    if expected_skipped_gt_not_in_y_base is not None:
        meta["note"] = (
            "This audit derives skipped-like rows from identity binding and annotation Y_base. "
            "Exact equality with compare skipped count depends on compare materialization filters."
        )
    return aggregate_rows, examples + class_rows[:0], meta | {"per_class_rows_count": len(class_rows), "examples_count": len(examples)}, class_rows  # type: ignore[return-value]


def _expected_skipped_from_summary(summary: Mapping[str, Any]) -> Optional[int]:
    vals = []
    for c in summary.get("checkpoint_summaries", []) or []:
        skipped = c.get("skipped", {}) or {}
        if "gt_not_in_y_base" in skipped:
            vals.append(_to_int(skipped.get("gt_not_in_y_base")))
    if not vals:
        return None
    # Checkpoints should have identical skipped counts in this clean compare.
    return vals[0]


def _write_takeover(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# GT Full-Y Clean Diagnosis V2",
        "",
        f"Status: `{summary.get('status', 'UNKNOWN')}`",
        "",
        f"Output: `{summary.get('output_dir', '')}`",
        "",
        "## Main result",
    ]
    main = summary.get("main_result", {}) or {}
    for k, v in main.items():
        lines.append(f"- {k}: `{v}`")
    lines.extend(["", "## Key files"])
    for f in summary.get("key_outputs", []) or []:
        lines.append(f"- `{f}`")
    lines.extend(["", "## Interpretation", ""])
    interp = summary.get("interpretation", {}) or {}
    for k, v in interp.items():
        lines.append(f"- **{k}**: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grouped_dir", required=True, help="Existing grouped attribution diagnosis output directory")
    ap.add_argument("--compare_dir", default="", help="Existing clean attribution compare output directory, optional")
    ap.add_argument("--output_dir", required=True, help="Output directory for v2 diagnosis package")
    ap.add_argument("--run_root", default="", help="Clean 15ep run root, optional; used for static supply metrics")
    ap.add_argument("--dataset_name", default="lvvis_train_base")
    ap.add_argument("--asset_root", default="", help="Asset root, optional; used for skipped origin audit")
    ap.add_argument("--annotation_json", default="", help="LV-VIS train annotation json, optional")
    ap.add_argument("--split_json", default="", help="Official base/novel split json, optional")
    ap.add_argument("--schedule_csv", default="", help="Schedule csv, optional; recorded for provenance")
    ap.add_argument("--weak_labels_json", default="", help="Weak label json, optional; recorded for provenance")
    ap.add_argument("--baseline", default="baseline_full_y")
    ap.add_argument("--target", default="soft_routing")
    ap.add_argument("--static_run", default="static_residual")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--min_class_gt_count", type=int, default=1)
    ap.add_argument("--max_skip_examples", type=int, default=128)
    args = ap.parse_args()

    grouped_dir = Path(args.grouped_dir)
    compare_dir = Path(args.compare_dir) if args.compare_dir else grouped_dir.parent.parent / "gt_full_y_clean_attribution_compare" / args.dataset_name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "summary_json": grouped_dir / "summary.json",
        "summary_by_run": grouped_dir / "summary_by_run.csv",
        "group_delta": grouped_dir / "summary_delta_vs_baseline_by_group.csv",
        "per_class_delta": grouped_dir / "per_class_delta_vs_baseline.csv",
        "per_class_attr": grouped_dir / "per_class_attribution.csv",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        _write_json(
            output_dir / "summary.json",
            {"status": "FAIL", "reason": "missing required grouped diagnosis inputs", "missing": missing, "grouped_dir": str(grouped_dir)},
        )
        return 2

    summary_in = _read_json(required["summary_json"])
    by_run = _read_csv(required["summary_by_run"])
    group_delta = _read_csv(required["group_delta"])
    class_delta = _read_csv(required["per_class_delta"])
    class_attr = _read_csv(required["per_class_attr"])

    soft_improved = _select_top_classes(class_delta, args.target, args.top_k, "improved", args.min_class_gt_count)
    soft_degraded = _select_top_classes(class_delta, args.target, args.top_k, "degraded", args.min_class_gt_count)
    anchor_degraded = _select_top_classes(
        class_delta,
        args.target,
        args.top_k,
        "degraded",
        args.min_class_gt_count,
        family="anchor_conditioned",
    )
    person_improved = _select_top_classes(
        class_delta,
        args.target,
        args.top_k,
        "improved",
        args.min_class_gt_count,
        person_conditioned=True,
    )
    static_degraded_classes = _select_top_classes(class_delta, args.static_run, args.top_k, "degraded", args.min_class_gt_count)
    static_degraded_groups = _group_delta(group_delta, args.static_run)[: args.top_k]
    soft_group_delta = _group_delta(group_delta, args.target)

    _write_csv(output_dir / "top20_soft_improved_classes.csv", soft_improved)
    _write_csv(output_dir / "top20_soft_degraded_classes.csv", soft_degraded)
    _write_csv(output_dir / "anchor_conditioned_soft_degraded_classes.csv", anchor_degraded)
    _write_csv(output_dir / "person_conditioned_soft_improved_classes.csv", person_improved)
    _write_csv(output_dir / "top20_static_degraded_classes.csv", static_degraded_classes)
    _write_csv(output_dir / "static_residual_degraded_groups.csv", static_degraded_groups)
    _write_csv(output_dir / "soft_routing_delta_by_group.csv", soft_group_delta)

    # Copy compact core tables for convenience.
    _write_csv(output_dir / "summary_by_run.csv", by_run)
    _write_csv(output_dir / "summary_delta_vs_baseline_by_group.csv", group_delta)

    run_root = Path(args.run_root) if args.run_root else None
    supply_rows, supply_summary = _static_supply_summary(run_root)
    _write_csv(output_dir / "static_residual_supply_by_epoch.csv", supply_rows)
    _write_json(output_dir / "static_residual_supply_summary.json", supply_summary)

    expected_skipped = _expected_skipped_from_summary(summary_in)
    skip_meta: Dict[str, Any]
    skip_aggregate: List[Dict[str, Any]]
    skip_examples: List[Dict[str, Any]]
    skip_per_class: List[Dict[str, Any]] = []
    try:
        res = _audit_skipped_origin(
            Path(args.asset_root) if args.asset_root else None,
            args.dataset_name,
            Path(args.annotation_json) if args.annotation_json else None,
            Path(args.split_json) if args.split_json else None,
            output_dir,
            expected_skipped,
            args.max_skip_examples,
        )
        if len(res) == 4:
            skip_aggregate, skip_examples, skip_meta, skip_per_class = res  # type: ignore[misc]
        else:
            skip_aggregate, skip_examples, skip_meta = res  # type: ignore[misc]
    except Exception as e:
        skip_aggregate = [{"bucket": "audit_error", "count": expected_skipped or "", "note": repr(e)}]
        skip_examples = []
        skip_meta = {"status": "FAIL", "reason": repr(e)}
    _write_csv(output_dir / "skipped_gt_not_in_y_base_origin.csv", skip_aggregate)
    _write_csv(output_dir / "skipped_gt_not_in_y_base_examples.jsonl.csv", skip_examples)
    _write_csv(output_dir / "skipped_gt_not_in_y_base_by_class.csv", skip_per_class)
    _write_json(output_dir / "skipped_gt_not_in_y_base_origin_summary.json", skip_meta)

    def find_overall(run: str) -> Optional[Dict[str, str]]:
        for r in by_run:
            if r.get("checkpoint") == run and r.get("group_name") == "overall":
                return r
        for r in by_run:
            if r.get("checkpoint") == run:
                return r
        return None

    def find_delta(run: str, group_name: str = "overall", group_value: str = "overall") -> Optional[Dict[str, str]]:
        for r in group_delta:
            if r.get("run") == run and r.get("group_name") == group_name and r.get("group_value") == group_value:
                return r
        return None

    soft_delta = find_delta(args.target) or {}
    static_delta = find_delta(args.static_run) or {}
    soft_gain = _to_float(soft_delta.get("delta_gt_top1_hit_rate")) > 0 and _to_float(soft_delta.get("delta_mean_normalized_gt_rank")) < 0
    static_failed = _to_float(static_delta.get("delta_gt_top1_hit_rate")) < 0 and _to_float(static_delta.get("delta_mean_normalized_gt_rank")) > 0

    status = "PASS"
    key_outputs = [
        "summary.json",
        "top20_soft_improved_classes.csv",
        "top20_soft_degraded_classes.csv",
        "anchor_conditioned_soft_degraded_classes.csv",
        "person_conditioned_soft_improved_classes.csv",
        "static_residual_degraded_groups.csv",
        "static_residual_supply_by_epoch.csv",
        "skipped_gt_not_in_y_base_origin.csv",
        "skipped_gt_not_in_y_base_by_class.csv",
        "GT_FULL_Y_CLEAN_DIAGNOSIS_V2_TAKEOVER.md",
    ]
    summary_out: Dict[str, Any] = {
        "status": status,
        "output_dir": str(output_dir),
        "dataset_name": args.dataset_name,
        "input_grouped_dir": str(grouped_dir),
        "input_compare_dir": str(compare_dir),
        "baseline": args.baseline,
        "target": args.target,
        "static_run": args.static_run,
        "main_result": {
            "soft_gain_vs_baseline": soft_gain,
            "soft_delta_gt_top1_hit_rate": soft_delta.get("delta_gt_top1_hit_rate", ""),
            "soft_delta_mean_normalized_gt_rank": soft_delta.get("delta_mean_normalized_gt_rank", ""),
            "static_failed_vs_baseline": static_failed,
            "static_delta_gt_top1_hit_rate": static_delta.get("delta_gt_top1_hit_rate", ""),
            "static_delta_mean_normalized_gt_rank": static_delta.get("delta_mean_normalized_gt_rank", ""),
        },
        "counts": {
            "per_class_delta_rows": len(class_delta),
            "per_class_attribution_rows": len(class_attr),
            "group_delta_rows": len(group_delta),
            "top_k": args.top_k,
            "min_class_gt_count": args.min_class_gt_count,
        },
        "static_supply_summary": supply_summary,
        "skipped_origin_summary": skip_meta,
        "provenance": {
            "annotation_json": args.annotation_json,
            "annotation_sha256": _sha256_file(Path(args.annotation_json)) if args.annotation_json else None,
            "split_json": args.split_json,
            "split_sha256": _sha256_file(Path(args.split_json)) if args.split_json else None,
            "schedule_csv": args.schedule_csv,
            "schedule_sha256": _sha256_file(Path(args.schedule_csv)) if args.schedule_csv else None,
            "weak_labels_json": args.weak_labels_json,
            "weak_labels_sha256": _sha256_file(Path(args.weak_labels_json)) if args.weak_labels_json else None,
        },
        "key_outputs": key_outputs,
        "interpretation": {
            "soft_routing": "Keep as main clean-mechanism direction if soft_gain_vs_baseline is true; inspect top improved/degraded classes before migration.",
            "static_residual": "Treat as failed training protocol if static_failed_vs_baseline is true and static supply collapse is present.",
            "next_decision": "Use class-level and skipped-row diagnostics to decide whether to add certificate-aware floors, anchor protection, hub caps, or demand-balanced sampling.",
        },
    }
    _write_json(output_dir / "summary.json", summary_out)
    _write_takeover(output_dir / "GT_FULL_Y_CLEAN_DIAGNOSIS_V2_TAKEOVER.md", summary_out)

    print(json.dumps(summary_out, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
