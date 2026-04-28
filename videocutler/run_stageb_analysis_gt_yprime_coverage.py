from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Record = Dict[str, Any]


def _bootstrap_repo_root_for_direct_cli() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


REPO_ROOT = _bootstrap_repo_root_for_direct_cli()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterable[Record]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _safe_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    if v is None:
        return default
    try:
        if isinstance(v, bool):
            return int(v)
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default


def _unique_ints(v: Any) -> List[int]:
    if v is None:
        return []
    if isinstance(v, str):
        parts = v.replace(";", ",").split(",")
    elif isinstance(v, Mapping):
        parts = v.keys()
    elif isinstance(v, Iterable):
        parts = v
    else:
        parts = [v]
    out: List[int] = []
    seen = set()
    for x in parts:
        ix = _safe_int(x)
        if ix is None or ix in seen:
            continue
        seen.add(ix)
        out.append(int(ix))
    return out


def _rate(n: int, d: int) -> Optional[float]:
    return None if d <= 0 else float(n) / float(d)


def _mean(xs: Sequence[float]) -> Optional[float]:
    return None if not xs else float(sum(xs) / len(xs))


def _median(xs: Sequence[float]) -> Optional[float]:
    return None if not xs else float(median(xs))


def _candidate_paths_from_summary(summary_path: Path, runtime_output_root: Path) -> List[Path]:
    out: List[Path] = []
    if not summary_path.is_file():
        return out
    try:
        summary = _load_json(summary_path)
    except Exception:
        return out
    for key in ("payload_path", "payload_output", "weak_labels_path", "output_json"):
        val = summary.get(key) if isinstance(summary, Mapping) else None
        if not val:
            continue
        p = Path(str(val)).expanduser()
        out.append(p)
        # Common Windows/local path in package reports. Remap to runtime root.
        s = str(p)
        for anchor in ("/mnt/e/Code/wsovvis", "E:/Code/wsovvis", "E:\\Code\\wsovvis"):
            if s.startswith(anchor):
                rel = s[len(anchor):].lstrip("/\\")
                out.append(runtime_output_root / rel.replace("\\", "/"))
    return out


def _resolve_weak_labels_path(*, runtime_output_root: Path, run_root: Path, explicit: Optional[str]) -> Path:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend([
        run_root / "weak_labels" / "weak_labels_train.json",
        run_root / "train" / "weak_labels_train.json",
        runtime_output_root / "codex" / "outputs" / "g3_weak_labels" / "weak_labels" / "weak_labels_train.json",
        REPO_ROOT / "codex" / "outputs" / "g3_weak_labels" / "weak_labels" / "weak_labels_train.json",
    ])
    candidates.extend(_candidate_paths_from_summary(runtime_output_root / "codex" / "outputs" / "g3_weak_labels" / "g3_weak_labels_summary.json", runtime_output_root))
    candidates.extend(_candidate_paths_from_summary(REPO_ROOT / "codex" / "outputs" / "g3_weak_labels" / "g3_weak_labels_summary.json", runtime_output_root))
    seen = set()
    for p in candidates:
        p = p.expanduser()
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError("weak_labels_train.json not found. Pass --weak_labels_json explicitly.")


def _resolve_annotation_path(*, runtime_output_root: Path, explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p.resolve()
    # Prefer project resolver if available.
    try:
        from videocutler.ext_stageb_ovvis.eval.external_lvvis import resolve_lvvis_annotation_paths  # type: ignore
        return resolve_lvvis_annotation_paths(validate_official_authority=False).train_json.resolve()
    except Exception:
        pass
    candidates = [
        Path(os.environ.get("WSOVVIS_LVVIS_ROOT", "")) / "annotations" / "train_instances.json" if os.environ.get("WSOVVIS_LVVIS_ROOT") else None,
        REPO_ROOT / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "train_instances.json",
        runtime_output_root / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "train_instances.json",
        Path("/home/zyy/code/wsovvis_asserts/dataset/LV-VIS/annotations/train_instances.json"),
        Path("/mnt/sda/zyy/code/wsovvis_asserts/dataset/LV-VIS/annotations/train_instances.json"),
    ]
    for p in candidates:
        if p is not None and p.is_file():
            return p.resolve()
    raise FileNotFoundError("LV-VIS train annotation not found. Pass --annotation_json explicitly.")


def _load_official_base_novel() -> Tuple[set[int], set[int], Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    try:
        from videocutler.ext_stageb_ovvis.data.datasets.lvvis_official_split import load_lvvis_base_and_novel_raw_ids  # type: ignore
        base, novel = load_lvvis_base_and_novel_raw_ids()
        meta = {"source": "load_lvvis_base_and_novel_raw_ids", "base_count": len(base), "novel_count": len(novel)}
        return {int(x) for x in base}, {int(x) for x in novel}, meta
    except Exception as exc:
        meta["import_error"] = repr(exc)
    p = REPO_ROOT / "package" / "reference" / "lvvis_official_base_novel_split.json"
    obj = _load_json(p)
    base = obj.get("base_raw_ids") or obj.get("base_ids") or obj.get("base") or []
    novel = obj.get("novel_raw_ids") or obj.get("novel_ids") or obj.get("novel") or []
    if base and isinstance(base[0], Mapping):
        base = [r.get("raw_id", r.get("id")) for r in base]
    if novel and isinstance(novel[0], Mapping):
        novel = [r.get("raw_id", r.get("id")) for r in novel]
    meta.update({"source": str(p), "base_count": len(base), "novel_count": len(novel)})
    return {int(x) for x in base}, {int(x) for x in novel}, meta


def _load_class_names(annotation: Mapping[str, Any]) -> Dict[int, str]:
    names: Dict[int, str] = {}
    for cat in annotation.get("categories", []) or []:
        cid = _safe_int(cat.get("id"), None)
        if cid is None:
            continue
        names[int(cid)] = str(cat.get("name") or cat.get("class_name") or cid)
    return names


def _load_gt_sets(annotation_json: Path, *, base_only: bool, base_ids: set[int]) -> Tuple[Dict[int, set[int]], Dict[int, str], Dict[str, Any]]:
    obj = _load_json(annotation_json)
    names = _load_class_names(obj if isinstance(obj, Mapping) else {})
    by_clip: Dict[int, set[int]] = defaultdict(set)
    ann_count = 0
    for ann in obj.get("annotations", []) if isinstance(obj, Mapping) else []:
        vid = _safe_int(ann.get("video_id"), None)
        cid = _safe_int(ann.get("category_id"), None)
        if vid is None or cid is None:
            continue
        if base_only and int(cid) not in base_ids:
            continue
        by_clip[int(vid)].add(int(cid))
        ann_count += 1
    meta = {
        "annotation_json": str(annotation_json),
        "annotation_count_seen_after_base_filter": ann_count,
        "clip_count_with_gt": len(by_clip),
        "base_only": bool(base_only),
    }
    return by_clip, names, meta


def _load_weak_sets(weak_labels_json: Path) -> Tuple[Dict[int, set[int]], Dict[int, Record], Dict[str, Any]]:
    obj = _load_json(weak_labels_json)
    if isinstance(obj, Mapping):
        records = obj.get("records") or obj.get("weak_labels") or obj.get("data") or []
    else:
        records = obj
    by_clip: Dict[int, set[int]] = {}
    by_clip_record: Dict[int, Record] = {}
    for rec in records or []:
        if not isinstance(rec, Mapping):
            continue
        clip = _safe_int(rec.get("clip_id", rec.get("video_id")), None)
        if clip is None:
            continue
        obs = _unique_ints(rec.get("observed_raw_ids") or rec.get("yprime_raw_ids") or rec.get("weak_raw_ids") or rec.get("observed_category_ids"))
        by_clip[int(clip)] = set(int(x) for x in obs)
        by_clip_record[int(clip)] = dict(rec)
    meta = {"weak_labels_json": str(weak_labels_json), "record_count": len(by_clip)}
    return by_clip, by_clip_record, meta


def _resolve_yprime_support_rows(run_root: Path, dataset_name: str, stage: str, explicit: Optional[str]) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(run_root / "analysis" / "yprime_support_coverage" / dataset_name / stage / "clip_yprime_support_rows.jsonl")
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None


def _load_carrier_support_rows(path: Optional[Path]) -> Tuple[Dict[Tuple[int, int], Record], Dict[str, Any]]:
    support: Dict[Tuple[int, int], Record] = {}
    count = 0
    if path is None or not path.is_file():
        return support, {"available": False, "path": str(path) if path else None, "record_count": 0}
    for row in _iter_jsonl(path):
        clip = _safe_int(row.get("clip_id"), None)
        y = _safe_int(row.get("yprime_raw_id"), None)
        if clip is None or y is None:
            continue
        support[(int(clip), int(y))] = dict(row)
        count += 1
    return support, {"available": True, "path": str(path), "record_count": count, "join_count": len(support)}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    run_root = Path(args.run_root).expanduser().resolve()
    runtime_output_root = Path(args.runtime_output_root).expanduser().resolve()
    stage = str(args.stage)
    dataset_name = str(args.dataset_name)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_root / "analysis" / "gt_yprime_coverage" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    base_ids, novel_ids, split_meta = _load_official_base_novel()
    split_pass = len(base_ids) == 641 and len(novel_ids) == 555
    weak_path = _resolve_weak_labels_path(runtime_output_root=runtime_output_root, run_root=run_root, explicit=args.weak_labels_json)
    ann_path = _resolve_annotation_path(runtime_output_root=runtime_output_root, explicit=args.annotation_json)
    gt_by_clip, names, ann_meta = _load_gt_sets(ann_path, base_only=bool(args.base_only), base_ids=base_ids)
    yprime_by_clip, _weak_records, weak_meta = _load_weak_sets(weak_path)
    yprime_rows_path = _resolve_yprime_support_rows(run_root, dataset_name, stage, args.yprime_support_rows)
    carrier_support, carrier_meta = _load_carrier_support_rows(yprime_rows_path)

    all_clips = sorted(set(yprime_by_clip) | set(gt_by_clip))
    yprime_pair_count = 0
    yprime_in_gt_count = 0
    yprime_not_in_gt_count = 0
    clip_all_yprime_in_gt_flags: List[bool] = []
    clip_has_yprime_flags: List[bool] = []

    gt_pair_count = 0
    gt_covered_count = 0
    gt_missing_count = 0
    clip_all_gt_covered_flags: List[bool] = []
    clip_has_gt_flags: List[bool] = []

    yprime_in_gt_and_has_carrier = 0
    yprime_in_gt_but_no_carrier = 0
    yprime_carrier_support_known_pairs = 0

    rows: List[Record] = []
    examples: List[Record] = []
    bucket_counts: Counter = Counter()
    class_stats: Dict[int, Counter] = defaultdict(Counter)

    for clip in all_clips:
        yset = set(int(x) for x in yprime_by_clip.get(clip, set()))
        gset = set(int(x) for x in gt_by_clip.get(clip, set()))
        if yset:
            clip_has_yprime_flags.append(True)
            clip_all_yprime_in_gt_flags.append(yset.issubset(gset))
        if gset:
            clip_has_gt_flags.append(True)
            clip_all_gt_covered_flags.append(gset.issubset(yset))

        for y in sorted(yset):
            yprime_pair_count += 1
            in_gt = y in gset
            if in_gt:
                yprime_in_gt_count += 1
                class_stats[y]["yprime_in_gt_count"] += 1
            else:
                yprime_not_in_gt_count += 1
                class_stats[y]["yprime_not_in_gt_count"] += 1
            class_stats[y]["yprime_pair_count"] += 1

            carrier_row = carrier_support.get((clip, y))
            carrier_known = carrier_row is not None
            has_carrier = bool(carrier_row.get("has_trajectory_support")) if isinstance(carrier_row, Mapping) else None
            if in_gt and carrier_known:
                yprime_carrier_support_known_pairs += 1
                if has_carrier:
                    yprime_in_gt_and_has_carrier += 1
                    bucket = "yprime_in_gt_and_has_carrier"
                else:
                    yprime_in_gt_but_no_carrier += 1
                    bucket = "yprime_in_gt_but_no_carrier"
            elif not in_gt:
                bucket = "yprime_not_in_gt"
            elif in_gt and not carrier_known:
                bucket = "yprime_in_gt_carrier_unknown"
            else:
                bucket = "unknown"
            bucket_counts[bucket] += 1
            class_stats[y][bucket] += 1

            row = {
                "record_type": "yprime_pair",
                "clip_id": int(clip),
                "raw_id": int(y),
                "name": names.get(int(y), str(y)),
                "in_gt_annotation": bool(in_gt),
                "carrier_support_known": bool(carrier_known),
                "has_carrier_support": has_carrier,
                "bucket": bucket,
                "gt_class_count_in_clip": int(len(gset)),
                "yprime_class_count_in_clip": int(len(yset)),
            }
            rows.append(row)
            if len(examples) < int(args.top_examples) and bucket != "yprime_in_gt_and_has_carrier":
                examples.append(row)

        for g in sorted(gset):
            gt_pair_count += 1
            covered = g in yset
            if covered:
                gt_covered_count += 1
                class_stats[g]["gt_covered_by_yprime_count"] += 1
            else:
                gt_missing_count += 1
                class_stats[g]["gt_missing_from_yprime_count"] += 1
                bucket_counts["gt_missing_from_yprime"] += 1
                row = {
                    "record_type": "gt_missing_from_yprime",
                    "clip_id": int(clip),
                    "raw_id": int(g),
                    "name": names.get(int(g), str(g)),
                    "in_yprime": False,
                    "bucket": "gt_missing_from_yprime",
                    "gt_class_count_in_clip": int(len(gset)),
                    "yprime_class_count_in_clip": int(len(yset)),
                }
                rows.append(row)
                if len(examples) < int(args.top_examples):
                    examples.append(row)
            class_stats[g]["gt_pair_count"] += 1

    class_rows: List[Record] = []
    for raw_id, c in sorted(class_stats.items()):
        yp = int(c.get("yprime_pair_count", 0))
        gp = int(c.get("gt_pair_count", 0))
        in_gt = int(c.get("yprime_in_gt_count", 0))
        y_no_gt = int(c.get("yprime_not_in_gt_count", 0))
        gt_cov = int(c.get("gt_covered_by_yprime_count", 0))
        gt_miss = int(c.get("gt_missing_from_yprime_count", 0))
        in_gt_no_carrier = int(c.get("yprime_in_gt_but_no_carrier", 0))
        in_gt_has_carrier = int(c.get("yprime_in_gt_and_has_carrier", 0))
        class_rows.append({
            "raw_id": int(raw_id),
            "name": names.get(int(raw_id), str(raw_id)),
            "yprime_pair_count": yp,
            "gt_pair_count": gp,
            "yprime_in_gt_count": in_gt,
            "yprime_not_in_gt_count": y_no_gt,
            "yprime_gt_annotation_support_rate": _rate(in_gt, yp),
            "gt_covered_by_yprime_count": gt_cov,
            "gt_missing_from_yprime_count": gt_miss,
            "gt_covered_by_yprime_rate": _rate(gt_cov, gp),
            "yprime_in_gt_and_has_carrier": in_gt_has_carrier,
            "yprime_in_gt_but_no_carrier": in_gt_no_carrier,
            "yprime_in_gt_but_no_carrier_rate": _rate(in_gt_no_carrier, in_gt_has_carrier + in_gt_no_carrier),
        })

    failure_rows = [{"bucket": k, "count": int(v), "rate_vs_yprime_pairs": _rate(int(v), yprime_pair_count), "rate_vs_all_rows": _rate(int(v), len(rows))} for k, v in bucket_counts.most_common()]

    yprime_gt_annotation_support_rate = _rate(yprime_in_gt_count, yprime_pair_count)
    yprime_trajectory_support_rate = None
    if carrier_meta.get("available"):
        # This is only for Y' pairs with row-level support audit rows, matching the prior support audit universe.
        known = yprime_carrier_support_known_pairs
        yprime_trajectory_support_rate = _rate(yprime_in_gt_and_has_carrier, known) if known else None
    summary: Record = {
        "status": "PASS" if split_pass else "WARN_SPLIT_COUNT_UNEXPECTED",
        "dataset_name": dataset_name,
        "stage": stage,
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "split_meta": split_meta,
        "official_split_pass_641_555": bool(split_pass),
        "weak_labels_meta": weak_meta,
        "annotation_meta": ann_meta,
        "carrier_support_meta": carrier_meta,
        "clip_count_union": int(len(all_clips)),
        "clip_count_with_yprime": int(len(clip_has_yprime_flags)),
        "clip_count_with_gt": int(len(clip_has_gt_flags)),
        "clip_yprime_pair_count": int(yprime_pair_count),
        "clip_gt_class_pair_count": int(gt_pair_count),
        "yprime_gt_annotation_support_rate": yprime_gt_annotation_support_rate,
        "yprime_false_positive_rate": _rate(yprime_not_in_gt_count, yprime_pair_count),
        "gt_covered_by_yprime_rate": _rate(gt_covered_count, gt_pair_count),
        "gt_missing_from_yprime_rate": _rate(gt_missing_count, gt_pair_count),
        "clip_all_yprime_in_gt_rate": _rate(sum(1 for x in clip_all_yprime_in_gt_flags if x), len(clip_all_yprime_in_gt_flags)),
        "clip_all_gt_covered_by_yprime_rate": _rate(sum(1 for x in clip_all_gt_covered_flags if x), len(clip_all_gt_covered_flags)),
        "yprime_in_gt_and_has_carrier_count": int(yprime_in_gt_and_has_carrier),
        "yprime_in_gt_but_no_carrier_count": int(yprime_in_gt_but_no_carrier),
        "yprime_annotation_supported_but_no_carrier_rate": _rate(yprime_in_gt_but_no_carrier, yprime_in_gt_count),
        "yprime_trajectory_support_rate_on_annotation_supported_pairs": yprime_trajectory_support_rate,
        "gt_to_carrier_coverage_gap": (None if yprime_gt_annotation_support_rate is None or yprime_trajectory_support_rate is None else float(yprime_gt_annotation_support_rate - yprime_trajectory_support_rate)),
        "failure_bucket_counts": {k: int(v) for k, v in bucket_counts.items()},
        "interpretation": {},
    }
    y_ann = summary["yprime_gt_annotation_support_rate"]
    y_car = summary["yprime_trajectory_support_rate_on_annotation_supported_pairs"]
    gt_cov_rate = summary["gt_covered_by_yprime_rate"]
    if y_ann is not None and y_ann >= 0.95 and y_car is not None and y_car < 0.60:
        verdict = "yprime_clean_but_proposal_carrier_support_gap_large"
    elif y_ann is not None and y_ann < 0.90:
        verdict = "weak_label_false_positive_or_scope_mismatch_nontrivial"
    elif gt_cov_rate is not None and gt_cov_rate < 0.75:
        verdict = "yprime_incomplete_relative_to_gt_extra_needed"
    else:
        verdict = "gt_yprime_coverage_mixed_or_needs_review"
    summary["interpretation"] = {
        "verdict": verdict,
        "primary_reading": "High Yprime-to-GT support with low carrier support indicates proposal/carrier coverage gap; low Yprime-to-GT support indicates weak-label false positives or clip-id/scope mismatch; low GT-to-Yprime support quantifies weak-label incompleteness.",
    }

    _dump_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "class_gt_yprime_summary.csv", class_rows, [
        "raw_id", "name", "yprime_pair_count", "gt_pair_count", "yprime_in_gt_count", "yprime_not_in_gt_count", "yprime_gt_annotation_support_rate", "gt_covered_by_yprime_count", "gt_missing_from_yprime_count", "gt_covered_by_yprime_rate", "yprime_in_gt_and_has_carrier", "yprime_in_gt_but_no_carrier", "yprime_in_gt_but_no_carrier_rate",
    ])
    _write_csv(output_dir / "failure_bucket_summary.csv", failure_rows, ["bucket", "count", "rate_vs_yprime_pairs", "rate_vs_all_rows"])
    with (output_dir / "clip_gt_yprime_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "gt_yprime_examples.jsonl").open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    md = [
        "# GT-Y′ Coverage Audit",
        "",
        f"- status: {summary['status']}",
        f"- verdict: {verdict}",
        f"- dataset: {dataset_name}",
        f"- clip_yprime_pair_count: {yprime_pair_count}",
        f"- clip_gt_class_pair_count: {gt_pair_count}",
        f"- yprime_gt_annotation_support_rate: {summary['yprime_gt_annotation_support_rate']}",
        f"- yprime_false_positive_rate: {summary['yprime_false_positive_rate']}",
        f"- gt_covered_by_yprime_rate: {summary['gt_covered_by_yprime_rate']}",
        f"- yprime_annotation_supported_but_no_carrier_rate: {summary['yprime_annotation_supported_but_no_carrier_rate']}",
        f"- gt_to_carrier_coverage_gap: {summary['gt_to_carrier_coverage_gap']}",
        "",
        "## Outputs",
        f"- summary: `{output_dir / 'summary.json'}`",
        f"- class summary: `{output_dir / 'class_gt_yprime_summary.csv'}`",
        f"- failure buckets: `{output_dir / 'failure_bucket_summary.csv'}`",
        f"- rows: `{output_dir / 'clip_gt_yprime_rows.jsonl'}`",
    ]
    (output_dir / "GT_YPRIME_COVERAGE_TAKEOVER.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only audit of GT annotation coverage by Y′ weak labels and proposal/carrier support gap.")
    p.add_argument("--run_root", required=True)
    p.add_argument("--runtime_output_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--stage", default="softem_aug")
    p.add_argument("--output_dir")
    p.add_argument("--annotation_json", help="Optional LV-VIS train_instances.json override")
    p.add_argument("--weak_labels_json", help="Optional weak_labels_train.json override")
    p.add_argument("--yprime_support_rows", help="Optional yprime_support_coverage clip_yprime_support_rows.jsonl override")
    p.add_argument("--base_only", type=lambda x: str(x).lower() not in {"0", "false", "no", "off"}, default=True)
    p.add_argument("--top_examples", type=int, default=128)
    args = p.parse_args()
    return args


def main() -> int:
    summary = run_audit(parse_args())
    print(json.dumps({
        "status": summary.get("status"),
        "verdict": summary.get("interpretation", {}).get("verdict"),
        "output_dir": summary.get("output_dir"),
        "yprime_gt_annotation_support_rate": summary.get("yprime_gt_annotation_support_rate"),
        "gt_covered_by_yprime_rate": summary.get("gt_covered_by_yprime_rate"),
        "yprime_annotation_supported_but_no_carrier_rate": summary.get("yprime_annotation_supported_but_no_carrier_rate"),
        "gt_to_carrier_coverage_gap": summary.get("gt_to_carrier_coverage_gap"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
