#!/usr/bin/env python3
"""
WSOVVIS hub score-source attribution audit.

Analysis-only script. It does not modify training/inference/checkpoint artifacts.

Purpose:
  For formal-aligned base_unobserved rows, decompose hub / wrong-extra failures into
  conservative source buckets using existing extra_mining_recall_diagnosis outputs.

Primary input expected:
  <run_root>/analysis/extra_mining_recall_diagnosis/<dataset>/<stage>/formal_aligned_row_diagnostics.jsonl

Optional companion files read when present:
  class_id_name_map_used.json
  formal_aligned_summary.json
  top_selected_extra_classes_named.json
  top_wrong_extra_winner_classes_named.json
  top_gt_suppressor_classes_named.json
  wrong_extra_hub_report_named.json

Outputs:
  <run_root>/analysis/hub_score_source_attribution/<dataset>/<stage>/summary.json
  <run_root>/analysis/hub_score_source_attribution/<dataset>/<stage>/source_bucket_summary.json
  <run_root>/analysis/hub_score_source_attribution/<dataset>/<stage>/hub_class_source_breakdown.json
  <run_root>/analysis/hub_score_source_attribution/<dataset>/<stage>/gt_class_source_breakdown.json
  <run_root>/analysis/hub_score_source_attribution/<dataset>/<stage>/row_source_attribution.jsonl
  <run_root>/analysis/hub_score_source_attribution/<dataset>/<stage>/top_examples.jsonl
  <run_root>/analysis/hub_score_source_attribution/<dataset>/<stage>/HUB_SCORE_SOURCE_ATTRIBUTION_TAKEOVER.md

Important:
  This script is deliberately conservative. It only assigns high-confidence
  same-trajectory / other-trajectory buckets when the required source trajectory
  fields are present in the row diagnostics. Otherwise it emits limited-confidence
  buckets based on margins/ranks/class frequencies and explicitly reports missing
  source fields. It never fabricates score-source evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SOURCE_BUCKETS = [
    "same_trajectory_confusion",
    "other_trajectory_hijack_or_cooccurrence_hub",
    "text_semantic_confusion",
    "class_specific_blind_spot",
    "near_miss_capacity_limited",
    "generic_global_hub",
    "insufficient_source_fields",
]

# Field aliases observed/anticipated across prior diagnosis scripts.
TRAJECTORY_ID_FIELDS = ["trajectory_id", "traj_id", "row_trajectory_id"]
CLIP_ID_FIELDS = ["clip_id", "video_id"]
GT_RAW_FIELDS = ["gt_raw_id", "matched_gt_raw_id", "matched_gt_raw_id_canonical", "gt_category_id"]
EXTRA_FIELDS = ["candidate_ids_extra", "candidate_ids_extra_raw", "active_extra_raw_ids"]
KNOWN_FIELDS = ["candidate_ids_known", "candidate_ids_yprime", "observed_raw_ids", "candidate_ids_Yprime"]
FAILURE_FIELDS = ["failure_bucket", "failure_mode", "primary_failure_bucket"]
WINNER_DOMAIN_FIELDS = ["winner_domain", "final_winner_domain", "pred_winner_domain"]
R_WINNER_DOMAIN_FIELDS = ["r_winner_domain", "R_winner_domain", "r_final_winner_domain"]
FINAL_WINNER_RAW_FIELDS = ["winner_raw_id", "final_winner_raw_id", "pred_raw_id", "top1_raw_id", "final_top1_raw_id"]
R_WINNER_RAW_FIELDS = ["r_winner_raw_id", "R_winner_raw_id", "r_final_winner_raw_id"]
SUPPRESSOR_RAW_FIELDS = [
    "gt_suppressor_raw_id",
    "suppressor_raw_id",
    "top_suppressor_raw_id",
    "best_suppressor_raw_id",
    "wrong_extra_winner_raw_id",
    "wrong_extra_raw_id",
]
SOURCE_TRAJ_FIELDS = [
    "hub_clip_argmax_trajectory_id",
    "suppressor_argmax_trajectory_id",
    "selected_extra_source_trajectory_id",
    "winner_source_trajectory_id",
    "clip_max_source_trajectory_id",
]
TEXT_COS_FIELDS = ["text_cos_gt_hub", "gt_hub_text_cos", "text_similarity_gt_suppressor"]
GT_MINING_RANK_FIELDS = ["gt_mining_rank", "best_gt_mining_rank", "mining_rank_gt"]
FINAL_GT_RANK_FIELDS = ["final_gt_rank", "gt_rank", "rank_of_gt_class"]
MARGIN_YPRIME_FIELDS = ["margin_gt_vs_Yprime", "gt_margin_vs_Yprime"]
MARGIN_WRONG_EXTRA_FIELDS = ["margin_gt_vs_wrong_extra", "gt_margin_vs_wrong_extra"]
MARGIN_OTHER_FIELDS = ["margin_gt_vs_other_nonYprime", "gt_margin_vs_other_nonYprime"]


def _first(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return default


def _as_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    return str(x)


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str) and not x.strip():
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None


def _as_set(xs: Any) -> set[str]:
    if xs is None:
        return set()
    if isinstance(xs, (str, int, float)):
        return {str(xs)}
    try:
        return {str(x) for x in xs}
    except Exception:
        return set()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_class_map(diag_dir: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    p = diag_dir / "class_id_name_map_used.json"
    if not p.exists():
        return mapping
    try:
        raw = _read_json(p)
    except Exception:
        return mapping

    def add(k: Any, v: Any) -> None:
        if k is None or v is None:
            return
        if isinstance(v, dict):
            name = v.get("name") or v.get("class_name") or v.get("category_name")
        else:
            name = v
        if name is not None:
            mapping[str(k)] = str(name)

    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict) and ("raw_id" in v or "id" in v):
                add(v.get("raw_id", v.get("id", k)), v)
            else:
                add(k, v)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                add(item.get("raw_id", item.get("id", item.get("category_id"))), item)
    return mapping


def _name(class_map: Dict[str, str], raw_id: Any) -> Optional[str]:
    if raw_id is None:
        return None
    return class_map.get(str(raw_id))


def _safe_div(a: float, b: float) -> Optional[float]:
    return a / b if b else None


def _quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


@dataclass
class RowAttribution:
    row: Dict[str, Any]
    trajectory_id: Optional[str]
    clip_id: Optional[str]
    gt_raw_id: Optional[str]
    gt_name: Optional[str]
    active_extra: List[str]
    candidate_known: List[str]
    gt_in_active_extra: bool
    failure_bucket: str
    final_winner_domain: Optional[str]
    r_winner_domain: Optional[str]
    final_winner_raw_id: Optional[str]
    final_winner_name: Optional[str]
    r_winner_raw_id: Optional[str]
    r_winner_name: Optional[str]
    suppressor_raw_id: Optional[str]
    suppressor_name: Optional[str]
    suppressor_source_trajectory_id: Optional[str]
    source_argmax_is_target: Optional[bool]
    text_cos_gt_hub: Optional[float]
    gt_mining_rank: Optional[int]
    final_gt_rank: Optional[int]
    margin_gt_vs_yprime: Optional[float]
    margin_gt_vs_wrong_extra: Optional[float]
    margin_gt_vs_other_nonyprime: Optional[float]
    source_bucket: str
    source_bucket_reason: str
    confidence: str
    missing_source_fields: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "clip_id": self.clip_id,
            "gt_raw_id": self.gt_raw_id,
            "gt_name": self.gt_name,
            "candidate_ids_extra_active": self.active_extra,
            "candidate_ids_known": self.candidate_known,
            "gt_in_active_extra": self.gt_in_active_extra,
            "failure_bucket": self.failure_bucket,
            "final_winner_domain": self.final_winner_domain,
            "r_winner_domain": self.r_winner_domain,
            "final_winner_raw_id": self.final_winner_raw_id,
            "final_winner_name": self.final_winner_name,
            "r_winner_raw_id": self.r_winner_raw_id,
            "r_winner_name": self.r_winner_name,
            "suppressor_raw_id": self.suppressor_raw_id,
            "suppressor_name": self.suppressor_name,
            "suppressor_source_trajectory_id": self.suppressor_source_trajectory_id,
            "source_argmax_is_target": self.source_argmax_is_target,
            "text_cos_gt_hub": self.text_cos_gt_hub,
            "gt_mining_rank": self.gt_mining_rank,
            "final_gt_rank": self.final_gt_rank,
            "margin_gt_vs_Yprime": self.margin_gt_vs_yprime,
            "margin_gt_vs_wrong_extra": self.margin_gt_vs_wrong_extra,
            "margin_gt_vs_other_nonYprime": self.margin_gt_vs_other_nonyprime,
            "source_bucket": self.source_bucket,
            "source_bucket_reason": self.source_bucket_reason,
            "confidence": self.confidence,
            "missing_source_fields": self.missing_source_fields,
        }


def _infer_suppressor(row: Dict[str, Any], gt: Optional[str], active_extra: set[str]) -> Optional[str]:
    direct = _as_str(_first(row, SUPPRESSOR_RAW_FIELDS))
    if direct is not None:
        return direct
    final_winner = _as_str(_first(row, FINAL_WINNER_RAW_FIELDS))
    if final_winner is not None and final_winner != gt:
        return final_winner
    r_winner = _as_str(_first(row, R_WINNER_RAW_FIELDS))
    if r_winner is not None and r_winner != gt:
        return r_winner
    # Conservative fallback: the first non-GT active extra, if any.
    for x in sorted(active_extra):
        if x != gt:
            return x
    return None


def _classify_row(
    row: Dict[str, Any],
    class_map: Dict[str, str],
    args: argparse.Namespace,
    selected_freq: Counter,
    suppressor_freq: Counter,
) -> RowAttribution:
    trajectory_id = _as_str(_first(row, TRAJECTORY_ID_FIELDS))
    clip_id = _as_str(_first(row, CLIP_ID_FIELDS))
    gt = _as_str(_first(row, GT_RAW_FIELDS))
    extra = _as_set(_first(row, EXTRA_FIELDS, []))
    known = _as_set(_first(row, KNOWN_FIELDS, []))
    active_extra = extra - known
    gt_in_active_extra = gt in active_extra if gt is not None else False

    failure_bucket = str(_first(row, FAILURE_FIELDS, "unknown"))
    final_domain = _as_str(_first(row, WINNER_DOMAIN_FIELDS))
    r_domain = _as_str(_first(row, R_WINNER_DOMAIN_FIELDS))
    final_winner = _as_str(_first(row, FINAL_WINNER_RAW_FIELDS))
    r_winner = _as_str(_first(row, R_WINNER_RAW_FIELDS))
    suppressor = _infer_suppressor(row, gt, active_extra)
    source_traj = _as_str(_first(row, SOURCE_TRAJ_FIELDS))
    source_is_target: Optional[bool] = None
    if source_traj is not None and trajectory_id is not None:
        source_is_target = source_traj == trajectory_id

    text_cos = _as_float(_first(row, TEXT_COS_FIELDS))
    gt_mining_rank = _as_int(_first(row, GT_MINING_RANK_FIELDS))
    final_gt_rank = _as_int(_first(row, FINAL_GT_RANK_FIELDS))
    m_y = _as_float(_first(row, MARGIN_YPRIME_FIELDS))
    m_we = _as_float(_first(row, MARGIN_WRONG_EXTRA_FIELDS))
    m_o = _as_float(_first(row, MARGIN_OTHER_FIELDS))

    missing_source_fields = []
    if source_traj is None:
        missing_source_fields.append("hub/suppressor source trajectory id")
    if text_cos is None:
        missing_source_fields.append("text cosine gt-hub")

    bucket = "insufficient_source_fields"
    reason = "source trajectory/text similarity fields absent; conservative fallback used"
    confidence = "limited"

    # High-confidence buckets first.
    if source_is_target is True and suppressor is not None:
        bucket = "same_trajectory_confusion"
        reason = "suppressor/hub score source trajectory equals target trajectory"
        confidence = "high"
    elif source_is_target is False and suppressor is not None:
        bucket = "other_trajectory_hijack_or_cooccurrence_hub"
        reason = "suppressor/hub score source trajectory differs from target trajectory"
        confidence = "high"
    elif text_cos is not None and text_cos >= args.text_alias_threshold:
        bucket = "text_semantic_confusion"
        reason = f"text cosine gt-hub >= threshold ({text_cos:.4f} >= {args.text_alias_threshold})"
        confidence = "medium"
    else:
        # Conservative evidence from formal diagnosis.
        # near miss: GT is just outside K; default K comes from args.primary_k.
        if gt_mining_rank is not None and args.primary_k < gt_mining_rank <= args.near_miss_k:
            bucket = "near_miss_capacity_limited"
            reason = f"gt mining rank {gt_mining_rank} is within near-miss band ({args.primary_k}+1..{args.near_miss_k})"
            confidence = "medium"
        elif gt_mining_rank is not None and gt_mining_rank > args.blind_spot_rank_threshold:
            bucket = "class_specific_blind_spot"
            reason = f"gt mining rank {gt_mining_rank} exceeds blind-spot threshold {args.blind_spot_rank_threshold}"
            confidence = "medium"
        elif suppressor is not None and (selected_freq[suppressor] >= args.global_hub_min_count or suppressor_freq[suppressor] >= args.global_hub_min_count):
            bucket = "generic_global_hub"
            reason = "suppressor appears as high-frequency selected/suppressor class"
            confidence = "medium"
        elif "gt_not_in_extra" in failure_bucket:
            bucket = "class_specific_blind_spot"
            reason = "GT not in active extra and no stronger source fields were available"
            confidence = "limited"
        elif "wrong_extra" in failure_bucket or final_domain == "extra":
            bucket = "generic_global_hub"
            reason = "wrong extra / extra-domain winner without source trajectory field"
            confidence = "limited"
        elif final_domain == "Yprime" or "Yprime" in failure_bucket:
            bucket = "other_trajectory_hijack_or_cooccurrence_hub"
            reason = "Yprime/observed wins without source trajectory field; treated as observed/cooccurrence attraction"
            confidence = "limited"

    return RowAttribution(
        row=row,
        trajectory_id=trajectory_id,
        clip_id=clip_id,
        gt_raw_id=gt,
        gt_name=_name(class_map, gt),
        active_extra=sorted(active_extra),
        candidate_known=sorted(known),
        gt_in_active_extra=gt_in_active_extra,
        failure_bucket=failure_bucket,
        final_winner_domain=final_domain,
        r_winner_domain=r_domain,
        final_winner_raw_id=final_winner,
        final_winner_name=_name(class_map, final_winner),
        r_winner_raw_id=r_winner,
        r_winner_name=_name(class_map, r_winner),
        suppressor_raw_id=suppressor,
        suppressor_name=_name(class_map, suppressor),
        suppressor_source_trajectory_id=source_traj,
        source_argmax_is_target=source_is_target,
        text_cos_gt_hub=text_cos,
        gt_mining_rank=gt_mining_rank,
        final_gt_rank=final_gt_rank,
        margin_gt_vs_yprime=m_y,
        margin_gt_vs_wrong_extra=m_we,
        margin_gt_vs_other_nonyprime=m_o,
        source_bucket=bucket,
        source_bucket_reason=reason,
        confidence=confidence,
        missing_source_fields=missing_source_fields,
    )


def _maybe_failure(row: Dict[str, Any], gt: Optional[str]) -> bool:
    fb = str(_first(row, FAILURE_FIELDS, "")).lower()
    if fb and "success" not in fb:
        return True
    top1 = row.get("final_top1_is_gt")
    if top1 is not None:
        return not bool(top1)
    winner = _as_str(_first(row, FINAL_WINNER_RAW_FIELDS))
    if gt is not None and winner is not None:
        return winner != gt
    return False


def _build_freqs(rows: List[Dict[str, Any]]) -> Tuple[Counter, Counter]:
    selected = Counter()
    suppressors = Counter()
    for r in rows:
        extra = _as_set(_first(r, EXTRA_FIELDS, [])) - _as_set(_first(r, KNOWN_FIELDS, []))
        selected.update(extra)
        gt = _as_str(_first(r, GT_RAW_FIELDS))
        sup = _infer_suppressor(r, gt, extra)
        if sup is not None:
            suppressors[sup] += 1
    return selected, suppressors


def _counter_to_named(counter: Counter, class_map: Dict[str, str], top_n: int) -> List[Dict[str, Any]]:
    return [
        {"raw_id": k, "name": _name(class_map, k), "count": v}
        for k, v in counter.most_common(top_n)
    ]


def _summarize_by_key(attrs: List[RowAttribution], key_fn) -> List[Dict[str, Any]]:
    groups: Dict[str, List[RowAttribution]] = defaultdict(list)
    for a in attrs:
        k = key_fn(a)
        if k is not None:
            groups[str(k)].append(a)
    out = []
    for k, vals in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        buckets = Counter(v.source_bucket for v in vals)
        conf = Counter(v.confidence for v in vals)
        names = Counter(v.suppressor_name or v.suppressor_raw_id or "<none>" for v in vals)
        ranks = [v.gt_mining_rank for v in vals if v.gt_mining_rank is not None]
        out.append({
            "key": k,
            "count": len(vals),
            "source_bucket_histogram": dict(buckets.most_common()),
            "confidence_histogram": dict(conf.most_common()),
            "top_suppressors": dict(names.most_common(10)),
            "mean_gt_mining_rank": statistics.mean(ranks) if ranks else None,
            "median_gt_mining_rank": statistics.median(ranks) if ranks else None,
        })
    return out


def run(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).resolve()
    diag_dir = Path(args.diagnosis_dir) if args.diagnosis_dir else run_root / "analysis" / "extra_mining_recall_diagnosis" / args.dataset_name / args.stage
    row_path = diag_dir / "formal_aligned_row_diagnostics.jsonl"
    out_dir = Path(args.output_dir) if args.output_dir else run_root / "analysis" / "hub_score_source_attribution" / args.dataset_name / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)

    if not row_path.exists():
        blocked = {
            "status": "BLOCKED_SOURCE_ATTRIBUTION_INCOMPLETE",
            "reason": "formal_aligned_row_diagnostics.jsonl not found",
            "expected_path": str(row_path),
            "run_root": str(run_root),
        }
        _write_json(out_dir / "summary.json", blocked)
        (out_dir / "HUB_SCORE_SOURCE_ATTRIBUTION_TAKEOVER.md").write_text(
            "# Hub Score-Source Attribution\n\n"
            "Status: BLOCKED_SOURCE_ATTRIBUTION_INCOMPLETE\n\n"
            f"Missing required row authority: `{row_path}`\n",
            encoding="utf-8",
        )
        print(json.dumps(blocked, indent=2, ensure_ascii=False))
        return 2

    class_map = _load_class_map(diag_dir)
    rows = list(_read_jsonl(row_path))
    selected_freq, suppressor_freq = _build_freqs(rows)

    formal_rows = []
    for r in rows:
        # formal_aligned_row_diagnostics is already formal row authority. If a
        # split field is present, honor --formal_split; otherwise keep all rows.
        split = _first(r, ["split", "split_name", "formal_split"])
        if split is not None and str(split) != args.formal_split:
            continue
        formal_rows.append(r)

    target_rows = []
    for r in formal_rows:
        gt = _as_str(_first(r, GT_RAW_FIELDS))
        if args.failures_only and not _maybe_failure(r, gt):
            continue
        target_rows.append(r)

    attrs = [_classify_row(r, class_map, args, selected_freq, suppressor_freq) for r in target_rows]

    bucket_counter = Counter(a.source_bucket for a in attrs)
    conf_counter = Counter(a.confidence for a in attrs)
    failure_counter = Counter(a.failure_bucket for a in attrs)
    raw_contains = sum(a.gt_in_active_extra for a in attrs)
    high_conf = sum(a.confidence == "high" for a in attrs)
    medium_conf = sum(a.confidence == "medium" for a in attrs)
    limited_conf = sum(a.confidence == "limited" for a in attrs)

    source_field_present = sum(a.suppressor_source_trajectory_id is not None for a in attrs)
    text_cos_present = sum(a.text_cos_gt_hub is not None for a in attrs)

    ranks = [a.gt_mining_rank for a in attrs if a.gt_mining_rank is not None]
    final_ranks = [a.final_gt_rank for a in attrs if a.final_gt_rank is not None]

    summary = {
        "status": "PASS" if attrs else "NO_TARGET_ROWS",
        "audit_type": "hub_score_source_attribution",
        "run_root": str(run_root),
        "dataset_name": args.dataset_name,
        "stage": args.stage,
        "formal_split": args.formal_split,
        "row_authority": str(row_path),
        "output_dir": str(out_dir),
        "training_modified": False,
        "iou_recomputed": False,
        "rle_decode_used": False,
        "full_vocab_rescoring_used": False,
        "oracle_gt_used_as_policy_input": False,
        "formal_row_count": len(formal_rows),
        "target_row_count": len(attrs),
        "failures_only": args.failures_only,
        "gt_in_active_extra_count": raw_contains,
        "gt_in_active_extra_rate": _safe_div(raw_contains, len(attrs)),
        "source_bucket_histogram": dict(bucket_counter.most_common()),
        "confidence_histogram": dict(conf_counter.most_common()),
        "failure_bucket_histogram": dict(failure_counter.most_common()),
        "source_trajectory_field_present_count": source_field_present,
        "source_trajectory_field_present_rate": _safe_div(source_field_present, len(attrs)),
        "text_cos_field_present_count": text_cos_present,
        "text_cos_field_present_rate": _safe_div(text_cos_present, len(attrs)),
        "confidence_counts": {
            "high": high_conf,
            "medium": medium_conf,
            "limited": limited_conf,
        },
        "mean_gt_mining_rank": statistics.mean(ranks) if ranks else None,
        "median_gt_mining_rank": statistics.median(ranks) if ranks else None,
        "mean_final_gt_rank": statistics.mean(final_ranks) if final_ranks else None,
        "median_final_gt_rank": statistics.median(final_ranks) if final_ranks else None,
        "top_selected_extra_classes": _counter_to_named(selected_freq, class_map, args.top_classes),
        "top_suppressor_classes": _counter_to_named(suppressor_freq, class_map, args.top_classes),
        "interpretation_guardrail": (
            "High-confidence same/other-trajectory attribution requires source trajectory fields. "
            "When absent, rows are labeled with medium/limited confidence using existing formal diagnosis fields only."
        ),
    }

    source_bucket_summary = []
    for bucket, count in bucket_counter.most_common():
        vals = [a for a in attrs if a.source_bucket == bucket]
        sup = Counter(a.suppressor_name or a.suppressor_raw_id or "<none>" for a in vals)
        gt = Counter(a.gt_name or a.gt_raw_id or "<none>" for a in vals)
        bucket_ranks = [a.gt_mining_rank for a in vals if a.gt_mining_rank is not None]
        source_bucket_summary.append({
            "source_bucket": bucket,
            "count": count,
            "rate": _safe_div(count, len(attrs)),
            "confidence_histogram": dict(Counter(a.confidence for a in vals).most_common()),
            "top_suppressors": dict(sup.most_common(args.top_classes)),
            "top_gt_classes": dict(gt.most_common(args.top_classes)),
            "mean_gt_mining_rank": statistics.mean(bucket_ranks) if bucket_ranks else None,
            "median_gt_mining_rank": statistics.median(bucket_ranks) if bucket_ranks else None,
        })

    hub_breakdown = _summarize_by_key(attrs, lambda a: a.suppressor_name or a.suppressor_raw_id)
    gt_breakdown = _summarize_by_key(attrs, lambda a: a.gt_name or a.gt_raw_id)

    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "source_bucket_summary.json", {"rows": source_bucket_summary})
    _write_json(out_dir / "hub_class_source_breakdown.json", {"rows": hub_breakdown})
    _write_json(out_dir / "gt_class_source_breakdown.json", {"rows": gt_breakdown})
    _write_jsonl(out_dir / "row_source_attribution.jsonl", (a.to_json() for a in attrs))

    # Compact top examples: prioritize high-confidence, then medium, then limited; within bucket by bad rank/margin.
    def example_sort(a: RowAttribution) -> Tuple[int, int, float]:
        conf_rank = {"high": 0, "medium": 1, "limited": 2}.get(a.confidence, 3)
        rank = a.gt_mining_rank if a.gt_mining_rank is not None else 10**9
        margin = a.margin_gt_vs_wrong_extra if a.margin_gt_vs_wrong_extra is not None else 0.0
        return (conf_rank, -rank, margin)

    examples = sorted(attrs, key=example_sort)[: args.top_examples]
    _write_jsonl(out_dir / "top_examples.jsonl", (a.to_json() for a in examples))

    takeover = [
        "# Hub Score-Source Attribution Takeover",
        "",
        f"Status: {summary['status']}",
        f"Run root: `{run_root}`",
        f"Dataset/stage: `{args.dataset_name}` / `{args.stage}`",
        f"Formal split: `{args.formal_split}`",
        f"Formal rows: {len(formal_rows)}",
        f"Target rows: {len(attrs)}",
        "",
        "## Source bucket histogram",
    ]
    for k, v in bucket_counter.most_common():
        takeover.append(f"- {k}: {v} ({_safe_div(v, len(attrs))})")
    takeover += [
        "",
        "## Confidence histogram",
    ]
    for k, v in conf_counter.most_common():
        takeover.append(f"- {k}: {v} ({_safe_div(v, len(attrs))})")
    takeover += [
        "",
        "## Guardrail",
        "This audit does not modify training outputs, recompute IoU, decode RLE, or full-vocab rescore. "
        "Rows without source trajectory fields are not used to claim exact same/other-trajectory causality.",
        "",
        "## Output files",
        "- `summary.json`",
        "- `source_bucket_summary.json`",
        "- `hub_class_source_breakdown.json`",
        "- `gt_class_source_breakdown.json`",
        "- `row_source_attribution.jsonl`",
        "- `top_examples.jsonl`",
    ]
    (out_dir / "HUB_SCORE_SOURCE_ATTRIBUTION_TAKEOVER.md").write_text("\n".join(takeover) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "output_dir": str(out_dir),
        "formal_row_count": len(formal_rows),
        "target_row_count": len(attrs),
        "source_bucket_histogram": summary["source_bucket_histogram"],
        "confidence_histogram": summary["confidence_histogram"],
    }, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WSOVVIS hub score-source attribution audit")
    p.add_argument("--run_root", required=True)
    p.add_argument("--runtime_output_root", default=None, help="Accepted for CLI compatibility; not modified")
    p.add_argument("--repo_root", default=None, help="Accepted for CLI compatibility; not modified")
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--stage", default="softem_aug")
    p.add_argument("--trajectory_source_branch", default="mainline", help="Accepted for CLI compatibility")
    p.add_argument("--device", default="cpu", help="Accepted for CLI compatibility; this audit is CPU-only")
    p.add_argument("--formal_split", default="base_unobserved")
    p.add_argument("--diagnosis_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--top_examples", type=int, default=256)
    p.add_argument("--top_classes", type=int, default=50)
    p.add_argument("--primary_k", type=int, default=3)
    p.add_argument("--near_miss_k", type=int, default=10)
    p.add_argument("--blind_spot_rank_threshold", type=int, default=20)
    p.add_argument("--text_alias_threshold", type=float, default=0.65)
    p.add_argument("--global_hub_min_count", type=int, default=50)
    p.add_argument("--include_success", action="store_true", help="Audit all formal rows instead of failure rows only")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.failures_only = not args.include_success
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
