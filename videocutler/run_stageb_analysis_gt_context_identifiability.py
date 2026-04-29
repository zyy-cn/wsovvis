#!/usr/bin/env python3
"""
Read-only GT context identifiability audit for WS-OVVIS / LV-VIS.

This audit estimates an information upper-bound for cross-clip consistency from
GT label context only, then optionally joins per-class scorer/proposal summaries.

It does not train, infer, or modify model artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


K_DEFAULTS = (1, 2, 3, 5, 10, 20, 50)
EPS = 1e-12


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


def _canonical_context_key(vals: Iterable[int]) -> str:
    return ";".join(str(v) for v in sorted(vals))


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in counter.values():
        p = v / total
        ent -= p * math.log(p + EPS, 2)
    return ent


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


def _load_annotation_contexts(annotation_json: Path) -> Tuple[Dict[int, Set[int]], Dict[int, int], Dict[int, Counter], Dict[int, str], Dict[int, int], Dict[int, int], Dict[str, Any]]:
    """Return video_id -> class set, video_id -> video_id, class -> video counter, class names,
    class instance counts, class clip-pair counts, and metadata.
    """
    obj = _json_load(annotation_json)
    categories = obj.get("categories") or []
    cat_names: Dict[int, str] = {}
    for c in categories:
        cid = _as_int(c.get("id") or c.get("category_id"))
        if cid is not None:
            cat_names[cid] = str(c.get("name") or c.get("synonyms") or cid)

    video_id_to_video_key: Dict[int, int] = {}
    videos = obj.get("videos") or []
    for v in videos:
        vid = _as_int(v.get("id") or v.get("video_id"))
        if vid is not None:
            video_id_to_video_key[vid] = vid

    contexts: Dict[int, Set[int]] = defaultdict(set)
    class_instance_count: Counter = Counter()
    class_video_counter: Dict[int, Counter] = defaultdict(Counter)
    ann_count = 0
    ann_no_video = 0
    ann_no_class = 0

    for ann in obj.get("annotations") or []:
        ann_count += 1
        cid = _as_int(ann.get("category_id") or ann.get("raw_id") or ann.get("class_id"))
        vid = _as_int(ann.get("video_id") or ann.get("clip_id") or ann.get("image_id"))
        if cid is None:
            ann_no_class += 1
            continue
        if vid is None:
            ann_no_video += 1
            continue
        contexts[vid].add(cid)
        class_instance_count[cid] += 1
        vkey = video_id_to_video_key.get(vid, vid)
        class_video_counter[cid][vkey] += 1

    class_clip_pair_count: Counter = Counter()
    for _vid, cls_set in contexts.items():
        for cid in cls_set:
            class_clip_pair_count[cid] += 1

    meta = {
        "annotation_path": str(annotation_json),
        "annotation_sha256_prefix": _sha256_file(annotation_json, max_bytes=1 << 20)[:16],
        "video_count_with_context": len(contexts),
        "annotation_count": ann_count,
        "annotation_without_video_id": ann_no_video,
        "annotation_without_category_id": ann_no_class,
        "category_name_count": len(cat_names),
    }
    return dict(contexts), video_id_to_video_key, class_video_counter, cat_names, dict(class_instance_count), dict(class_clip_pair_count), meta


def _load_optional_csv(path: Optional[Path]) -> Dict[int, Dict[str, str]]:
    if not path or not path.exists():
        return {}
    out: Dict[int, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = None
            for key in ("raw_id", "class_id", "category_id", "gt_raw_id", "matched_gt_raw_id"):
                raw = _as_int(row.get(key))
                if raw is not None:
                    break
            if raw is not None:
                out[raw] = row
    return out


def _pick_join(row: Dict[str, str], keys: Sequence[str]) -> str:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return ""


def _greedy_min_k_identifiable(contexts: List[Set[int]], target: int, all_context_union: Set[int]) -> Tuple[Optional[int], str]:
    if not contexts:
        return None, "none"
    final_intersection = set.intersection(*[set(x) for x in contexts]) if contexts else set()
    if final_intersection != {target}:
        return None, "impossible_final_intersection_not_singleton"
    confounders = set(all_context_union) - {target}
    if not confounders:
        return 1, "trivial_no_confounders"
    # Exact brute force for small clip counts and small confounder universe.
    n = len(contexts)
    if n <= 18 and len(confounders) <= 48:
        import itertools
        for k in range(1, n + 1):
            for idxs in itertools.combinations(range(n), k):
                inter = set(contexts[idxs[0]])
                for idx in idxs[1:]:
                    inter &= contexts[idx]
                if inter == {target}:
                    return k, "exact"
    # Greedy set cover: choose contexts that eliminate most remaining confounders.
    remaining = set(confounders)
    selected = 0
    used: Set[int] = set()
    while remaining:
        best_i = None
        best_gain = -1
        for i, ctx in enumerate(contexts):
            if i in used:
                continue
            gain = len(remaining - set(ctx))
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i is None or best_gain <= 0:
            return None, "greedy_failed"
        selected += 1
        used.add(best_i)
        remaining -= (remaining - set(contexts[best_i]))
    return selected, "greedy"


def _bootstrap_rates(contexts: List[Set[int]], target: int, ks: Sequence[int], repeats: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    n = len(contexts)
    res: Dict[str, Any] = {}
    if n == 0:
        for k in ks:
            res[f"bootstrap_identifiable_rate@{k}"] = ""
            res[f"median_intersection_size@{k}"] = ""
        return res
    for k0 in ks:
        k = min(int(k0), n)
        if k <= 0:
            continue
        ok = 0
        sizes: List[int] = []
        trials = max(1, repeats)
        for _ in range(trials):
            sample = rng.sample(contexts, k) if k < n else contexts
            inter = set(sample[0])
            for s in sample[1:]:
                inter &= s
            sizes.append(len(inter))
            if inter == {target}:
                ok += 1
        sizes.sort()
        med = sizes[len(sizes) // 2]
        res[f"bootstrap_identifiable_rate@{k0}"] = ok / trials
        res[f"median_intersection_size@{k0}"] = med
    return res


def _support_tier(clip_count: int) -> str:
    if clip_count <= 0:
        return "none"
    if clip_count == 1:
        return "singleton"
    if clip_count <= 3:
        return "very_low"
    if clip_count <= 9:
        return "low"
    if clip_count <= 29:
        return "mid"
    return "high"


def _context_stats_for_target(
    target: int,
    contexts_by_clip: Dict[int, Set[int]],
    target_context: Dict[int, Set[int]],
    cat_names: Dict[int, str],
    split_type: str,
    base_set: Set[int],
    novel_set: Set[int],
    instance_count: Dict[int, int],
    class_video_counter: Dict[int, Counter],
    scorer_join: Dict[int, Dict[str, str]],
    proposal_join: Dict[int, Dict[str, str]],
    ks: Sequence[int],
    bootstrap_repeats: int,
    seed: int,
    context_label: str,
) -> Dict[str, Any]:
    clip_ids = sorted([clip for clip, cls in target_context.items() if target in cls])
    contexts = [set(target_context[clip]) for clip in clip_ids]
    clip_count = len(clip_ids)
    video_counter = class_video_counter.get(target, Counter())
    video_count = len(video_counter)
    inst = int(instance_count.get(target, 0))
    context_keys = Counter(_canonical_context_key(ctx) for ctx in contexts)
    unique_context_count = len(context_keys)
    dominant_context_rate = _safe_div(max(context_keys.values()) if context_keys else 0, clip_count)
    context_entropy = _entropy(context_keys)
    context_sizes = [len(ctx) for ctx in contexts]
    mean_context_size = sum(context_sizes) / len(context_sizes) if context_sizes else 0.0
    min_context_size = min(context_sizes) if context_sizes else 0

    if contexts:
        inter = set(contexts[0])
        union = set(contexts[0])
        for ctx in contexts[1:]:
            inter &= ctx
            union |= ctx
    else:
        inter = set()
        union = set()

    is_identifiable = (inter == {target}) and clip_count >= 2
    singleton_underdetermined = clip_count == 1
    persistent = sorted([x for x in inter if x != target])

    co_counter: Counter = Counter()
    for ctx in contexts:
        for x in ctx:
            if x != target:
                co_counter[x] += 1
    cooccur_class_count = len(co_counter)
    max_p = max((_safe_div(v, clip_count) for v in co_counter.values()), default=0.0)
    def _conf(th: float) -> List[int]:
        return sorted([cid for cid, v in co_counter.items() if _safe_div(v, clip_count) >= th])
    near = _conf(0.90)
    strong = _conf(0.75)
    moderate = _conf(0.50)
    top_co = [
        {
            "raw_id": cid,
            "name": cat_names.get(cid, str(cid)),
            "p_given_target": _safe_div(v, clip_count),
            "count": int(v),
            "split": "base" if cid in base_set else "novel" if cid in novel_set else "other",
        }
        for cid, v in co_counter.most_common(12)
    ]

    min_k, min_k_method = _greedy_min_k_identifiable(contexts, target, union)
    boot = _bootstrap_rates(contexts, target, ks, bootstrap_repeats, seed + target)

    scorer = scorer_join.get(target, {})
    prop = proposal_join.get(target, {})
    row: Dict[str, Any] = {
        "context_label": context_label,
        "raw_id": target,
        "class_name": cat_names.get(target, str(target)),
        "split_type": split_type,
        "clip_count": clip_count,
        "video_count": video_count,
        "instance_count": inst,
        "clips_per_video_mean": _safe_div(clip_count, video_count),
        "unique_context_count": unique_context_count,
        "context_entropy": context_entropy,
        "dominant_context_rate": dominant_context_rate,
        "mean_context_size": mean_context_size,
        "min_context_size": min_context_size,
        "support_tier": _support_tier(clip_count),
        "cooccur_class_count": cooccur_class_count,
        "max_p_cooccur": max_p,
        "intersection_size": len(inter),
        "intersection_classes": json.dumps(sorted(inter), ensure_ascii=False),
        "intersection_class_names": json.dumps([cat_names.get(x, str(x)) for x in sorted(inter)], ensure_ascii=False),
        "is_identifiable": bool(is_identifiable),
        "singleton_underdetermined": bool(singleton_underdetermined),
        "persistent_confounder_count": len(persistent),
        "persistent_confounders": json.dumps(persistent, ensure_ascii=False),
        "persistent_confounder_names": json.dumps([cat_names.get(x, str(x)) for x in persistent], ensure_ascii=False),
        "near_persistent_confounder_count": len([x for x in near if x != target]),
        "strong_confounder_count": len([x for x in strong if x != target]),
        "moderate_confounder_count": len([x for x in moderate if x != target]),
        "top_cooccur_classes": json.dumps(top_co, ensure_ascii=False),
        "novel_persistent_confounder_count": len([x for x in persistent if x in novel_set]),
        "novel_near_persistent_confounder_count": len([x for x in near if x in novel_set and x != target]),
        "base_persistent_confounder_count": len([x for x in persistent if x in base_set]),
        "base_near_persistent_confounder_count": len([x for x in near if x in base_set and x != target]),
        "min_k_identifiable": "" if min_k is None else min_k,
        "min_k_method": min_k_method,
        "stable_k_identifiable": "" if min_k is None else min_k,
        # Optional D-scorer join fields. Keep names canonical even if empty.
        "D_gt_rank1_rate": _pick_join(scorer, ["D_gt_rank1_rate", "gt_rank1_rate", "gt_top1_hit_rate", "rank1_rate"]),
        "D_gt_top5_rate": _pick_join(scorer, ["D_gt_top5_rate", "gt_top5_rate", "top5_rate"]),
        "D_mean_gt_rank": _pick_join(scorer, ["D_mean_gt_rank", "mean_gt_rank", "mean_rank", "mean_text_gt_rank_full_vocab"]),
        "D_large_margin_wrong_rate": _pick_join(scorer, ["D_large_margin_wrong_rate", "wrong_large_negative_margin_rate"]),
        "D_top_suppressor_classes": _pick_join(scorer, ["D_top_suppressor_classes", "top_suppressor_classes", "wrong_top1_label_top_counts"]),
        # Optional proposal/support join fields.
        "videocutler_support_rate": _pick_join(prop, ["videocutler_support_rate", "oracle_support_rate_at_0_5", "clip_yprime_support_rate_at_0_5"]),
        "sidecar_support_rate": _pick_join(prop, ["sidecar_support_rate", "yprime_trajectory_support_rate", "sidecar_support_rate_at_0_5"]),
        "proposal_support_gap": _pick_join(prop, ["proposal_support_gap", "sidecar_oracle_gap", "gt_to_carrier_coverage_gap"]),
    }
    row.update(boot)
    return row


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _make_contexts(original_contexts: Dict[int, Set[int]], allowed: Set[int]) -> Dict[int, Set[int]]:
    return {clip: set(cls & allowed) for clip, cls in original_contexts.items() if cls & allowed}


def _aggregate_summary(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    n = len(rows)
    auditable = [r for r in rows if int(r.get("clip_count") or 0) >= 3]
    aud_n = len(auditable)
    ident = [r for r in auditable if str(r.get("is_identifiable")).lower() == "true"]
    low_support = [r for r in rows if str(r.get("support_tier")) in ("none", "singleton", "very_low", "low")]
    near_hub = [r for r in auditable if int(r.get("near_persistent_confounder_count") or 0) > 0]
    min_ks = [int(r["min_k_identifiable"]) for r in auditable if str(r.get("min_k_identifiable", "")).isdigit()]
    return {
        "context_label": label,
        "class_count": n,
        "auditable_class_count_clip_ge_3": aud_n,
        "identifiable_class_count": len(ident),
        "identifiable_class_rate_among_auditable": _safe_div(len(ident), aud_n),
        "context_entangled_class_count_among_auditable": aud_n - len(ident),
        "context_entangled_class_rate_among_auditable": _safe_div(aud_n - len(ident), aud_n),
        "low_support_class_count": len(low_support),
        "low_support_class_rate": _safe_div(len(low_support), n),
        "near_persistent_confounder_class_count": len(near_hub),
        "near_persistent_confounder_class_rate_among_auditable": _safe_div(len(near_hub), aud_n),
        "mean_min_k_identifiable": sum(min_ks) / len(min_ks) if min_ks else None,
        "median_min_k_identifiable": sorted(min_ks)[len(min_ks)//2] if min_ks else None,
    }


def _hub_confounder_summary(rows: List[Dict[str, Any]], cat_names: Dict[int, str], base_set: Set[int], novel_set: Set[int]) -> List[Dict[str, Any]]:
    agg: Dict[int, Dict[str, Any]] = {}
    def _get(cid: int) -> Dict[str, Any]:
        if cid not in agg:
            agg[cid] = {
                "hub_raw_id": cid,
                "hub_name": cat_names.get(cid, str(cid)),
                "split": "base" if cid in base_set else "novel" if cid in novel_set else "other",
                "persistent_confounded_target_count": 0,
                "near_persistent_confounded_target_count": 0,
                "strong_confounded_target_count": 0,
                "targets_persistent": [],
                "targets_near_persistent": [],
            }
        return agg[cid]
    for r in rows:
        tgt = int(r["raw_id"])
        try:
            pers = json.loads(r.get("persistent_confounders") or "[]")
        except Exception:
            pers = []
        try:
            top = json.loads(r.get("top_cooccur_classes") or "[]")
        except Exception:
            top = []
        for cid in pers:
            h = _get(int(cid))
            h["persistent_confounded_target_count"] += 1
            h["targets_persistent"].append(tgt)
        for item in top:
            cid = int(item.get("raw_id"))
            p = float(item.get("p_given_target") or 0.0)
            if p >= 0.90:
                h = _get(cid)
                h["near_persistent_confounded_target_count"] += 1
                h["targets_near_persistent"].append(tgt)
            if p >= 0.75:
                _get(cid)["strong_confounded_target_count"] += 1
    out = []
    for cid, h in agg.items():
        h = dict(h)
        h["targets_persistent"] = json.dumps(sorted(set(h["targets_persistent"])), ensure_ascii=False)
        h["targets_near_persistent"] = json.dumps(sorted(set(h["targets_near_persistent"])), ensure_ascii=False)
        out.append(h)
    out.sort(key=lambda x: (int(x["near_persistent_confounded_target_count"]), int(x["persistent_confounded_target_count"])), reverse=True)
    return out


def _assign_failure_bucket(base_row: Optional[Dict[str, Any]], all_row: Optional[Dict[str, Any]]) -> str:
    if base_row is None:
        return "missing_base_row"
    clip_count = int(base_row.get("clip_count") or 0)
    if clip_count < 3:
        return "low_support_undetermined"
    base_ident = str(base_row.get("is_identifiable")).lower() == "true"
    all_ident = str(all_row.get("is_identifiable")).lower() == "true" if all_row else False
    # Optional scorer classification if available.
    d_rank1 = base_row.get("D_gt_rank1_rate") or (all_row or {}).get("D_gt_rank1_rate", "")
    scorer_known = d_rank1 not in (None, "")
    scorer_success = False
    if scorer_known:
        try:
            scorer_success = float(d_rank1) >= 0.5
        except Exception:
            scorer_known = False
    if base_ident and all_ident and scorer_success:
        return "context_identifiable_scorer_success"
    if base_ident and all_ident and scorer_known and not scorer_success:
        return "context_identifiable_scorer_fail"
    if base_ident and not all_ident:
        return "base_identifiable_novel_or_all_context_confounded"
    if not base_ident and scorer_known and not scorer_success:
        return "context_entangled_scorer_fail"
    if not base_ident and scorer_success:
        return "context_entangled_scorer_success"
    if base_ident:
        return "context_identifiable_scorer_unknown"
    return "context_entangled_scorer_unknown"


def _write_markdown(path: Path, summary: Dict[str, Any], failure_counts: Counter, output_dir: Path) -> None:
    lines = []
    lines.append("# GT Context Identifiability Audit")
    lines.append("")
    lines.append("## Status")
    lines.append(f"- status: `{summary.get('status')}`")
    lines.append(f"- output_dir: `{output_dir}`")
    lines.append("")
    lines.append("## Split / Input")
    split = summary.get("split", {})
    ann = summary.get("annotation", {})
    lines.append(f"- split_path: `{split.get('split_path')}`")
    lines.append(f"- base_count: `{split.get('base_count')}`")
    lines.append(f"- novel_count: `{split.get('novel_count')}`")
    lines.append(f"- annotation_path: `{ann.get('annotation_path')}`")
    lines.append(f"- video_count_with_context: `{ann.get('video_count_with_context')}`")
    lines.append("")
    lines.append("## Key Summary")
    for s in summary.get("context_summaries", []):
        lines.append(f"### {s.get('context_label')}")
        lines.append(f"- class_count: `{s.get('class_count')}`")
        lines.append(f"- auditable_class_count_clip_ge_3: `{s.get('auditable_class_count_clip_ge_3')}`")
        lines.append(f"- identifiable_rate_among_auditable: `{s.get('identifiable_class_rate_among_auditable')}`")
        lines.append(f"- context_entangled_rate_among_auditable: `{s.get('context_entangled_class_rate_among_auditable')}`")
        lines.append(f"- low_support_class_rate: `{s.get('low_support_class_rate')}`")
        lines.append(f"- near_persistent_confounder_rate_among_auditable: `{s.get('near_persistent_confounder_class_rate_among_auditable')}`")
        lines.append("")
    lines.append("## Base Target Failure Buckets")
    for k, v in failure_counts.most_common():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Files")
    for name in [
        "summary.json",
        "per_class_context_identifiability.csv",
        "hub_confounder_summary.csv",
        "context_identifiability_by_split.csv",
        "bootstrap_identifiability_curves.csv",
        "failure_bucket_summary.csv",
    ]:
        lines.append(f"- `{name}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    split_path = Path(args.split_json)
    annotation_path = Path(args.annotation_json)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        run_root = Path(args.run_root)
        out_dir = run_root / "analysis" / "gt_context_identifiability" / args.dataset_name
    _ensure_dir(out_dir)

    base_set, novel_set, split_meta = _parse_split(split_path)
    if args.require_official_counts and (len(base_set) != 641 or len(novel_set) != 555):
        raise SystemExit(f"FAIL_SPLIT_AUTHORITY: expected 641/555, got {len(base_set)}/{len(novel_set)}")
    all_known = set(base_set | novel_set)

    contexts_raw, video_map, class_video_counter, cat_names, instance_count, class_clip_count, ann_meta = _load_annotation_contexts(annotation_path)
    base_contexts = _make_contexts(contexts_raw, base_set)
    all_contexts = _make_contexts(contexts_raw, all_known)

    scorer_join = _load_optional_csv(Path(args.d_scorer_per_class_csv) if args.d_scorer_per_class_csv else None)
    proposal_join = _load_optional_csv(Path(args.proposal_support_per_class_csv) if args.proposal_support_per_class_csv else None)
    ks = tuple(int(x) for x in args.bootstrap_ks.split(",") if x.strip())

    rows: List[Dict[str, Any]] = []
    base_only_rows: Dict[int, Dict[str, Any]] = {}
    all_visible_base_rows: Dict[int, Dict[str, Any]] = {}
    novel_rows: Dict[int, Dict[str, Any]] = {}

    # Base-only target/context.
    for target in sorted(base_set):
        r = _context_stats_for_target(
            target=target,
            contexts_by_clip=base_contexts,
            target_context=base_contexts,
            cat_names=cat_names,
            split_type="base",
            base_set=base_set,
            novel_set=novel_set,
            instance_count=instance_count,
            class_video_counter=class_video_counter,
            scorer_join=scorer_join,
            proposal_join=proposal_join,
            ks=ks,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed,
            context_label="base_only",
        )
        base_only_rows[target] = r
        rows.append(r)

    # Base target, all-visible context.
    for target in sorted(base_set):
        r = _context_stats_for_target(
            target=target,
            contexts_by_clip=all_contexts,
            target_context=all_contexts,
            cat_names=cat_names,
            split_type="base",
            base_set=base_set,
            novel_set=novel_set,
            instance_count=instance_count,
            class_video_counter=class_video_counter,
            scorer_join=scorer_join,
            proposal_join=proposal_join,
            ks=ks,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed,
            context_label="base_target_all_visible_context",
        )
        all_visible_base_rows[target] = r
        rows.append(r)

    # Novel target audit-only under all-visible context.
    if not args.skip_novel_audit:
        for target in sorted(novel_set):
            r = _context_stats_for_target(
                target=target,
                contexts_by_clip=all_contexts,
                target_context=all_contexts,
                cat_names=cat_names,
                split_type="novel",
                base_set=base_set,
                novel_set=novel_set,
                instance_count=instance_count,
                class_video_counter=class_video_counter,
                scorer_join=scorer_join,
                proposal_join=proposal_join,
                ks=ks,
                bootstrap_repeats=args.bootstrap_repeats,
                seed=args.seed,
                context_label="novel_audit_only_all_visible_context",
            )
            novel_rows[target] = r
            rows.append(r)

    # Failure buckets for base target by comparing base-only vs all-visible.
    failure_rows: List[Dict[str, Any]] = []
    failure_counts: Counter = Counter()
    for target in sorted(base_set):
        bucket = _assign_failure_bucket(base_only_rows.get(target), all_visible_base_rows.get(target))
        failure_counts[bucket] += 1
        failure_rows.append({
            "raw_id": target,
            "class_name": cat_names.get(target, str(target)),
            "failure_bucket": bucket,
            "base_only_identifiable": base_only_rows.get(target, {}).get("is_identifiable", ""),
            "all_visible_identifiable": all_visible_base_rows.get(target, {}).get("is_identifiable", ""),
            "base_clip_count": base_only_rows.get(target, {}).get("clip_count", ""),
            "base_intersection_size": base_only_rows.get(target, {}).get("intersection_size", ""),
            "all_visible_intersection_size": all_visible_base_rows.get(target, {}).get("intersection_size", ""),
            "novel_persistent_confounder_count": all_visible_base_rows.get(target, {}).get("novel_persistent_confounder_count", ""),
            "novel_near_persistent_confounder_count": all_visible_base_rows.get(target, {}).get("novel_near_persistent_confounder_count", ""),
        })

    # Bootstrap curve slim file.
    boot_rows: List[Dict[str, Any]] = []
    for r in rows:
        br = {"context_label": r["context_label"], "raw_id": r["raw_id"], "class_name": r["class_name"], "split_type": r["split_type"], "clip_count": r["clip_count"]}
        for k in ks:
            br[f"bootstrap_identifiable_rate@{k}"] = r.get(f"bootstrap_identifiable_rate@{k}", "")
            br[f"median_intersection_size@{k}"] = r.get(f"median_intersection_size@{k}", "")
        boot_rows.append(br)

    hub_rows = _hub_confounder_summary(rows, cat_names, base_set, novel_set)
    split_rows = [
        _aggregate_summary(list(base_only_rows.values()), "base_only"),
        _aggregate_summary(list(all_visible_base_rows.values()), "base_target_all_visible_context"),
    ]
    if novel_rows:
        split_rows.append(_aggregate_summary(list(novel_rows.values()), "novel_audit_only_all_visible_context"))

    # Identifiability drop due to novel: base-only identifiable but all-visible not identifiable.
    aud_base_targets = [t for t, r in base_only_rows.items() if int(r.get("clip_count") or 0) >= 3]
    drop_due_novel = [
        t for t in aud_base_targets
        if str(base_only_rows[t].get("is_identifiable")).lower() == "true"
        and str(all_visible_base_rows[t].get("is_identifiable")).lower() != "true"
    ]

    summary: Dict[str, Any] = {
        "status": "PASS",
        "dataset_name": args.dataset_name,
        "split": split_meta,
        "annotation": ann_meta,
        "output_dir": str(out_dir),
        "params": {
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_ks": list(ks),
            "seed": args.seed,
            "skip_novel_audit": args.skip_novel_audit,
            "d_scorer_per_class_csv": args.d_scorer_per_class_csv or "",
            "proposal_support_per_class_csv": args.proposal_support_per_class_csv or "",
        },
        "context_summaries": split_rows,
        "identifiability_drop_due_to_novel_count": len(drop_due_novel),
        "identifiability_drop_due_to_novel_rate_among_auditable_base": _safe_div(len(drop_due_novel), len(aud_base_targets)),
        "failure_bucket_counts": dict(failure_counts),
        "top_persistent_or_near_confounders": hub_rows[:20],
        "notes": [
            "This is a GT-label context audit; it is not a training, inference, or mAP result.",
            "Base-only is the supervision-view upper bound; all-visible includes novel GT context as hidden visual confounders.",
            "D-scorer and proposal-support fields are optional joins and may be empty if their per-class CSVs are not provided.",
        ],
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(out_dir / "per_class_context_identifiability.csv", rows)
    _write_csv(out_dir / "hub_confounder_summary.csv", hub_rows)
    _write_csv(out_dir / "context_identifiability_by_split.csv", split_rows)
    _write_csv(out_dir / "bootstrap_identifiability_curves.csv", boot_rows)
    _write_csv(out_dir / "failure_bucket_summary.csv", failure_rows)
    _write_markdown(out_dir / "GT_CONTEXT_IDENTIFIABILITY_TAKEOVER.md", summary, failure_counts, out_dir)

    print(json.dumps({
        "status": "PASS",
        "output_dir": str(out_dir),
        "base_count": len(base_set),
        "novel_count": len(novel_set),
        "identifiability_drop_due_to_novel_rate_among_auditable_base": summary["identifiability_drop_due_to_novel_rate_among_auditable_base"],
        "failure_bucket_counts": dict(failure_counts),
    }, ensure_ascii=False, indent=2))
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only GT context identifiability audit for WS-OVVIS.")
    p.add_argument("--run_root", default="/mnt/sda/zyy/code/wsovvis/codex/outputs/G8_inference_and_eval/sinkhorn_safeneg_k32_b010_preaug_k10_repro_20260427")
    p.add_argument("--runtime_output_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--annotation_json", default="/home/zyy/code/wsovvis_asserts/dataset/LV-VIS/annotations/train_instances.json")
    p.add_argument("--split_json", default="package/reference/lvvis_official_base_novel_split.json")
    p.add_argument("--output_dir", default="")
    p.add_argument("--require_official_counts", action="store_true", default=True)
    p.add_argument("--no_require_official_counts", dest="require_official_counts", action="store_false")
    p.add_argument("--skip_novel_audit", action="store_true")
    p.add_argument("--bootstrap_repeats", type=int, default=100)
    p.add_argument("--bootstrap_ks", default=",".join(str(x) for x in K_DEFAULTS))
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--d_scorer_per_class_csv", default="", help="Optional per-class CSV from D full-Y + GT-carrier scorer audit.")
    p.add_argument("--proposal_support_per_class_csv", default="", help="Optional per-class CSV from proposal/support audit.")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
