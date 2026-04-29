#!/usr/bin/env python3
"""
Read-only residual-peeling-conditioned GT-carrier scorer audit.

Goal:
  Given label-only iterative residual identifiability results and D-arm row-level
  GT-carrier full-Y score rows, re-evaluate each GT-carrier row under residual
  candidate sets:
    1) base_residual:       Y_base(v) minus K_{t-1} plus GT
    2) all_visible_residual:Y_all(v) minus K_{t-1} plus GT
    3) fullY_minus_known:   available full-score universe minus K_{t-1} plus GT

This script is read-only: it does not alter training, inference, checkpoints, or
model semantics. It streams large JSONL files and writes compact summaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _as_str_id(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return None
        return str(int(x))
    s = str(x).strip()
    if not s or s.lower() in {"none", "nan", "null"}:
        return None
    try:
        return str(int(float(s)))
    except Exception:
        return s


def _as_int(x: Any, default: int = 0) -> int:
    sid = _as_str_id(x)
    if sid is None:
        return default
    try:
        return int(float(sid))
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _truth(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "pass"}


def _json_loads_maybe(x: Any, default: Any) -> Any:
    if x is None:
        return default
    if isinstance(x, (dict, list)):
        return x
    s = str(x).strip()
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def _extract_id_from_row(row: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in row:
            sid = _as_str_id(row.get(k))
            if sid is not None:
                return sid
    return None


def extract_raw_id(row: Dict[str, Any]) -> Optional[str]:
    return _extract_id_from_row(
        row,
        [
            "raw_id",
            "raw_category_id",
            "category_id",
            "class_raw_id",
            "gt_raw_id",
            "gt_category_id",
            "matched_gt_raw_id",
            "matched_gt_raw_id_canonical",
        ],
    )


def extract_gt_id(row: Dict[str, Any]) -> Optional[str]:
    return _extract_id_from_row(
        row,
        [
            "gt_raw_id",
            "gt_category_id",
            "gt_class_raw_id",
            "matched_gt_raw_id",
            "matched_gt_raw_id_canonical",
            "raw_category_id",
            "category_id",
            "raw_id",
        ],
    )


def extract_context_key(row: Dict[str, Any]) -> Optional[str]:
    # Prefer video/clip identity over image identity. This matches prior LV-VIS
    # video/clip-level semantic-label audits, while remaining schema-flexible.
    for k in [
        "clip_id",
        "clip_key",
        "video_id",
        "video_raw_id",
        "video",
        "video_name",
        "sequence_id",
        "image_id",
        "file_name",
    ]:
        if k in row:
            v = row.get(k)
            sid = _as_str_id(v)
            if sid is not None:
                return sid
    # Sometimes run artifacts carry nested metadata.
    for mk in ["meta", "metadata", "row_meta"]:
        m = row.get(mk)
        if isinstance(m, dict):
            got = extract_context_key(m)
            if got is not None:
                return got
    return None


def load_split(split_json: Path) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    obj = json.loads(split_json.read_text(encoding="utf-8"))

    def collect(keys: Sequence[str]) -> Set[str]:
        for k in keys:
            if k in obj:
                val = obj[k]
                if isinstance(val, dict):
                    return {sid for sid in (_as_str_id(x) for x in val.keys()) if sid is not None}
                if isinstance(val, list):
                    out: Set[str] = set()
                    for item in val:
                        if isinstance(item, dict):
                            sid = _extract_id_from_row(item, ["raw_id", "id", "category_id"])
                        else:
                            sid = _as_str_id(item)
                        if sid is not None:
                            out.add(sid)
                    return out
        return set()

    base = collect(["base_raw_ids", "base_ids", "base", "base_classes", "base_category_ids"])
    novel = collect(["novel_raw_ids", "novel_ids", "novel", "novel_classes", "novel_category_ids"])
    names: Dict[str, str] = {}
    for key in ["id_to_name", "raw_id_to_name", "category_id_to_name", "names"]:
        if isinstance(obj.get(key), dict):
            for k, v in obj[key].items():
                sid = _as_str_id(k)
                if sid is not None:
                    names[sid] = str(v)
    return base, novel, names


def load_annotation_contexts(annotation_json: Path, base_ids: Set[str], novel_ids: Set[str]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, str]]:
    obj = json.loads(annotation_json.read_text(encoding="utf-8"))
    categories = obj.get("categories") or []
    cat_names: Dict[str, str] = {}
    for c in categories:
        sid = _extract_id_from_row(c, ["id", "raw_id", "category_id"])
        if sid is not None:
            cat_names[sid] = str(c.get("name", sid))

    # image_id -> video_id map, for COCO-ish frame-level annotations.
    image_to_video: Dict[str, str] = {}
    for im in obj.get("images", []) or []:
        iid = _as_str_id(im.get("id"))
        vid = _as_str_id(im.get("video_id", im.get("video", im.get("video_name"))))
        if iid is not None:
            image_to_video[iid] = vid or iid

    # Some LV-VIS variants have videos as first-class objects.
    video_ids: Set[str] = set()
    for v in obj.get("videos", []) or []:
        vid = _as_str_id(v.get("id", v.get("video_id", v.get("name"))))
        if vid is not None:
            video_ids.add(vid)

    base_ctx: Dict[str, Set[str]] = defaultdict(set)
    all_ctx: Dict[str, Set[str]] = defaultdict(set)

    anns = obj.get("annotations", []) or []
    for ann in anns:
        if ann.get("ignore") or ann.get("iscrowd"):
            # Keep the audit conservative: ignored/crowd annotations are not
            # used as positive clip context.
            continue
        cid = _extract_id_from_row(ann, ["category_id", "raw_id", "raw_category_id", "cat_id"])
        if cid is None:
            continue
        vid = _as_str_id(ann.get("video_id", ann.get("video", ann.get("video_name"))))
        if vid is None:
            iid = _as_str_id(ann.get("image_id"))
            vid = image_to_video.get(iid, iid)
        if vid is None:
            continue
        if cid in base_ids:
            base_ctx[vid].add(cid)
            all_ctx[vid].add(cid)
        elif cid in novel_ids:
            all_ctx[vid].add(cid)
        else:
            # Keep unknown taxonomy categories out of official base/novel contexts.
            pass

    return dict(base_ctx), dict(all_ctx), cat_names


def load_label_rows(label_csv: Path, variant: str) -> Dict[str, Dict[str, Any]]:
    rows = list(csv.DictReader(label_csv.open(newline="", encoding="utf-8")))
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        rv = r.get("variant") or r.get("run_variant") or r.get("context_variant") or ""
        if rv != variant:
            continue
        rid = extract_raw_id(r)
        if rid is None:
            continue
        split_type = r.get("split_type", "base") or "base"
        if split_type != "base":
            continue
        # Only train-observed classes are relevant to this conditioned scorer audit.
        clip_count = _as_int(r.get("clip_count", r.get("base_clip_count", 0)), 0)
        if clip_count <= 0:
            continue
        if rid not in out:
            out[rid] = r
    return out


def get_certificate(row: Dict[str, Any]) -> str:
    return row.get("certificate_type") or row.get("certificate") or row.get("final_bucket") or row.get("bucket") or "unknown"


def get_resolved(row: Dict[str, Any]) -> bool:
    if "resolved" in row:
        return _truth(row.get("resolved"))
    if "is_resolved" in row:
        return _truth(row.get("is_resolved"))
    cert = get_certificate(row)
    return cert not in {"unresolved", "observed_but_insufficient_context", "absent"}


def get_iteration(row: Dict[str, Any]) -> int:
    for k in ["resolved_at_iteration", "iteration", "resolved_iteration"]:
        if k in row and str(row.get(k)).strip() != "":
            return _as_int(row.get(k), 0)
    cert = get_certificate(row)
    if cert == "initial_context_identifiable":
        return 0
    return 999999


def build_known_sets(label_by_id: Dict[str, Dict[str, Any]], person_raw_id: Optional[str], include_person: bool) -> Dict[int, Set[str]]:
    initial: Set[str] = set()
    resolved_by_iter: Dict[int, Set[str]] = defaultdict(set)
    max_iter = 0
    for rid, row in label_by_id.items():
        if not get_resolved(row):
            continue
        cert = get_certificate(row)
        it = get_iteration(row)
        if cert == "initial_context_identifiable" or it == 0:
            initial.add(rid)
        else:
            resolved_by_iter[it].add(rid)
            max_iter = max(max_iter, it)
    if include_person and person_raw_id:
        initial.add(person_raw_id)

    known_before: Dict[int, Set[str]] = {}
    acc = set(initial)
    known_before[0] = set(acc)
    for it in range(1, max_iter + 2):
        known_before[it] = set(acc)
        acc.update(resolved_by_iter.get(it, set()))
    return known_before


def score_map_from_row(row: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Extract per-class scores from a schema-flexible row.

    Returns score_map and metadata with extraction mode. This supports exact
    residual rank recomputation only if all required residual candidates are in
    score_map. If D rows only store topK, summaries will report low coverage.
    """
    meta = {"mode": None, "score_count": 0}

    # Dict-like score maps.
    for k in [
        "scores_by_raw_id",
        "score_by_raw_id",
        "candidate_scores_by_raw_id",
        "logits_by_raw_id",
        "class_scores_by_raw_id",
        "scores",
        "logits",
    ]:
        val = row.get(k)
        if isinstance(val, dict):
            sm = {sid: _as_float(v) for sid, v in ((_as_str_id(a), b) for a, b in val.items()) if sid is not None}
            if sm:
                meta.update({"mode": k, "score_count": len(sm)})
                return sm, meta

    # JSON-encoded dicts.
    for k in [
        "scores_by_raw_id",
        "score_by_raw_id",
        "candidate_scores_by_raw_id",
        "logits_by_raw_id",
        "class_scores_by_raw_id",
        "scores_json",
        "scores",
        "logits",
    ]:
        if k in row and isinstance(row.get(k), str):
            val = _json_loads_maybe(row.get(k), None)
            if isinstance(val, dict):
                sm = {sid: _as_float(v) for sid, v in ((_as_str_id(a), b) for a, b in val.items()) if sid is not None}
                if sm:
                    meta.update({"mode": k, "score_count": len(sm)})
                    return sm, meta

    # Parallel candidate ids and scores.
    id_keys = ["candidate_raw_ids", "candidate_ids", "raw_ids", "topk_raw_ids", "top_raw_ids", "class_ids"]
    score_keys = ["candidate_scores", "scores", "topk_scores", "top_scores", "logits"]
    for ik in id_keys:
        ids_val = _json_loads_maybe(row.get(ik), row.get(ik)) if ik in row else None
        if not isinstance(ids_val, list):
            continue
        ids = [_as_str_id(x) for x in ids_val]
        ids = [x for x in ids if x is not None]
        for sk in score_keys:
            scores_val = _json_loads_maybe(row.get(sk), row.get(sk)) if sk in row else None
            if not isinstance(scores_val, list):
                continue
            if len(scores_val) != len(ids):
                continue
            sm = {sid: _as_float(sc) for sid, sc in zip(ids, scores_val)}
            if sm:
                meta.update({"mode": f"{ik}+{sk}", "score_count": len(sm)})
                return sm, meta

    # List of candidate dicts.
    for k in ["candidates", "topk", "top_candidates", "candidate_scores"]:
        val = _json_loads_maybe(row.get(k), row.get(k)) if k in row else None
        if not isinstance(val, list):
            continue
        sm: Dict[str, float] = {}
        for item in val:
            if not isinstance(item, dict):
                continue
            sid = _extract_id_from_row(item, ["raw_id", "raw_category_id", "category_id", "class_raw_id", "id"])
            score = None
            for sk in ["score", "logit", "similarity", "sim", "value"]:
                if sk in item:
                    score = _as_float(item.get(sk))
                    break
            if sid is not None and score is not None:
                sm[sid] = score
        if sm:
            meta.update({"mode": k, "score_count": len(sm)})
            return sm, meta

    meta.update({"mode": "none", "score_count": 0})
    return {}, meta


def rank_in_candidates(score_map: Dict[str, float], gt: str, candidates: Set[str]) -> Dict[str, Any]:
    candidates = set(candidates)
    candidates.add(gt)
    available = {c: score_map[c] for c in candidates if c in score_map}
    missing = candidates - set(available.keys())
    exact = len(missing) == 0
    if gt not in available or not available:
        return {
            "evaluable": False,
            "exact": exact,
            "candidate_size": len(candidates),
            "available_candidate_size": len(available),
            "coverage": (len(available) / len(candidates)) if candidates else 0.0,
            "missing_count": len(missing),
        }
    sorted_items = sorted(available.items(), key=lambda kv: (-kv[1], kv[0]))
    rank = 1 + next(i for i, (cid, _) in enumerate(sorted_items) if cid == gt)
    top1 = sorted_items[0][0]
    gt_score = available[gt]
    best_non = None
    for cid, sc in sorted_items:
        if cid != gt:
            best_non = (cid, sc)
            break
    margin = None if best_non is None else gt_score - best_non[1]
    return {
        "evaluable": True,
        "exact": exact,
        "candidate_size": len(candidates),
        "available_candidate_size": len(available),
        "coverage": (len(available) / len(candidates)) if candidates else 0.0,
        "missing_count": len(missing),
        "rank": rank,
        "rank1": rank == 1,
        "top5": rank <= 5,
        "top1_raw_id": top1,
        "gt_score": gt_score,
        "best_non_gt_raw_id": best_non[0] if best_non else "",
        "best_non_gt_score": best_non[1] if best_non else None,
        "margin": margin,
    }


def summarize_bool(vals: List[bool]) -> Optional[float]:
    if not vals:
        return None
    return sum(1 for v in vals if v) / len(vals)


def summarize_nums(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def bucket_rate(x: float) -> str:
    if x >= 0.75:
        return "strongly_recognized_ge_0.75"
    if x >= 0.50:
        return "recognized_ge_0.50"
    if x >= 0.25:
        return "weakly_recognized_ge_0.25"
    if x >= 0.10:
        return "mostly_failed_ge_0.10"
    return "collapsed_lt_0.10"


def main() -> int:
    ap = argparse.ArgumentParser(description="Residual-peeling-conditioned GT-carrier scorer audit")
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--dataset_name", default="lvvis_train_base")
    ap.add_argument("--variant", default="person_aware")
    ap.add_argument("--d_row_scores_jsonl", required=True)
    ap.add_argument("--annotation_json", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--per_class_csv", default="")
    ap.add_argument("--person_raw_id", default="773")
    ap.add_argument("--candidate_policies", default="base_residual,all_visible_residual,fullY_minus_known")
    ap.add_argument("--include_person_in_initial", action="store_true", default=True)
    ap.add_argument("--no_include_person_in_initial", action="store_false", dest="include_person_in_initial")
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--top_examples", type=int, default=128)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    out_dir = run_root / "analysis" / "residual_peeling_conditioned_gtcarrier_scorer" / args.dataset_name / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    split_json = Path(args.split_json)
    base_ids, novel_ids, split_names = load_split(split_json)
    annotation_json = Path(args.annotation_json)
    base_ctx, all_ctx, cat_names = load_annotation_contexts(annotation_json, base_ids, novel_ids)
    for k, v in split_names.items():
        cat_names.setdefault(k, v)

    label_csv = Path(args.per_class_csv) if args.per_class_csv else run_root / "analysis" / "iterative_residual_label_identifiability" / args.dataset_name / "per_class_iterative_residual_identifiability.csv"
    label_by_id = load_label_rows(label_csv, args.variant)
    known_before = build_known_sets(label_by_id, _as_str_id(args.person_raw_id), args.include_person_in_initial)
    policies = [p.strip() for p in args.candidate_policies.split(",") if p.strip()]

    # Store examples and aggregate counters.
    policy_stats: Dict[str, Dict[str, Any]] = {p: defaultdict(list) for p in policies}
    by_cert_policy: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(lambda: defaultdict(list))
    per_class_acc: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(lambda: defaultdict(list))
    examples: List[Dict[str, Any]] = []
    schema_counter = Counter()
    row_count = 0
    used_row_count = 0
    skipped_no_label = 0
    skipped_no_context = 0
    skipped_no_scores = 0

    d_jsonl = Path(args.d_row_scores_jsonl)
    with d_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row_count += 1
            if args.max_rows and row_count > args.max_rows:
                break
            try:
                row = json.loads(line)
            except Exception:
                continue
            gt = extract_gt_id(row)
            if gt is None:
                continue
            lbl = label_by_id.get(gt)
            if lbl is None:
                skipped_no_label += 1
                continue
            ctx_key = extract_context_key(row)
            if ctx_key is None:
                skipped_no_context += 1
                continue
            smap, smeta = score_map_from_row(row)
            schema_counter[smeta.get("mode") or "none"] += 1
            if not smap:
                skipped_no_scores += 1
                continue

            used_row_count += 1
            cert = get_certificate(lbl)
            resolved = get_resolved(lbl)
            it = get_iteration(lbl)
            k_prev = set(known_before.get(it, known_before.get(0, set())))
            # Never remove the target from its own candidate set.
            k_prev.discard(gt)

            base_set = set(base_ctx.get(ctx_key, set()))
            all_set = set(all_ctx.get(ctx_key, set()))
            full_available = set(smap.keys())

            policy_candidates: Dict[str, Set[str]] = {}
            if "base_residual" in policies:
                policy_candidates["base_residual"] = (base_set - k_prev) | {gt}
            if "all_visible_residual" in policies:
                policy_candidates["all_visible_residual"] = (all_set - k_prev) | {gt}
            if "fullY_minus_known" in policies:
                policy_candidates["fullY_minus_known"] = (full_available - k_prev) | {gt}

            # FullY baseline over available score universe.
            full_res = rank_in_candidates(smap, gt, set(smap.keys()))
            full_rank1 = bool(full_res.get("rank1")) if full_res.get("evaluable") else False
            full_top1 = full_res.get("top1_raw_id", "")

            for pol in policies:
                cand = policy_candidates.get(pol, set())
                res = rank_in_candidates(smap, gt, cand)
                st = policy_stats[pol]
                st["row_count"].append(1)
                st["evaluable"].append(bool(res.get("evaluable")))
                st["exact"].append(bool(res.get("exact")))
                st["candidate_size"].append(float(res.get("candidate_size", 0)))
                st["coverage"].append(float(res.get("coverage", 0)))
                st["fullY_rank1"].append(full_rank1)
                if res.get("evaluable"):
                    st["rank1"].append(bool(res.get("rank1")))
                    st["top5"].append(bool(res.get("top5")))
                    if res.get("rank") is not None:
                        st["rank"].append(float(res.get("rank")))
                    if res.get("margin") is not None:
                        st["margin"].append(float(res.get("margin")))
                    known_removed_top1 = bool(full_top1 and full_top1 in k_prev and not full_rank1)
                    st["known_top1_suppressor_removed"].append(known_removed_top1)
                    non_known_after = bool((not res.get("rank1")) and res.get("top1_raw_id") not in k_prev)
                    st["non_known_suppressor_after"].append(non_known_after)

                    bc = by_cert_policy[(cert, pol)]
                    bc["row_count"].append(1)
                    bc["rank1"].append(bool(res.get("rank1")))
                    bc["top5"].append(bool(res.get("top5")))
                    bc["fullY_rank1"].append(full_rank1)
                    bc["candidate_size"].append(float(res.get("candidate_size", 0)))
                    bc["exact"].append(bool(res.get("exact")))
                    bc["coverage"].append(float(res.get("coverage", 0)))

                    pc = per_class_acc[(gt, pol)]
                    pc["rank1"].append(bool(res.get("rank1")))
                    pc["top5"].append(bool(res.get("top5")))
                    pc["fullY_rank1"].append(full_rank1)
                    pc["rank"].append(float(res.get("rank", 0)))
                    pc["candidate_size"].append(float(res.get("candidate_size", 0)))
                    pc["exact"].append(bool(res.get("exact")))
                    pc["coverage"].append(float(res.get("coverage", 0)))
                    pc["certificate"] = cert
                    pc["resolved"] = resolved
                    pc["iteration"] = it
                    pc["class_name"] = cat_names.get(gt, gt)

                    if (not res.get("rank1")) and len(examples) < args.top_examples:
                        examples.append({
                            "policy": pol,
                            "gt_raw_id": gt,
                            "gt_name": cat_names.get(gt, gt),
                            "context_key": ctx_key,
                            "certificate_type": cert,
                            "resolved_at_iteration": it,
                            "candidate_size": res.get("candidate_size"),
                            "available_candidate_size": res.get("available_candidate_size"),
                            "coverage": res.get("coverage"),
                            "exact": res.get("exact"),
                            "fullY_top1_raw_id": full_top1,
                            "fullY_top1_name": cat_names.get(str(full_top1), str(full_top1)),
                            "residual_top1_raw_id": res.get("top1_raw_id"),
                            "residual_top1_name": cat_names.get(str(res.get("top1_raw_id")), str(res.get("top1_raw_id"))),
                            "residual_rank": res.get("rank"),
                            "residual_margin": res.get("margin"),
                            "known_removed_fullY_top1": bool(full_top1 and full_top1 in k_prev),
                        })

    def pack_stats(st: Dict[str, List[Any]]) -> Dict[str, Any]:
        row_n = len(st.get("row_count", []))
        evals = st.get("evaluable", [])
        rank1s = st.get("rank1", [])
        fulls = st.get("fullY_rank1", [])
        out = {
            "row_count": row_n,
            "evaluable_count": sum(1 for x in evals if x),
            "evaluable_rate": summarize_bool(evals),
            "exact_rate": summarize_bool(st.get("exact", [])),
            "candidate_score_coverage_mean": summarize_nums(st.get("coverage", [])),
            "candidate_size_mean": summarize_nums(st.get("candidate_size", [])),
            "fullY_rank1_rate": summarize_bool(fulls),
            "residual_rank1_rate": summarize_bool(rank1s),
            "residual_top5_rate": summarize_bool(st.get("top5", [])),
            "rank1_gain_vs_fullY": (summarize_bool(rank1s) - summarize_bool(fulls)) if rank1s and fulls else None,
            "mean_residual_rank": summarize_nums(st.get("rank", [])),
            "mean_residual_margin": summarize_nums(st.get("margin", [])),
            "known_top1_suppressor_removed_rate": summarize_bool(st.get("known_top1_suppressor_removed", [])),
            "non_known_suppressor_after_residual_rate": summarize_bool(st.get("non_known_suppressor_after", [])),
        }
        return out

    summary_by_policy = {pol: pack_stats(dict(st)) for pol, st in policy_stats.items()}

    cert_rows: List[Dict[str, Any]] = []
    for (cert, pol), st in sorted(by_cert_policy.items()):
        d = pack_stats(dict(st))
        d.update({"certificate_type": cert, "policy": pol})
        cert_rows.append(d)

    class_rows: List[Dict[str, Any]] = []
    for (cid, pol), acc in sorted(per_class_acc.items(), key=lambda kv: (kv[0][1], int(kv[0][0]) if kv[0][0].isdigit() else kv[0][0])):
        class_rows.append({
            "raw_id": cid,
            "class_name": acc.get("class_name", cid),
            "policy": pol,
            "certificate_type": acc.get("certificate", ""),
            "resolved": acc.get("resolved", False),
            "resolved_at_iteration": acc.get("iteration", ""),
            "row_count": len(acc.get("rank1", [])),
            "fullY_rank1_rate": summarize_bool(acc.get("fullY_rank1", [])),
            "residual_rank1_rate": summarize_bool(acc.get("rank1", [])),
            "residual_top5_rate": summarize_bool(acc.get("top5", [])),
            "mean_residual_rank": summarize_nums(acc.get("rank", [])),
            "candidate_size_mean": summarize_nums(acc.get("candidate_size", [])),
            "exact_rate": summarize_bool(acc.get("exact", [])),
            "candidate_score_coverage_mean": summarize_nums(acc.get("coverage", [])),
        })

    # Write outputs.
    with (out_dir / "summary_by_certificate.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["certificate_type", "policy", "row_count", "evaluable_count", "evaluable_rate", "exact_rate", "candidate_score_coverage_mean", "candidate_size_mean", "fullY_rank1_rate", "residual_rank1_rate", "residual_top5_rate", "rank1_gain_vs_fullY", "mean_residual_rank", "mean_residual_margin", "known_top1_suppressor_removed_rate", "non_known_suppressor_after_residual_rate"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in cert_rows:
            w.writerow(r)

    with (out_dir / "per_class_residual_conditioned_scorer.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["raw_id", "class_name", "policy", "certificate_type", "resolved", "resolved_at_iteration", "row_count", "fullY_rank1_rate", "residual_rank1_rate", "residual_top5_rate", "mean_residual_rank", "candidate_size_mean", "exact_rate", "candidate_score_coverage_mean"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in class_rows:
            w.writerow(r)

    with (out_dir / "failure_examples.jsonl").open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    summary = {
        "status": "PASS" if used_row_count > 0 else "FAIL_NO_USABLE_ROWS",
        "run_root": str(run_root),
        "dataset_name": args.dataset_name,
        "variant": args.variant,
        "d_row_scores_jsonl": str(d_jsonl),
        "annotation_json": str(annotation_json),
        "split_json": str(split_json),
        "per_class_csv": str(label_csv),
        "output_dir": str(out_dir),
        "base_count": len(base_ids),
        "novel_count": len(novel_ids),
        "label_train_observed_base_count": len(label_by_id),
        "row_scores_seen": row_count,
        "row_scores_used": used_row_count,
        "skipped_no_label": skipped_no_label,
        "skipped_no_context": skipped_no_context,
        "skipped_no_scores": skipped_no_scores,
        "score_schema_modes": dict(schema_counter),
        "candidate_policies": policies,
        "summary_by_policy": summary_by_policy,
        "outputs": {
            "summary_json": str(out_dir / "summary.json"),
            "summary_by_certificate_csv": str(out_dir / "summary_by_certificate.csv"),
            "per_class_csv": str(out_dir / "per_class_residual_conditioned_scorer.csv"),
            "examples_jsonl": str(out_dir / "failure_examples.jsonl"),
            "takeover_md": str(out_dir / "RESIDUAL_PEELING_CONDITIONED_GTCARRIER_SCORER_TAKEOVER.md"),
        },
        "warnings": [],
    }
    for pol, st in summary_by_policy.items():
        if st.get("exact_rate") is not None and st.get("exact_rate") < 0.99:
            summary["warnings"].append(f"{pol}: exact_rate={st.get('exact_rate')} < 0.99; residual ranks may be partial if row score map is topK-only")

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Residual-Peeling-Conditioned GT-Carrier Scorer Audit",
        "",
        f"- status: {summary['status']}",
        f"- row_scores_seen: {row_count}",
        f"- row_scores_used: {used_row_count}",
        f"- score_schema_modes: {dict(schema_counter)}",
        "",
        "## Summary by policy",
    ]
    for pol, st in summary_by_policy.items():
        lines.append(f"- {pol}: fullY_rank1={st.get('fullY_rank1_rate')}, residual_rank1={st.get('residual_rank1_rate')}, gain={st.get('rank1_gain_vs_fullY')}, candidate_size_mean={st.get('candidate_size_mean')}, exact_rate={st.get('exact_rate')}")
    if summary["warnings"]:
        lines.append("")
        lines.append("## Warnings")
        for w in summary["warnings"]:
            lines.append(f"- {w}")
    (out_dir / "RESIDUAL_PEELING_CONDITIONED_GTCARRIER_SCORER_TAKEOVER.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
