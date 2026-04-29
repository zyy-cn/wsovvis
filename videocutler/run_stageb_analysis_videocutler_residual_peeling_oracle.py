#!/usr/bin/env python3
"""
Read-only VideoCutLER-carrier residual-peeling oracle audit.

This audit replays the current text-carrier scorer on real VideoCutLER trajectory
carriers, then evaluates each matched trajectory under residual-peeling candidate
sets derived from label-only iterative residual identifiability.

It does not train, infer masks, alter checkpoints, or modify repository state.
Large inputs are streamed where possible and outputs are compact summaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

Record = Dict[str, Any]


# ----------------------------- generic helpers -----------------------------

def _as_str_id(x: Any) -> Optional[str]:
    if x is None or isinstance(x, bool):
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


def _iter_jsonl(path: Path) -> Iterable[Record]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _read_csv_rows(path: Path) -> List[Record]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen: Set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _extract_id(row: Record, keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in row:
            sid = _as_str_id(row.get(k))
            if sid is not None:
                return sid
    return None


def _extract_trajectory_id(row: Record) -> Optional[str]:
    return _extract_id(
        row,
        [
            "trajectory_id",
            "traj_id",
            "trajectory_uid",
            "track_id",
            "proposal_id",
            "carrier_id",
            "row_id",
        ],
    )


def _extract_gt_id(row: Record) -> Optional[str]:
    return _extract_id(
        row,
        [
            "matched_gt_raw_id_canonical",
            "matched_gt_raw_id",
            "best_gt_raw_id",
            "gt_raw_id",
            "gt_category_id",
            "gt_class_raw_id",
            "raw_category_id",
        ],
    )


def _extract_clip_key(row: Record) -> Optional[str]:
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
            sid = _as_str_id(row.get(k))
            if sid is not None:
                return sid
    for mk in ["meta", "metadata", "row_meta"]:
        m = row.get(mk)
        if isinstance(m, dict):
            got = _extract_clip_key(m)
            if got is not None:
                return got
    return None


def _extract_iou(row: Record) -> Optional[float]:
    for k in [
        "matched_gt_iou",
        "best_gt_iou",
        "gt_iou",
        "iou",
        "max_iou",
        "best_iou",
        "trajectory_gt_iou",
    ]:
        if k in row and row.get(k) not in {None, ""}:
            return _as_float(row.get(k), default=0.0)
    return None


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("zero or non-finite vector norm")
    return (v / norm).astype(np.float32, copy=False)


def _rank_desc(scores: np.ndarray, gt_index: int) -> Tuple[int, int, float, float]:
    gt_score = float(scores[gt_index])
    # Stable enough for audit: rank is 1 + count strictly greater than GT score.
    rank = int(np.sum(scores > gt_score)) + 1
    top_idx = int(np.argmax(scores))
    if len(scores) <= 1:
        margin = float("inf")
    else:
        # maximum non-GT score
        if top_idx != gt_index:
            best_non = float(scores[top_idx])
        else:
            tmp = scores.copy()
            tmp[gt_index] = -np.inf
            best_non = float(np.max(tmp))
        margin = gt_score - best_non
    return rank, top_idx, gt_score, margin


def _rate(n: float, d: float) -> Optional[float]:
    return float(n) / float(d) if d else None


# ----------------------------- split/context -----------------------------

def load_split(split_json: Path) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    obj = json.loads(split_json.read_text(encoding="utf-8"))

    def collect(keys: Sequence[str]) -> Set[str]:
        for k in keys:
            if k not in obj:
                continue
            val = obj[k]
            if isinstance(val, dict):
                return {sid for sid in (_as_str_id(x) for x in val.keys()) if sid is not None}
            if isinstance(val, list):
                out: Set[str] = set()
                for item in val:
                    if isinstance(item, dict):
                        sid = _extract_id(item, ["raw_id", "id", "category_id"])
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
    cat_names: Dict[str, str] = {}
    for c in obj.get("categories", []) or []:
        sid = _extract_id(c, ["id", "raw_id", "category_id"])
        if sid is not None:
            cat_names[sid] = str(c.get("name", sid))

    image_to_video: Dict[str, str] = {}
    for im in obj.get("images", []) or []:
        iid = _as_str_id(im.get("id"))
        vid = _as_str_id(im.get("video_id", im.get("video", im.get("video_name"))))
        if iid is not None:
            image_to_video[iid] = vid or iid

    base_ctx: Dict[str, Set[str]] = defaultdict(set)
    all_ctx: Dict[str, Set[str]] = defaultdict(set)
    for ann in obj.get("annotations", []) or []:
        if ann.get("ignore") or ann.get("iscrowd"):
            continue
        cid = _as_str_id(ann.get("category_id", ann.get("raw_id")))
        if cid is None:
            continue
        image_id = _as_str_id(ann.get("image_id"))
        video_id = _as_str_id(ann.get("video_id"))
        key = video_id or (image_to_video.get(image_id) if image_id is not None else None) or image_id
        if key is None:
            continue
        if cid in base_ids:
            base_ctx[key].add(cid)
            all_ctx[key].add(cid)
        elif cid in novel_ids:
            all_ctx[key].add(cid)
    return dict(base_ctx), dict(all_ctx), cat_names


# ----------------------------- iterative labels -----------------------------

def load_iterative_labels(per_class_csv: Path, variant: str, person_raw_id: str) -> Tuple[Dict[str, Record], Dict[int, Set[str]], Set[str], Dict[str, Any]]:
    rows = _read_csv_rows(per_class_csv)
    available = sorted({str(r.get("variant", "")) for r in rows if r.get("variant")})
    selected: Dict[str, Record] = {}
    duplicate = Counter()
    for r in rows:
        if str(r.get("variant", "")) != str(variant):
            continue
        rid = _extract_id(r, ["raw_id", "class_raw_id", "category_id"])
        if rid is None:
            continue
        if rid in selected:
            duplicate[rid] += 1
            continue
        selected[rid] = r

    initial: Set[str] = set()
    resolved_by_iter: Dict[int, Set[str]] = defaultdict(set)
    for rid, r in selected.items():
        if not _truth(r.get("resolved", r.get("is_resolved", ""))):
            continue
        cert = str(r.get("certificate_type", r.get("certificate", "")))
        it = _as_int(r.get("resolved_at_iteration", r.get("iteration", 0)), default=0)
        if cert == "initial_context_identifiable" or it <= 0:
            initial.add(rid)
            resolved_by_iter[0].add(rid)
        else:
            resolved_by_iter[it].add(rid)
    if person_raw_id in selected:
        # person-aware variant usually already includes person in the initial anchor set.
        initial.add(person_raw_id)
        resolved_by_iter[0].add(person_raw_id)

    meta = {
        "available_variants": available,
        "selected_rows": len(selected),
        "duplicate_ids_skipped": dict(duplicate),
        "initial_known_count": len(initial),
        "max_iteration": max(resolved_by_iter.keys()) if resolved_by_iter else 0,
    }
    return selected, {int(k): set(v) for k, v in resolved_by_iter.items()}, initial, meta


def known_before_iteration(resolved_by_iter: Dict[int, Set[str]], initial: Set[str], iteration: int) -> Set[str]:
    if iteration <= 0:
        return set(initial)
    out = set(initial)
    for it, ids in resolved_by_iter.items():
        if it < iteration:
            out.update(ids)
    return out


# ----------------------------- vector/prototype IO -----------------------------

_VECTOR_LOCATOR_RE = re.compile(r"^(?P<path>[A-Za-z0-9_./-]+)#(?P<key>[A-Za-z0-9_]+)\[(?P<idx>[0-9]+)\]$")
_PROTO_LOCATOR_RE = re.compile(r"^(?P<path>[A-Za-z0-9_./-]+)#protos\[(?P<idx>[0-9]+)\]$")
_NPZ_CACHE: Dict[Tuple[str, str], np.ndarray] = {}
_NPZ_CACHE_ORDER: List[Tuple[str, str]] = []


def _load_npz_array(path: Path, key: str, cache_size: int = 8) -> np.ndarray:
    ckey = (str(path), str(key))
    if ckey in _NPZ_CACHE:
        return _NPZ_CACHE[ckey]
    with np.load(path, allow_pickle=False) as payload:
        if key not in payload.files:
            raise KeyError(f"missing key={key} in {path}")
        arr = np.asarray(payload[key])
    _NPZ_CACHE[ckey] = arr
    _NPZ_CACHE_ORDER.append(ckey)
    while len(_NPZ_CACHE_ORDER) > int(cache_size):
        old = _NPZ_CACHE_ORDER.pop(0)
        _NPZ_CACHE.pop(old, None)
    return arr


def read_vector_from_locator(parent_dir: Path, locator: str) -> np.ndarray:
    # Prefer repo implementation if available, but keep fallback self-contained.
    try:
        from videocutler.ext_stageb_ovvis.banks.carrier_bank import read_vector_from_locator as repo_reader  # type: ignore
        return np.asarray(repo_reader(parent_dir, locator), dtype=np.float32)
    except Exception:
        pass
    m = _VECTOR_LOCATOR_RE.match(str(locator))
    if not m:
        raise ValueError(f"invalid vector locator: {locator}")
    path = parent_dir / Path(m.group("path"))
    key = str(m.group("key"))
    idx = int(m.group("idx"))
    arr = _load_npz_array(path, key)
    if idx < 0 or idx >= int(arr.shape[0]):
        raise IndexError(f"index out of range in {locator}")
    return np.asarray(arr[idx], dtype=np.float32)


def load_text_prototypes(text_records_jsonl: Path) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    # Prefer repo text bank reader when possible.
    try:
        from videocutler.ext_stageb_ovvis.banks.text_bank import read_text_prototype_records, resolve_text_prototype  # type: ignore
        records = read_text_prototype_records(text_records_jsonl)
        raw_ids: List[str] = []
        names: Dict[str, str] = {}
        vectors: List[np.ndarray] = []
        for rec in records:
            rid = _as_str_id(rec.get("raw_id"))
            if rid is None:
                continue
            vec = np.asarray(resolve_text_prototype(text_records_jsonl, rec), dtype=np.float32)
            raw_ids.append(rid)
            names[rid] = str(rec.get("name", rec.get("class_name", rid)))
            vectors.append(_l2_normalize(vec))
        mat = np.stack(vectors, axis=0).astype(np.float32) if vectors else np.zeros((0, 0), dtype=np.float32)
        return raw_ids, names, mat
    except Exception:
        pass

    raw_ids = []
    names: Dict[str, str] = {}
    vectors = []
    for rec in _iter_jsonl(text_records_jsonl):
        rid = _as_str_id(rec.get("raw_id", rec.get("category_id")))
        if rid is None:
            continue
        proto_path = str(rec.get("proto_path", rec.get("path", "")))
        m = _PROTO_LOCATOR_RE.match(proto_path)
        if not m:
            # Direct vector fallback.
            val = rec.get("vector", rec.get("prototype", None))
            if val is None:
                continue
            vec = np.asarray(val, dtype=np.float32)
        else:
            payload_path = text_records_jsonl.parent / Path(m.group("path"))
            arr = _load_npz_array(payload_path, "protos")
            vec = np.asarray(arr[int(m.group("idx"))], dtype=np.float32)
        raw_ids.append(rid)
        names[rid] = str(rec.get("name", rec.get("class_name", rid)))
        vectors.append(_l2_normalize(vec))
    mat = np.stack(vectors, axis=0).astype(np.float32) if vectors else np.zeros((0, 0), dtype=np.float32)
    return raw_ids, names, mat


def load_carrier_records(carrier_records_jsonl: Path) -> Dict[str, Record]:
    out: Dict[str, Record] = {}
    for row in _iter_jsonl(carrier_records_jsonl):
        tid = _extract_trajectory_id(row)
        if tid is not None and tid not in out:
            out[tid] = row
    return out


def load_trajectory_records(trajectory_records_jsonl: Path) -> Dict[str, Record]:
    out: Dict[str, Record] = {}
    for row in _iter_jsonl(trajectory_records_jsonl):
        tid = _extract_trajectory_id(row)
        if tid is not None and tid not in out:
            out[tid] = row
    return out


def _candidate_match_files(run_root: Path, dataset_name: str) -> List[Path]:
    candidates = [
        run_root / "analysis" / "videocutler_multiplicity_precision" / dataset_name / "trajectory_precision_rows.csv",
        run_root / "analysis" / "videocutler_gt_trajectory_coverage" / dataset_name / "trajectory_coverage_rows.csv",
        run_root / "analysis" / "videocutler_gt_trajectory_coverage" / dataset_name / "coverage_examples.jsonl",
    ]
    for p in run_root.glob(f"analysis/**/{dataset_name}/**/*trajectory*rows*.csv"):
        candidates.append(p)
    for p in run_root.glob(f"analysis/**/{dataset_name}/**/*evidence_by_trajectory*.jsonl"):
        candidates.append(p)
    seen: Set[str] = set()
    out: List[Path] = []
    for p in candidates:
        if p.is_file() and str(p) not in seen:
            out.append(p)
            seen.add(str(p))
    return out


def load_gt_match_rows(gt_match_path: Optional[Path], run_root: Path, dataset_name: str) -> Tuple[Dict[str, Record], Optional[str], Dict[str, Any]]:
    paths: List[Path] = []
    if gt_match_path is not None and str(gt_match_path):
        paths.append(gt_match_path)
    paths.extend(_candidate_match_files(run_root, dataset_name))
    meta = {"candidate_paths": [str(p) for p in paths], "selected_path": None, "rows_loaded": 0, "matched_gt_rows": 0}
    for p in paths:
        if not p.is_file():
            continue
        rows: Iterable[Record]
        if p.suffix.lower() == ".csv":
            rows = _read_csv_rows(p)
        else:
            rows = _iter_jsonl(p)
        out: Dict[str, Record] = {}
        total = 0
        gt_rows = 0
        for row in rows:
            total += 1
            tid = _extract_trajectory_id(row)
            gt = _extract_gt_id(row)
            if tid is None or gt is None:
                continue
            if tid not in out:
                out[tid] = row
                gt_rows += 1
        if out:
            meta.update({"selected_path": str(p), "rows_loaded": total, "matched_gt_rows": gt_rows})
            return out, str(p), meta
    return {}, None, meta


# ----------------------------- scorer evaluation -----------------------------

def _resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    repo_root = Path(args.repo_root or os.getcwd()).resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()

    def first_existing(paths: Sequence[Path]) -> Path:
        for p in paths:
            if p.is_file():
                return p
        return paths[0]

    trajectory_records = Path(args.trajectory_records_jsonl) if args.trajectory_records_jsonl else first_existing([
        asset_root / "exports" / args.dataset_name / "trajectory_records.jsonl",
        repo_root / "exports" / args.dataset_name / "trajectory_records.jsonl",
    ])
    carrier_records = Path(args.carrier_records_jsonl) if args.carrier_records_jsonl else first_existing([
        asset_root / "carrier_bank" / args.dataset_name / "carrier_records.jsonl",
        repo_root / "carrier_bank" / args.dataset_name / "carrier_records.jsonl",
    ])
    text_records = Path(args.text_records_jsonl) if args.text_records_jsonl else first_existing([
        asset_root / "text_bank" / "text_prototype_records.jsonl",
        repo_root / "text_bank" / "text_prototype_records.jsonl",
    ])
    per_class_csv = Path(args.per_class_csv) if args.per_class_csv else run_root / "analysis" / "iterative_residual_label_identifiability" / args.dataset_name / "per_class_iterative_residual_identifiability.csv"
    output_dir = run_root / "analysis" / "videocutler_residual_peeling_oracle" / args.dataset_name / args.variant
    return {
        "repo_root": repo_root,
        "asset_root": asset_root,
        "run_root": run_root,
        "trajectory_records": trajectory_records,
        "carrier_records": carrier_records,
        "text_records": text_records,
        "per_class_csv": per_class_csv,
        "annotation_json": Path(args.annotation_json),
        "split_json": Path(args.split_json),
        "output_dir": output_dir,
    }


def _candidate_ids_for_policy(policy: str, gt_id: str, clip_key: str, known: Set[str], base_ctx: Dict[str, Set[str]], all_ctx: Dict[str, Set[str]], full_ids: Set[str]) -> Set[str]:
    if policy == "base_residual":
        cand = set(base_ctx.get(clip_key, set())) - known
    elif policy == "all_visible_residual":
        cand = set(all_ctx.get(clip_key, set())) - known
    elif policy == "fullY_minus_known":
        cand = set(full_ids) - known
    elif policy == "fullY":
        cand = set(full_ids)
    else:
        raise ValueError(f"unsupported candidate policy: {policy}")
    cand.add(gt_id)
    return cand


def _eval_candidate_set(scores: np.ndarray, cand_ids: Set[str], raw_to_idx: Dict[str, int], gt_id: str) -> Optional[Dict[str, Any]]:
    keep_idx = [raw_to_idx[x] for x in cand_ids if x in raw_to_idx]
    if gt_id not in raw_to_idx:
        return None
    gt_global = raw_to_idx[gt_id]
    if gt_global not in keep_idx:
        keep_idx.append(gt_global)
    if not keep_idx:
        return None
    arr = scores[np.asarray(keep_idx, dtype=np.int64)]
    local_gt = keep_idx.index(gt_global)
    rank, top_local, gt_score, margin = _rank_desc(arr, local_gt)
    top_raw = None
    try:
        top_raw = [k for k, v in raw_to_idx.items() if v == keep_idx[top_local]][0]
    except Exception:
        top_raw = str(keep_idx[top_local])
    return {
        "rank": rank,
        "top1": top_raw,
        "rank1": rank == 1,
        "top5": rank <= 5,
        "margin": margin,
        "candidate_size": len(keep_idx),
    }


def _summarize_rows(rows: List[Record]) -> Dict[str, Any]:
    n = len(rows)
    if not n:
        return {"row_count": 0}
    out: Dict[str, Any] = {"row_count": n}
    for pfx in ["fullY", "base_residual", "all_visible_residual", "fullY_minus_known"]:
        avail = [r for r in rows if f"{pfx}_rank1" in r]
        if not avail:
            continue
        out[f"{pfx}_rank1_rate"] = _rate(sum(1 for r in avail if _truth(r[f"{pfx}_rank1"])), len(avail))
        out[f"{pfx}_top5_rate"] = _rate(sum(1 for r in avail if _truth(r.get(f"{pfx}_top5"))), len(avail))
        out[f"{pfx}_mean_rank"] = sum(_as_float(r.get(f"{pfx}_rank")) for r in avail) / len(avail)
        out[f"{pfx}_mean_margin"] = sum(_as_float(r.get(f"{pfx}_margin")) for r in avail) / len(avail)
        out[f"{pfx}_candidate_size_mean"] = sum(_as_float(r.get(f"{pfx}_candidate_size")) for r in avail) / len(avail)
    if "fullY_rank1_rate" in out:
        for pfx in ["base_residual", "all_visible_residual", "fullY_minus_known"]:
            if f"{pfx}_rank1_rate" in out:
                out[f"{pfx}_rank1_gain_vs_fullY"] = out[f"{pfx}_rank1_rate"] - out["fullY_rank1_rate"]
    # Suppressor removal: fullY top1 wrong but known; after policy top1 correct.
    for pfx in ["base_residual", "all_visible_residual", "fullY_minus_known"]:
        denom = [r for r in rows if str(r.get("fullY_top1")) != str(r.get("gt_raw_id"))]
        if denom:
            out[f"{pfx}_known_top1_suppressor_removed_rate"] = _rate(
                sum(1 for r in denom if _truth(r.get("fullY_top1_in_known")) and _truth(r.get(f"{pfx}_rank1"))),
                len(denom),
            )
            out[f"{pfx}_non_known_suppressor_after_residual_rate"] = _rate(
                sum(1 for r in rows if str(r.get(f"{pfx}_top1")) != str(r.get("gt_raw_id"))),
                len(rows),
            )
    ious = [float(r["matched_gt_iou"]) for r in rows if str(r.get("matched_gt_iou", "")) not in {"", "None"}]
    if ious:
        out["matched_gt_iou_mean"] = sum(ious) / len(ious)
        out["matched_gt_iou_ge_0.3_rate"] = _rate(sum(v >= 0.3 for v in ious), len(ious))
        out["matched_gt_iou_ge_0.5_rate"] = _rate(sum(v >= 0.5 for v in ious), len(ious))
    return out


def run(args: argparse.Namespace) -> Dict[str, Any]:
    paths = _resolve_paths(args)
    out_dir = paths["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    required = ["trajectory_records", "carrier_records", "text_records", "per_class_csv", "annotation_json", "split_json"]
    missing = [str(paths[k]) for k in required if not paths[k].is_file()]
    if missing:
        raise FileNotFoundError("missing required input(s): " + "; ".join(missing))

    base_ids, novel_ids, split_names = load_split(paths["split_json"])
    base_ctx, all_ctx, ann_names = load_annotation_contexts(paths["annotation_json"], base_ids, novel_ids)
    labels, resolved_by_iter, initial_known, label_meta = load_iterative_labels(paths["per_class_csv"], args.variant, str(args.person_raw_id))

    text_raw_ids, text_names, text_mat = load_text_prototypes(paths["text_records"])
    if text_mat.ndim != 2 or text_mat.shape[0] <= 0:
        raise ValueError(f"no text prototypes loaded from {paths['text_records']}")
    raw_to_idx = {rid: idx for idx, rid in enumerate(text_raw_ids)}
    full_ids = set(text_raw_ids)

    carriers = load_carrier_records(paths["carrier_records"])
    trajectories = load_trajectory_records(paths["trajectory_records"])
    gt_match_path = Path(args.gt_match_path) if args.gt_match_path else None
    gt_matches, selected_match_path, match_meta = load_gt_match_rows(gt_match_path, paths["run_root"], args.dataset_name)

    carrier_parent = paths["carrier_records"].parent
    policies = [p.strip() for p in str(args.candidate_policies).split(",") if p.strip()]
    policies = [p for p in policies if p in {"base_residual", "all_visible_residual", "fullY_minus_known"}]
    if not policies:
        raise ValueError("no valid candidate policy requested")

    row_results: List[Record] = []
    failure_examples: List[Record] = []
    counters = Counter()
    t0 = time.perf_counter()
    total_carriers = len(carriers)

    for idx, (tid, crec) in enumerate(carriers.items(), start=1):
        if args.max_rows and len(row_results) >= int(args.max_rows):
            break
        counters["carrier_rows_seen"] += 1
        traj = trajectories.get(tid, {})
        mrow = gt_matches.get(tid, {})
        combined: Record = {}
        combined.update(traj)
        combined.update(crec)
        combined.update(mrow)
        gt_id = _extract_gt_id(combined)
        if gt_id is None:
            counters["skipped_no_gt_match"] += 1
            continue
        if gt_id not in base_ids:
            counters["skipped_gt_not_base"] += 1
            continue
        if gt_id not in labels:
            counters["skipped_no_label_row"] += 1
            continue
        label_row = labels[gt_id]
        if not _truth(label_row.get("resolved", label_row.get("is_resolved", ""))):
            # Still useful but not main residual oracle; keep if requested.
            if not args.include_unresolved:
                counters["skipped_label_unresolved"] += 1
                continue
        iou = _extract_iou(combined)
        if iou is not None and iou < float(args.min_iou):
            counters["skipped_low_iou"] += 1
            continue
        clip_key = _extract_clip_key(combined)
        if clip_key is None:
            counters["skipped_no_clip_key"] += 1
            continue
        if clip_key not in base_ctx and clip_key not in all_ctx:
            counters["skipped_no_context"] += 1
            continue
        if gt_id not in raw_to_idx:
            counters["skipped_gt_no_text_proto"] += 1
            continue
        locator = None
        for k in ["z_norm_path", "traj_z_norm_path", "vector_path", "carrier_path", "z_path"]:
            if k in crec and str(crec.get(k, "")):
                locator = str(crec[k])
                break
        if locator is None:
            counters["skipped_no_vector_locator"] += 1
            continue
        try:
            z = _l2_normalize(read_vector_from_locator(carrier_parent, locator))
        except Exception as exc:
            counters["skipped_vector_read_error"] += 1
            if len(failure_examples) < int(args.top_examples):
                failure_examples.append({"trajectory_id": tid, "reason": "vector_read_error", "error": str(exc)})
            continue
        if int(z.shape[0]) != int(text_mat.shape[1]):
            counters["skipped_dim_mismatch"] += 1
            if len(failure_examples) < int(args.top_examples):
                failure_examples.append({"trajectory_id": tid, "reason": "dim_mismatch", "z_dim": int(z.shape[0]), "text_dim": int(text_mat.shape[1])})
            continue
        scores = text_mat @ z
        cert = str(label_row.get("certificate_type", label_row.get("certificate", "unknown")))
        iteration = _as_int(label_row.get("resolved_at_iteration", label_row.get("iteration", 0)), default=0)
        known = known_before_iteration(resolved_by_iter, initial_known, iteration)
        if gt_id in known:
            # Known classes can still be target rows; never remove the GT itself.
            known = set(known)
            known.discard(gt_id)

        res: Record = {
            "trajectory_id": tid,
            "clip_id": clip_key,
            "gt_raw_id": gt_id,
            "gt_name": ann_names.get(gt_id, split_names.get(gt_id, text_names.get(gt_id, gt_id))),
            "certificate_type": cert,
            "resolved_at_iteration": iteration,
            "known_size_before": len(known),
            "matched_gt_iou": "" if iou is None else float(iou),
        }
        full_eval = _eval_candidate_set(scores, full_ids, raw_to_idx, gt_id)
        if full_eval is None:
            counters["skipped_full_eval_failed"] += 1
            continue
        res.update({
            "fullY_rank": full_eval["rank"],
            "fullY_rank1": full_eval["rank1"],
            "fullY_top5": full_eval["top5"],
            "fullY_top1": full_eval["top1"],
            "fullY_margin": full_eval["margin"],
            "fullY_candidate_size": full_eval["candidate_size"],
            "fullY_top1_in_known": str(full_eval["top1"]) in known,
        })
        for policy in policies:
            cand = _candidate_ids_for_policy(policy, gt_id, clip_key, known, base_ctx, all_ctx, full_ids)
            ev = _eval_candidate_set(scores, cand, raw_to_idx, gt_id)
            if ev is None:
                counters[f"{policy}_eval_failed"] += 1
                continue
            res.update({
                f"{policy}_rank": ev["rank"],
                f"{policy}_rank1": ev["rank1"],
                f"{policy}_top5": ev["top5"],
                f"{policy}_top1": ev["top1"],
                f"{policy}_margin": ev["margin"],
                f"{policy}_candidate_size": ev["candidate_size"],
            })
        row_results.append(res)
        counters["row_scores_used"] += 1
        if len(failure_examples) < int(args.top_examples):
            if any((not _truth(res.get(f"{p}_rank1"))) for p in policies):
                failure_examples.append(dict(res))
        if int(args.progress_every) > 0 and idx % int(args.progress_every) == 0:
            elapsed = max(1e-9, time.perf_counter() - t0)
            print(
                f"[videocutler-residual-oracle] carriers={idx}/{total_carriers} used={len(row_results)} "
                f"rate={idx/elapsed:.1f}/s elapsed={elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )

    # Summaries.
    summary_by_policy: Dict[str, Dict[str, Any]] = {}
    all_summary = _summarize_rows(row_results)
    for p in policies:
        summary_by_policy[p] = {
            "row_count": len(row_results),
            "fullY_rank1_rate": all_summary.get("fullY_rank1_rate"),
            "residual_rank1_rate": all_summary.get(f"{p}_rank1_rate"),
            "residual_top5_rate": all_summary.get(f"{p}_top5_rate"),
            "rank1_gain_vs_fullY": all_summary.get(f"{p}_rank1_gain_vs_fullY"),
            "mean_residual_rank": all_summary.get(f"{p}_mean_rank"),
            "mean_residual_margin": all_summary.get(f"{p}_mean_margin"),
            "mean_candidate_size": all_summary.get(f"{p}_candidate_size_mean"),
            "known_top1_suppressor_removed_rate": all_summary.get(f"{p}_known_top1_suppressor_removed_rate"),
            "non_known_suppressor_after_residual_rate": all_summary.get(f"{p}_non_known_suppressor_after_residual_rate"),
        }

    by_cert_rows: List[Record] = []
    by_cert_group: Dict[str, List[Record]] = defaultdict(list)
    for r in row_results:
        by_cert_group[str(r.get("certificate_type", "unknown"))].append(r)
    for cert, rows in sorted(by_cert_group.items()):
        s = _summarize_rows(rows)
        row: Record = {"certificate_type": cert, "row_count": len(rows), "fullY_rank1_rate": s.get("fullY_rank1_rate")}
        for p in policies:
            row[f"{p}_rank1_rate"] = s.get(f"{p}_rank1_rate")
            row[f"{p}_top5_rate"] = s.get(f"{p}_top5_rate")
            row[f"{p}_rank1_gain_vs_fullY"] = s.get(f"{p}_rank1_gain_vs_fullY")
            row[f"{p}_mean_candidate_size"] = s.get(f"{p}_candidate_size_mean")
        by_cert_rows.append(row)

    # Per-class aggregation.
    by_class_rows: List[Record] = []
    by_class: Dict[str, List[Record]] = defaultdict(list)
    for r in row_results:
        by_class[str(r["gt_raw_id"])].append(r)
    for rid, rows in sorted(by_class.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0])):
        s = _summarize_rows(rows)
        row: Record = {
            "raw_id": rid,
            "class_name": rows[0].get("gt_name", rid),
            "row_count": len(rows),
            "certificate_type": rows[0].get("certificate_type", ""),
            "fullY_rank1_rate": s.get("fullY_rank1_rate"),
        }
        for p in policies:
            row[f"{p}_rank1_rate"] = s.get(f"{p}_rank1_rate")
            row[f"{p}_top5_rate"] = s.get(f"{p}_top5_rate")
            row[f"{p}_rank1_gain_vs_fullY"] = s.get(f"{p}_rank1_gain_vs_fullY")
            row[f"{p}_mean_candidate_size"] = s.get(f"{p}_candidate_size_mean")
        by_class_rows.append(row)

    outputs = {
        "summary_json": str(out_dir / "summary.json"),
        "summary_by_certificate_csv": str(out_dir / "summary_by_certificate.csv"),
        "per_class_csv": str(out_dir / "per_class_videocutler_residual_oracle.csv"),
        "examples_jsonl": str(out_dir / "failure_examples.jsonl"),
        "takeover_md": str(out_dir / "VIDEOCUTLER_RESIDUAL_PEELING_ORACLE_TAKEOVER.md"),
    }
    _write_csv(out_dir / "summary_by_certificate.csv", by_cert_rows)
    _write_csv(out_dir / "per_class_videocutler_residual_oracle.csv", by_class_rows)
    with (out_dir / "failure_examples.jsonl").open("w", encoding="utf-8") as handle:
        for ex in failure_examples:
            handle.write(json.dumps(ex, ensure_ascii=False) + "\n")

    summary = {
        "status": "PASS" if row_results else "FAIL",
        "run_root": str(paths["run_root"]),
        "dataset_name": args.dataset_name,
        "variant": args.variant,
        "min_iou": float(args.min_iou),
        "paths": {k: str(v) for k, v in paths.items() if isinstance(v, Path)},
        "gt_match_source": selected_match_path,
        "gt_match_meta": match_meta,
        "base_count": len(base_ids),
        "novel_count": len(novel_ids),
        "text_proto_count": len(text_raw_ids),
        "trajectory_record_count": len(trajectories),
        "carrier_record_count": len(carriers),
        "label_meta": label_meta,
        "counters": dict(counters),
        "row_scores_used": len(row_results),
        "summary_all_rows": all_summary,
        "summary_by_policy": summary_by_policy,
        "outputs": outputs,
        "warnings": [],
    }
    if not selected_match_path:
        summary["warnings"].append("No separate GT match file was found; only GT fields embedded in trajectory/carrier rows could be used.")
    if not row_results:
        summary["warnings"].append("No evaluable VideoCutLER rows. Check GT match source, trajectory IDs, min_iou, carrier paths, and text prototype dimensions.")

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# VideoCutLER Residual Peeling Oracle Audit",
        "",
        f"- status: {summary['status']}",
        f"- row_scores_used: {len(row_results)}",
        f"- gt_match_source: {selected_match_path}",
        f"- min_iou: {float(args.min_iou)}",
        "",
        "## Summary by policy",
    ]
    for p, vals in summary_by_policy.items():
        lines.append(f"- {p}: fullY_rank1={vals.get('fullY_rank1_rate')}, residual_rank1={vals.get('residual_rank1_rate')}, gain={vals.get('rank1_gain_vs_fullY')}, cand_size={vals.get('mean_candidate_size')}")
    lines.append("")
    lines.append("## Counters")
    for k, v in sorted(counters.items()):
        lines.append(f"- {k}: {v}")
    (out_dir / "VIDEOCUTLER_RESIDUAL_PEELING_ORACLE_TAKEOVER.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only VideoCutLER-carrier residual-peeling oracle audit")
    p.add_argument("--run_root", required=True)
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--variant", default="person_aware")
    p.add_argument("--repo_root", default="")
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--annotation_json", required=True)
    p.add_argument("--split_json", required=True)
    p.add_argument("--per_class_csv", default="")
    p.add_argument("--trajectory_records_jsonl", default="")
    p.add_argument("--carrier_records_jsonl", default="")
    p.add_argument("--text_records_jsonl", default="")
    p.add_argument("--gt_match_path", default="")
    p.add_argument("--candidate_policies", default="base_residual,all_visible_residual,fullY_minus_known")
    p.add_argument("--person_raw_id", default="773")
    p.add_argument("--min_iou", type=float, default=0.5)
    p.add_argument("--include_unresolved", action="store_true")
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--top_examples", type=int, default=128)
    p.add_argument("--progress_every", type=int, default=5000)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    summary = run(args)
    print(json.dumps({
        "status": summary.get("status"),
        "output_dir": summary.get("outputs", {}).get("summary_json", ""),
        "row_scores_used": summary.get("row_scores_used"),
        "summary_by_policy": summary.get("summary_by_policy"),
        "counters": summary.get("counters"),
        "warnings": summary.get("warnings"),
    }, ensure_ascii=False, indent=2))
    if summary.get("status") != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
