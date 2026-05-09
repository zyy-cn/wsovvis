#!/usr/bin/env python3
"""A8 Pclip clip-local GT class-mean projector audit.

This is a read-only oracle diagnostic: fit a text->DINO linear projector from
clip-local GT class visual prototypes. For each clip v and GT class c, all GT
trajectory carriers of class c inside the clip are averaged into m_{v,c}; the
projector is then fit from text anchor t_c to m_{v,c}.

It does not use weak labels, clip-level Hungarian assignment, or any A8
checkpoint. It *does* use row-level GT class bindings to build clip-local visual
prototypes, so it is an oracle upper diagnostic rather than a weakly-supervised
training protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _ensure_repo(repo_root: Path) -> None:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _load_gtceil(repo_root: Path):
    path = repo_root / "tools" / "run_a8_gt_trajectory_semantic_ceiling_eval.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("_a8_gtceil_helper", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if str(k) not in fields:
                fields.append(str(k))
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fields:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


def _sha256(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _as_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or str(x).strip() == "":
            return default
        return int(float(str(x)))
    except Exception:
        return default


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def _finite_mean(xs: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else None


def _summary_ranks(ranks: Sequence[int]) -> Dict[str, Any]:
    n = int(len(ranks))
    if n <= 0:
        return {"count": 0, "rank@1": 0.0, "rank@5": 0.0, "rank@10": 0.0, "rank@20": 0.0, "rank@50": 0.0, "mean_rank": None, "median_rank": None}
    arr = np.asarray(ranks, dtype=np.float64)
    return {
        "count": n,
        "rank@1": float(np.mean(arr <= 1)),
        "rank@5": float(np.mean(arr <= 5)),
        "rank@10": float(np.mean(arr <= 10)),
        "rank@20": float(np.mean(arr <= 20)),
        "rank@50": float(np.mean(arr <= 50)),
        "mean_rank": float(np.mean(arr)),
        "median_rank": float(np.median(arr)),
    }


def _load_npz_first(path: Path) -> Tuple[np.ndarray, str]:
    z = np.load(path)
    for key in ("protos", "features", "arr_0", "llama_hidden_mean", "clip_of_llm_mean", "llama_direct_concept_mean"):
        if key in z:
            return np.asarray(z[key]), key
    keys = list(z.keys())
    if not keys:
        raise RuntimeError(f"empty npz: {path}")
    return np.asarray(z[keys[0]]), str(keys[0])


def _payload_for_variant(root: Path, variant: str) -> Path:
    table = {
        "clip_of_llm_mean": root / "payload" / "clip_of_llm_mean.fp16.npz",
        "llama_hidden_mean": root / "payload" / "llama_hidden_mean.fp16.npz",
        "llama_direct_concept_mean": root / "payload" / "llama_direct_concept_mean.fp16.npz",
    }
    if variant not in table:
        raise ValueError(f"unsupported external text bank variant: {variant}")
    if not table[variant].is_file():
        raise FileNotFoundError(table[variant])
    return table[variant]


def _load_lvvis_classes(root: Path) -> Tuple[List[int], Dict[int, str]]:
    path = root / "lvvis_class_names.json"
    payload = _read_json(path)
    rows = payload.get("classes", []) if isinstance(payload, Mapping) else payload
    ids: List[int] = []
    names: Dict[int, str] = {}
    for r in rows:
        if not isinstance(r, Mapping):
            continue
        rid = _as_int(r.get("raw_id"))
        if rid is None:
            continue
        ids.append(int(rid))
        names[int(rid)] = str(r.get("name", r.get("class_name", f"raw_id_{rid}")))
    if not ids or ids != sorted(ids):
        raise RuntimeError(f"invalid class list: {path}")
    return ids, names


def _load_external_text_bank(root: Path, variant: str) -> Tuple[List[int], np.ndarray, Dict[int, str], Dict[str, Any]]:
    ids, names = _load_lvvis_classes(root)
    p = _payload_for_variant(root, variant)
    arr, key = _load_npz_first(p)
    if arr.ndim != 2 or int(arr.shape[0]) != len(ids):
        raise RuntimeError(f"bad payload shape={arr.shape}, ids={len(ids)}")
    arr = _l2(np.asarray(arr, dtype=np.float32))
    manifest = root / "manifest.json"
    meta = {
        "variant": variant,
        "root": str(root),
        "feature_dim": int(arr.shape[1]),
        "class_count": int(len(ids)),
        "payload_path": str(p),
        "payload_array_key": key,
        "payload_sha256": _sha256(p),
        "manifest_path": str(manifest) if manifest.is_file() else "",
        "manifest_sha256": _sha256(manifest) if manifest.is_file() else "",
    }
    return ids, arr, names, meta


def _load_clip_text_bank(asset_root: Path, dataset_name: str) -> Tuple[List[int], np.ndarray, Dict[int, str], Dict[str, Any]]:
    from videocutler.ext_stageb_ovvis.eval.g8_bridge import load_text_vocab_with_names  # type: ignore
    ids, _records, mat, names = load_text_vocab_with_names(asset_root, dataset_name)
    out_ids = [int(x) for x in ids]
    name_map = {int(k): str(v) for k, v in dict(names).items()}
    mat = _l2(np.asarray(mat, dtype=np.float32))
    return out_ids, mat, name_map, {"variant": "clip_current", "feature_dim": int(mat.shape[1]), "class_count": int(len(out_ids))}


def _load_text_bank(asset_root: Path, train_dataset_name: str, variant: str, visual_only_root: Path, direct_root: Path) -> Tuple[List[int], np.ndarray, Dict[int, str], Dict[str, Any]]:
    if variant == "clip_current":
        return _load_clip_text_bank(asset_root, train_dataset_name)
    if variant in {"clip_of_llm_mean", "llama_hidden_mean"}:
        return _load_external_text_bank(visual_only_root, variant)
    if variant == "llama_direct_concept_mean":
        return _load_external_text_bank(direct_root, variant)
    raise ValueError(f"unsupported variant: {variant}")


def _visible_ids(path: Path) -> List[int]:
    ids = []
    for r in _read_csv(path):
        rid = _as_int(r.get("raw_id"))
        if rid is not None and str(r.get("in_row_gap", "0")).strip() == "1":
            ids.append(int(rid))
    ids = sorted(set(ids))
    if len(ids) != 525:
        raise RuntimeError(f"expected visible525, got {len(ids)} from {path}")
    return ids


def _rows_and_carriers(gtceil: Any, asset_root: Path, dataset_name: str, ann: Path, max_rows: int) -> Tuple[List[Mapping[str, Any]], np.ndarray, Dict[str, Any]]:
    rows0, meta = gtceil._candidate_source_rows(
        gt_carrier_path=asset_root / "carrier_bank_gt" / dataset_name / "carrier_records.jsonl",
        gt_identity_path=asset_root / "carrier_bank_gt" / dataset_name / "gt_carrier_identity_binding.jsonl",
        gt_trajectory_path=asset_root / "exports_gt" / dataset_name / "trajectory_records.jsonl",
        annotation_json=ann,
        max_rows=int(max_rows or 0),
    )
    carrier, keep, counters = gtceil._load_carrier_matrix(
        gt_carrier_path=asset_root / "carrier_bank_gt" / dataset_name / "carrier_records.jsonl",
        rows=rows0,
    )
    rows = [rows0[i] for i in keep]
    return rows, _l2(np.asarray(carrier, dtype=np.float32)), {"source_meta": meta, "vector_counters": dict(counters), "row_count": len(rows)}


def _row_clip_key(row: Mapping[str, Any]) -> str:
    """Return a stable clip key from GT helper rows.

    Different A8 helpers have used slightly different field names across
    snapshots. This function accepts the known aliases and falls back to
    video_id only if no finer clip key is present.
    """
    for key in ("clip_id", "clip_key", "video_clip_id", "video_id"):
        val = row.get(key)
        if val is not None and str(val).strip() != "":
            return str(val)
    tid = str(row.get("trajectory_id", ""))
    # Known trajectory ids often look like source:dataset:clip:row.
    parts = tid.split(":")
    if len(parts) >= 4:
        return parts[-2]
    return "unknown_clip"


def _clip_class_prototypes(rows: Sequence[Mapping[str, Any]], carrier: np.ndarray) -> Tuple[Dict[Tuple[str, int], np.ndarray], Dict[Tuple[str, int], int], Dict[int, int]]:
    """Build m_{v,c}=mean({z_i | clip(i)=v, gt(i)=c})."""
    buf: Dict[Tuple[str, int], List[np.ndarray]] = defaultdict(list)
    for i, row in enumerate(rows):
        rid = _as_int(row.get("raw_category_id"))
        if rid is None or i >= carrier.shape[0]:
            continue
        v = np.asarray(carrier[i], dtype=np.float32)
        if not np.isfinite(v).all():
            continue
        key = (_row_clip_key(row), int(rid))
        buf[key].append(v)
    proto: Dict[Tuple[str, int], np.ndarray] = {}
    counts: Dict[Tuple[str, int], int] = {}
    class_counts: Dict[int, int] = defaultdict(int)
    for key, vecs in buf.items():
        proto[key] = _l2(np.mean(np.stack(vecs, axis=0), axis=0, keepdims=True))[0]
        counts[key] = int(len(vecs))
        class_counts[int(key[1])] += 1
    return proto, counts, dict(class_counts)


def _global_class_prototypes(rows: Sequence[Mapping[str, Any]], carrier: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    """Auxiliary global class prototypes for reporting only."""
    buf: Dict[int, List[np.ndarray]] = defaultdict(list)
    for i, row in enumerate(rows):
        rid = _as_int(row.get("raw_category_id"))
        if rid is None or i >= carrier.shape[0]:
            continue
        v = np.asarray(carrier[i], dtype=np.float32)
        if np.isfinite(v).all():
            buf[int(rid)].append(v)
    proto: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    for rid, vecs in buf.items():
        proto[int(rid)] = _l2(np.mean(np.stack(vecs, axis=0), axis=0, keepdims=True))[0]
        counts[int(rid)] = int(len(vecs))
    return proto, counts


def _submatrix(ids_all: Sequence[int], mat: np.ndarray, ids: Sequence[int]) -> np.ndarray:
    idx = {int(r): i for i, r in enumerate(ids_all)}
    missing = [int(r) for r in ids if int(r) not in idx]
    if missing:
        raise RuntimeError(f"missing text ids: count={len(missing)} first={missing[:10]}")
    return np.asarray(mat[[idx[int(r)] for r in ids]], dtype=np.float32)


def _fit_map(x: np.ndarray, y: np.ndarray, method: str, alpha: float) -> np.ndarray:
    x = _l2(np.asarray(x, dtype=np.float32))
    y = _l2(np.asarray(y, dtype=np.float32))
    method = str(method).lower()
    if method == "ridge":
        n, d = int(x.shape[0]), int(x.shape[1])
        if n <= d:
            # Dual ridge: W = X^T (X X^T + alpha I)^-1 Y.
            a = x @ x.T + float(alpha) * np.eye(n, dtype=np.float32)
            coef = np.linalg.solve(a.astype(np.float64), y.astype(np.float64))
            w = x.T.astype(np.float64) @ coef
        else:
            # Primal ridge avoids a large n x n system for clip-class fits.
            a = x.T @ x + float(alpha) * np.eye(d, dtype=np.float32)
            b = x.T @ y
            w = np.linalg.solve(a.astype(np.float64), b.astype(np.float64))
        return np.asarray(w, dtype=np.float32)
    if method in {"least_squares", "lstsq"}:
        w, *_ = np.linalg.lstsq(x.astype(np.float64), y.astype(np.float64), rcond=None)
        return np.asarray(w, dtype=np.float32)
    raise ValueError(f"unsupported method={method}; use ridge,least_squares")


def _project(text: np.ndarray, w: np.ndarray) -> np.ndarray:
    return _l2(np.asarray(text, dtype=np.float32) @ np.asarray(w, dtype=np.float32))


def _eval_rows(rows: Sequence[Mapping[str, Any]], carrier: np.ndarray, candidate_ids: Sequence[int], projected_candidates: np.ndarray, names: Mapping[int, str], group_of_gt: Optional[Mapping[int, str]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    z = _l2(np.asarray(carrier, dtype=np.float32))
    cand = list(map(int, candidate_ids))
    cindex = {rid: i for i, rid in enumerate(cand)}
    scores = z @ np.asarray(projected_candidates, dtype=np.float32).T
    per_rows: List[Dict[str, Any]] = []
    by_group: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        gt = _as_int(row.get("raw_category_id"))
        if gt is None or int(gt) not in cindex:
            continue
        s = np.asarray(scores[i], dtype=np.float64)
        order = np.argsort(-s, kind="mergesort")
        gt_pos = cindex[int(gt)]
        rank = int(np.where(order == gt_pos)[0][0]) + 1
        top1 = cand[int(order[0])]
        wrong = np.delete(s, gt_pos)
        nearest_wrong = float(np.max(wrong)) if wrong.size else float("nan")
        margin = float(s[gt_pos] - nearest_wrong) if math.isfinite(nearest_wrong) else float("nan")
        group = group_of_gt.get(int(gt), "all") if group_of_gt else "all"
        by_group[group].append(rank)
        by_group["all"].append(rank) if group != "all" else None
        per_rows.append({
            "trajectory_id": row.get("trajectory_id"),
            "video_id": row.get("video_id"),
            "clip_id": row.get("clip_id"),
            "gt_raw_id": int(gt),
            "gt_name": names.get(int(gt), f"raw_id_{gt}"),
            "gt_group": group,
            "rank": rank,
            "top1_raw_id": top1,
            "top1_name": names.get(top1, f"raw_id_{top1}"),
            "top1_is_gt": int(top1 == int(gt)),
            "top1_is_person": int(top1 == 773),
            "gt_score": float(s[gt_pos]),
            "top1_score": float(s[int(order[0])]),
            "margin_gt_vs_top_wrong": margin,
        })
    summaries: List[Dict[str, Any]] = []
    for g in sorted(by_group.keys(), key=lambda x: {"all": 0, "base": 1, "novel": 2}.get(x, 9)):
        ranks = by_group[g]
        rs = [r for r in per_rows if (g == "all" or r.get("gt_group") == g)]
        summaries.append({
            "group": g,
            "class_count": len({int(r["gt_raw_id"]) for r in rs}),
            **_summary_ranks(ranks),
            "mean_margin_gt_vs_top_wrong": _finite_mean([float(r["margin_gt_vs_top_wrong"]) for r in rs]),
            "top1_person_rate": float(np.mean([int(r["top1_is_person"]) for r in rs])) if rs else 0.0,
        })
    return per_rows, summaries


def _eval_global_proto(projected_all: np.ndarray, ids_all: Sequence[int], visual_proto: Mapping[int, np.ndarray], eval_ids: Sequence[int], candidate_ids: Sequence[int]) -> Dict[str, Any]:
    idx_all = {int(r): i for i, r in enumerate(ids_all)}
    cand = [int(r) for r in candidate_ids if int(r) in visual_proto and int(r) in idx_all]
    ev = [int(r) for r in eval_ids if int(r) in visual_proto and int(r) in idx_all]
    if not ev or not cand:
        return {"eval_count": 0, "candidate_count": len(cand)}
    cand_proj = projected_all[[idx_all[r] for r in cand]]
    cand_vis = np.stack([visual_proto[r] for r in cand], axis=0).astype(np.float32)
    t2v, v2t = [], []
    cand_pos = {rid: i for i, rid in enumerate(cand)}
    for rid in ev:
        qtxt = projected_all[idx_all[rid]]
        sims = _l2(cand_vis) @ qtxt
        order = np.argsort(-sims, kind="mergesort")
        pos = cand_pos[rid]
        t2v.append(int(np.where(order == pos)[0][0]) + 1)
        qvis = _l2(visual_proto[rid].reshape(1, -1))[0]
        sims2 = _l2(cand_proj) @ qvis
        order2 = np.argsort(-sims2, kind="mergesort")
        v2t.append(int(np.where(order2 == pos)[0][0]) + 1)
    out = {"eval_count": len(ev), "candidate_count": len(cand)}
    out.update({f"t2v_{k}": v for k, v in _summary_ranks(t2v).items()})
    out.update({f"v2t_{k}": v for k, v in _summary_ranks(v2t).items()})
    return out


def _eval_clip_class_proto(
    projected_all: np.ndarray,
    ids_all: Sequence[int],
    clip_proto: Mapping[Tuple[str, int], np.ndarray],
    candidate_ids: Sequence[int],
) -> Dict[str, Any]:
    """Evaluate clip-local visual prototypes against class text anchors.

    v2t: each m_{v,c} ranks the correct class text t_c among candidate ids.
    t2v: each text t_c ranks the nearest positive clip-local prototype of the
         same class among all clip-local prototypes. Multiple positives are
         allowed; rank is the first positive.
    """
    idx_all = {int(r): i for i, r in enumerate(ids_all)}
    cand = [int(r) for r in candidate_ids if int(r) in idx_all]
    cand_set = set(cand)
    pairs = [(ck, int(rid)) for (ck, rid) in clip_proto.keys() if int(rid) in cand_set]
    if not cand or not pairs:
        return {"eval_count": 0, "candidate_text_count": len(cand), "candidate_proto_count": len(pairs)}
    cand_proj = projected_all[[idx_all[r] for r in cand]]
    cand_pos = {rid: i for i, rid in enumerate(cand)}
    proto_mat = np.stack([clip_proto[p] for p in pairs], axis=0).astype(np.float32)
    proto_mat = _l2(proto_mat)

    v2t: List[int] = []
    for pi, (_clip, rid) in enumerate(pairs):
        sims = cand_proj @ proto_mat[pi]
        order = np.argsort(-sims, kind="mergesort")
        pos = cand_pos[int(rid)]
        v2t.append(int(np.where(order == pos)[0][0]) + 1)

    t2v: List[int] = []
    pair_raw = np.asarray([rid for (_clip, rid) in pairs], dtype=np.int64)
    for rid in sorted({int(rid) for (_clip, rid) in pairs}):
        q = projected_all[idx_all[rid]]
        sims = proto_mat @ q
        order = np.argsort(-sims, kind="mergesort")
        positive_positions = np.where(pair_raw[order] == int(rid))[0]
        if positive_positions.size > 0:
            t2v.append(int(positive_positions[0]) + 1)

    out = {
        "eval_pair_count": len(pairs),
        "eval_class_count": len({int(rid) for (_clip, rid) in pairs}),
        "candidate_text_count": len(cand),
        "candidate_proto_count": len(pairs),
    }
    out.update({f"v2t_{k}": v for k, v in _summary_ranks(v2t).items()})
    out.update({f"t2v_{k}": v for k, v in _summary_ranks(t2v).items()})
    return out


def _load_split(path: Path) -> Tuple[set[int], set[int]]:
    obj = _read_json(path)
    def collect(keys: Sequence[str]) -> set[int]:
        for k in keys:
            if k in obj:
                val = obj[k]
                if isinstance(val, Mapping):
                    val = val.keys()
                return {int(float(x)) for x in val}
        return set()
    base = collect(["base_raw_ids", "base_ids", "base", "base_classes"])
    novel = collect(["novel_raw_ids", "novel_ids", "novel", "novel_classes"])
    if not base or not novel:
        # Fallback: support list of dicts with split field.
        rows = obj.get("classes", []) if isinstance(obj, Mapping) else []
        for r in rows:
            if not isinstance(r, Mapping):
                continue
            rid = _as_int(r.get("raw_id", r.get("id")))
            split = str(r.get("split", "")).lower()
            if rid is not None and split == "base":
                base.add(int(rid))
            elif rid is not None and split == "novel":
                novel.add(int(rid))
    if not base or not novel:
        raise RuntimeError(f"cannot parse base/novel split from {path}")
    return base, novel


def main() -> int:
    ap = argparse.ArgumentParser(description="Pclip GT class-mean projector audit")
    ap.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    ap.add_argument("--asset_root", default="/mnt/sda/zyy/code/wsovvis_asserts")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--train_dataset_name", default="lvvis_train_base")
    ap.add_argument("--val_dataset_name", default="lvvis_val")
    ap.add_argument("--train_annotation_json", default="")
    ap.add_argument("--val_annotation_json", default="")
    ap.add_argument("--visible_csv", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--visual_only_root", default="/mnt/sda/zyy/code/wsovvis_asserts/text_bank_llama3/lvvis/lvvis_visual_only_v1")
    ap.add_argument("--direct_concept_root", default="/mnt/sda/zyy/code/wsovvis_asserts/text_bank_llama3/lvvis/llama3_direct_concept_v1")
    ap.add_argument("--variants", default="clip_current,clip_of_llm_mean,llama_hidden_mean,llama_direct_concept_mean")
    ap.add_argument("--methods", default="ridge,least_squares")
    ap.add_argument("--ridge_alpha", type=float, default=1e-2)
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _ensure_repo(repo_root)
    gtceil = _load_gtceil(repo_root)

    train_ann = Path(args.train_annotation_json).expanduser().resolve() if args.train_annotation_json else repo_root / "dataset" / "LV-VIS" / "annotations" / "train_instances.json"
    val_ann = Path(args.val_annotation_json).expanduser().resolve() if args.val_annotation_json else repo_root / "dataset" / "LV-VIS" / "annotations" / "val_instances.json"
    visible = _visible_ids(Path(args.visible_csv).expanduser().resolve())
    base_ids, novel_ids = _load_split(Path(args.split_json).expanduser().resolve())

    train_rows, train_carrier, train_meta = _rows_and_carriers(gtceil, asset_root, args.train_dataset_name, train_ann, int(args.max_rows))
    val_rows, val_carrier, val_meta = _rows_and_carriers(gtceil, asset_root, args.val_dataset_name, val_ann, int(args.max_rows))
    train_clip_proto, train_clip_counts, train_clip_class_counts = _clip_class_prototypes(train_rows, train_carrier)
    val_clip_proto, val_clip_counts, val_clip_class_counts = _clip_class_prototypes(val_rows, val_carrier)
    train_global_proto, train_global_counts = _global_class_prototypes(train_rows, train_carrier)
    val_global_proto, val_global_counts = _global_class_prototypes(val_rows, val_carrier)

    variants = [x.strip() for x in str(args.variants).split(",") if x.strip()]
    methods = [x.strip() for x in str(args.methods).split(",") if x.strip()]
    summary_rows: List[Dict[str, Any]] = []
    run_reports: List[Dict[str, Any]] = []

    for variant in variants:
        text_ids, text_mat, names, tb_meta = _load_text_bank(asset_root, args.train_dataset_name, variant, Path(args.visual_only_root).expanduser().resolve(), Path(args.direct_concept_root).expanduser().resolve())
        text_id_set = set(map(int, text_ids))
        fit_pairs = sorted([(str(ck), int(rid)) for (ck, rid) in train_clip_proto.keys() if int(rid) in visible and int(rid) in text_id_set])
        fit_ids = sorted({int(rid) for (_ck, rid) in fit_pairs})
        if len(fit_pairs) < 2 or len(fit_ids) < 2:
            raise RuntimeError(f"not enough fit pairs/classes for {variant}: pairs={len(fit_pairs)} classes={len(fit_ids)}")
        idx_all_text = {int(r): i for i, r in enumerate(text_ids)}
        x_fit = np.stack([text_mat[idx_all_text[int(rid)]] for (_ck, rid) in fit_pairs], axis=0).astype(np.float32)
        y_fit = np.stack([train_clip_proto[(ck, int(rid))] for (ck, rid) in fit_pairs], axis=0).astype(np.float32)
        candidate_visible_ids = [rid for rid in visible if rid in text_id_set]
        candidate_full_ids = [int(rid) for rid in text_ids]
        for method in methods:
            run_name = f"{variant}__{method}"
            run_dir = output_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            w = _fit_map(x_fit, y_fit, method, float(args.ridge_alpha))
            # Save weight remotely; it is intentionally not needed for local report transfer.
            np.savez_compressed(
                run_dir / "pclip_gt_class_mean_mapper.npz",
                W=w.astype(np.float32),
                fit_raw_ids=np.asarray([int(rid) for (_ck, rid) in fit_pairs], dtype=np.int64),
            )
            proj_all = _project(text_mat, w)
            text_idx = {int(r): i for i, r in enumerate(text_ids)}
            proj_visible = proj_all[[text_idx[r] for r in candidate_visible_ids]]
            proj_full = proj_all[[text_idx[r] for r in candidate_full_ids]]

            train_per, train_sum = _eval_rows(train_rows, train_carrier, candidate_visible_ids, proj_visible, names)
            val_vis_per, val_vis_sum = _eval_rows(val_rows, val_carrier, candidate_visible_ids, proj_visible, names)
            split_map = {rid: "base" for rid in base_ids}
            split_map.update({rid: "novel" for rid in novel_ids})
            val_full_per, val_full_sum = _eval_rows(val_rows, val_carrier, candidate_full_ids, proj_full, names, group_of_gt=split_map)
            proto_train = _eval_clip_class_proto(proj_all, text_ids, train_clip_proto, candidate_visible_ids)
            proto_val = _eval_clip_class_proto(proj_all, text_ids, val_clip_proto, candidate_visible_ids)
            global_proto_train = _eval_global_proto(proj_all, text_ids, train_global_proto, fit_ids, candidate_visible_ids)
            global_proto_val = _eval_global_proto(proj_all, text_ids, val_global_proto, [rid for rid in visible if rid in val_global_proto], candidate_visible_ids)

            _write_csv(run_dir / "train_visible525_per_row.csv", train_per)
            _write_csv(run_dir / "val_visible525_per_row.csv", val_vis_per)
            _write_csv(run_dir / "val_full_vocab_per_row.csv", val_full_per)
            _write_json(run_dir / "summary.json", {
                "status": "PASS",
                "name": run_name,
                "definition": "Pclip GT class-mean text-to-DINO projector: fit repeated class text anchors to clip-local GT class visual means m_{v,c}.",
                "variant": variant,
                "method": method,
                "ridge_alpha": float(args.ridge_alpha),
                "text_bank": tb_meta,
                "fit_pair_count": len(fit_pairs),
                "fit_class_count": len(fit_ids),
                "train_clip_class_count": len(train_clip_proto),
                "val_clip_class_count": len(val_clip_proto),
                "candidate_visible_count": len(candidate_visible_ids),
                "candidate_full_count": len(candidate_full_ids),
                "train_visible525": train_sum,
                "val_visible525": val_vis_sum,
                "val_full_vocab": val_full_sum,
                "clip_class_proto_train": proto_train,
                "clip_class_proto_val": proto_val,
                "global_class_proto_train_for_reference": global_proto_train,
                "global_class_proto_val_for_reference": global_proto_val,
                "artifacts": {
                    "mapper_npz": str(run_dir / "pclip_gt_class_mean_mapper.npz"),
                    "train_visible525_per_row_csv": str(run_dir / "train_visible525_per_row.csv"),
                    "val_visible525_per_row_csv": str(run_dir / "val_visible525_per_row.csv"),
                    "val_full_vocab_per_row_csv": str(run_dir / "val_full_vocab_per_row.csv"),
                },
                "policy": {
                    "uses_weak_labels": False,
                    "uses_hungarian": False,
                    "uses_row_level_gt_for_training": True,
                    "uses_clip_class_gt_mean_for_training": True,
                    "fit_scope": "train_visible_525_clip_class_gt_mean_prototypes",
                },
            })

            def pick(summary: List[Mapping[str, Any]], group: str, key: str) -> Any:
                for r in summary:
                    if r.get("group") == group:
                        return r.get(key)
                return None

            row = {
                "name": run_name,
                "variant": variant,
                "method": method,
                "feature_dim": tb_meta.get("feature_dim"),
                "fit_pair_count": len(fit_pairs),
                "fit_class_count": len(fit_ids),
                "train_clip_class_count": len(train_clip_proto),
                "val_clip_class_count": len(val_clip_proto),
                "train_vis_rank@1": pick(train_sum, "all", "rank@1"),
                "train_vis_rank@5": pick(train_sum, "all", "rank@5"),
                "train_vis_mean_rank": pick(train_sum, "all", "mean_rank"),
                "val_vis_rank@1": pick(val_vis_sum, "all", "rank@1"),
                "val_vis_rank@5": pick(val_vis_sum, "all", "rank@5"),
                "val_vis_mean_rank": pick(val_vis_sum, "all", "mean_rank"),
                "val_all_rank@1": pick(val_full_sum, "all", "rank@1"),
                "val_base_rank@1": pick(val_full_sum, "base", "rank@1"),
                "val_novel_rank@1": pick(val_full_sum, "novel", "rank@1"),
                "val_all_mean_rank": pick(val_full_sum, "all", "mean_rank"),
                "val_base_mean_rank": pick(val_full_sum, "base", "mean_rank"),
                "val_novel_mean_rank": pick(val_full_sum, "novel", "mean_rank"),
                "proto_train_t2v_rank@1": proto_train.get("t2v_rank@1"),
                "proto_train_v2t_rank@1": proto_train.get("v2t_rank@1"),
                "proto_val_t2v_rank@1": proto_val.get("t2v_rank@1"),
                "proto_val_v2t_rank@1": proto_val.get("v2t_rank@1"),
                "global_proto_val_t2v_rank@1": global_proto_val.get("t2v_rank@1"),
                "summary_json": str(run_dir / "summary.json"),
            }
            summary_rows.append(row)
            run_reports.append(row)

    _write_csv(output_root / "pclip_gt_class_mean_summary.csv", summary_rows)
    report = {
        "status": "PASS",
        "definition": "Full Pclip GT class-mean projector audit. Direct clip-local GT class-mean supervision; no weak supervision; row-level train/val ranking and val full-vocab base/novel ranking.",
        "output_root": str(output_root),
        "train_dataset_name": args.train_dataset_name,
        "val_dataset_name": args.val_dataset_name,
        "visible_csv": str(Path(args.visible_csv).expanduser().resolve()),
        "split_json": str(Path(args.split_json).expanduser().resolve()),
        "variants": variants,
        "methods": methods,
        "run_count": len(summary_rows),
        "train_meta": train_meta,
        "val_meta": val_meta,
        "prototype_counts": {
            "train_clip_class_count": len(train_clip_proto),
            "val_clip_class_count": len(val_clip_proto),
            "train_global_class_count": len(train_global_proto),
            "val_global_class_count": len(val_global_proto),
        },
        "artifacts": {"summary_csv": str(output_root / "pclip_gt_class_mean_summary.csv")},
        "rows": run_reports,
    }
    _write_json(output_root / "pclip_gt_class_mean_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
