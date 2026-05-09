#!/usr/bin/env python3
"""A8 train/val class-structure shift audit for visible525 GT carriers.

Read-only diagnostic. It audits whether the visible525 classes have stable
visual geometry between train and val, which classes drift most, and whether
row-level errors are explained by intra-class spread versus inter-class
separation.

It intentionally does not train, mutate checkpoints, or define a new primary
metric. It works on GT trajectory carriers and identity bindings only.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def _repo_root_default() -> Path:
    return Path(__file__).resolve().parents[1]


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


def _as_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None or str(x).strip() == "":
            return default
        return int(float(str(x)))
    except Exception:
        return default


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def _cos(x: np.ndarray) -> np.ndarray:
    z = _l2(x)
    return z @ z.T


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    if len(x) == 0:
        return ranks
    sx = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sx[end] == sx[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return float("nan")
    ra, rb = _rankdata_average(a[m]), _rankdata_average(b[m])
    if float(np.std(ra)) <= 0 or float(np.std(rb)) <= 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _upper(mat: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    n = int(mat.shape[0])
    iu = np.triu_indices(n, k=1)
    vals = np.asarray(mat[iu], dtype=np.float64)
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        vals = vals[v[iu[0]] & v[iu[1]]]
    return vals


def _topk(sim: np.ndarray, ids: Sequence[int], i: int, k: int) -> List[int]:
    row = np.asarray(sim[i], dtype=np.float64).copy()
    if i < len(row):
        row[i] = -np.inf
    row[~np.isfinite(row)] = -np.inf
    if not np.isfinite(row).any():
        return []
    order = np.argsort(-row, kind="mergesort")[: max(1, int(k))]
    return [int(ids[int(j)]) for j in order if np.isfinite(row[int(j)])]


def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return float("nan")
    return float(len(sa & sb) / max(len(sa | sb), 1))


def _finite(xs: Iterable[float]) -> List[float]:
    out = []
    for x in xs:
        try:
            v = float(x)
            if math.isfinite(v):
                out.append(v)
        except Exception:
            pass
    return out


def _mean(xs: Iterable[float]) -> float:
    vals = _finite(xs)
    return float(np.mean(vals)) if vals else float("nan")


def _median(xs: Iterable[float]) -> float:
    vals = _finite(xs)
    return float(np.median(vals)) if vals else float("nan")


def _percentile(xs: Iterable[float], q: float) -> float:
    vals = _finite(xs)
    return float(np.percentile(vals, q)) if vals else float("nan")


def _summary_numeric(rows: Sequence[Mapping[str, Any]], key: str, prefix: str = "") -> Dict[str, Any]:
    vals = [_safe_float(r.get(key)) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {
            prefix + key + "_mean": None,
            prefix + key + "_median": None,
            prefix + key + "_p90": None,
            prefix + key + "_max": None,
        }
    arr = np.asarray(vals, dtype=np.float64)
    return {
        prefix + key + "_mean": float(np.mean(arr)),
        prefix + key + "_median": float(np.median(arr)),
        prefix + key + "_p90": float(np.percentile(arr, 90)),
        prefix + key + "_max": float(np.max(arr)),
    }


def _summary_ranks(ranks: Sequence[int], prefix: str = "") -> Dict[str, Any]:
    if not ranks:
        return {
            prefix + "count": 0,
            prefix + "rank@1": 0.0,
            prefix + "rank@5": 0.0,
            prefix + "rank@10": 0.0,
            prefix + "rank@20": 0.0,
            prefix + "rank@50": 0.0,
            prefix + "mean_rank": None,
            prefix + "median_rank": None,
        }
    arr = np.asarray(ranks, dtype=np.float64)
    return {
        prefix + "count": int(len(arr)),
        prefix + "rank@1": float(np.mean(arr <= 1)),
        prefix + "rank@5": float(np.mean(arr <= 5)),
        prefix + "rank@10": float(np.mean(arr <= 10)),
        prefix + "rank@20": float(np.mean(arr <= 20)),
        prefix + "rank@50": float(np.mean(arr <= 50)),
        prefix + "mean_rank": float(np.mean(arr)),
        prefix + "median_rank": float(np.median(arr)),
    }


def _load_visible_ids(path: Path) -> set[int]:
    ids: set[int] = set()
    for row in _read_csv(path):
        rid = _as_int(row.get("raw_id"))
        if rid is not None and str(row.get("in_row_gap", "0")).strip() == "1":
            ids.add(int(rid))
    if len(ids) != 525:
        raise RuntimeError(f"expected 525 visible ids, got {len(ids)} from {path}")
    return ids


def _load_class_names(repo_root: Path, asset_root: Path, dataset_name: str) -> Dict[int, str]:
    _ensure_repo(repo_root)
    try:
        from videocutler.ext_stageb_ovvis.eval.g8_bridge import load_text_vocab_with_names  # type: ignore
        raw_ids, _records, _mat, class_map = load_text_vocab_with_names(asset_root, dataset_name)
        return {int(k): str(v) for k, v in dict(class_map).items()}
    except Exception:
        return {}


def _rows_and_carriers(gtceil: Any, *, asset_root: Path, dataset_name: str, ann: Path, max_rows: int) -> Tuple[List[Mapping[str, Any]], np.ndarray, Dict[str, Any]]:
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


def _group_vectors(rows: Sequence[Mapping[str, Any]], carrier: np.ndarray, visible: Optional[set[int]] = None) -> Dict[int, List[np.ndarray]]:
    out: Dict[int, List[np.ndarray]] = defaultdict(list)
    for i, row in enumerate(rows):
        rid = _as_int(row.get("raw_category_id"))
        if rid is None or i >= carrier.shape[0]:
            continue
        rid = int(rid)
        if visible is not None and rid not in visible:
            continue
        v = np.asarray(carrier[i], dtype=np.float32)
        if np.isfinite(v).all():
            out[rid].append(v)
    return dict(out)


def _prototypes(groups: Mapping[int, Sequence[np.ndarray]]) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for rid, vecs in groups.items():
        if not vecs:
            continue
        out[int(rid)] = _l2(np.mean(np.stack(vecs, axis=0), axis=0, keepdims=True))[0]
    return out


def _matrix_for_ids(proto: Mapping[int, np.ndarray], ids: Sequence[int], dim: int = 768) -> Tuple[np.ndarray, np.ndarray]:
    mat = np.full((len(ids), dim), np.nan, dtype=np.float32)
    valid = np.zeros((len(ids),), dtype=bool)
    for i, rid in enumerate(ids):
        v = proto.get(int(rid))
        if v is None:
            continue
        vv = np.asarray(v, dtype=np.float32).reshape(-1)
        mat[i, : min(dim, vv.shape[0])] = vv[:dim]
        valid[i] = np.isfinite(mat[i]).all()
    return mat, valid


def _graph_metrics(a_sim: np.ndarray, b_sim: np.ndarray, ids: Sequence[int], valid: np.ndarray, k: int, prefix: str = "") -> Dict[str, Any]:
    valid = np.asarray(valid, dtype=bool)
    jac = []
    person = 773
    person_in_a_topk = 0
    person_in_b_topk = 0
    deg_a: Counter[int] = Counter()
    deg_b: Counter[int] = Counter()
    for i in range(len(ids)):
        if not valid[i]:
            continue
        an = _topk(a_sim, ids, i, k)
        bn = _topk(b_sim, ids, i, k)
        jac.append(_jaccard(an, bn))
        if person in an:
            person_in_a_topk += 1
        if person in bn:
            person_in_b_topk += 1
        for nb in an:
            deg_a[int(nb)] += 1
        for nb in bn:
            deg_b[int(nb)] += 1
    return {
        prefix + "valid_class_count": int(valid.sum()),
        prefix + "spearman_pairwise": _spearman(_upper(a_sim, valid), _upper(b_sim, valid)),
        prefix + "mean_topk_jaccard": _mean(jac),
        prefix + "a_person_in_topk_rate": float(person_in_a_topk / max(int(valid.sum()), 1)),
        prefix + "b_person_in_topk_rate": float(person_in_b_topk / max(int(valid.sum()), 1)),
        prefix + "a_person_indegree@k": int(deg_a.get(person, 0)),
        prefix + "b_person_indegree@k": int(deg_b.get(person, 0)),
        prefix + "a_max_indegree@k": int(max(deg_a.values()) if deg_a else 0),
        prefix + "b_max_indegree@k": int(max(deg_b.values()) if deg_b else 0),
    }


def _nearest_from_sim(sim: np.ndarray, ids: Sequence[int], i: int) -> Tuple[Optional[int], float, float]:
    row = np.asarray(sim[i], dtype=np.float64).copy()
    if i < len(row):
        row[i] = -np.inf
    row[~np.isfinite(row)] = -np.inf
    if not np.isfinite(row).any():
        return None, float("nan"), float("nan")
    j = int(np.argmax(row))
    simv = float(row[j])
    dist = float(1.0 - simv)
    return int(ids[j]), dist, simv


def _intra_stats(groups: Mapping[int, Sequence[np.ndarray]], proto: Mapping[int, np.ndarray]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for rid, vecs in groups.items():
        mu = proto.get(int(rid))
        if mu is None or not vecs:
            continue
        arr = _l2(np.stack(vecs, axis=0))
        sims = arr @ np.asarray(mu, dtype=np.float32)
        d = 1.0 - np.asarray(sims, dtype=np.float64)
        out[int(rid)] = {
            "count": int(len(d)),
            "intra_radius_mean": float(np.mean(d)),
            "intra_radius_p50": float(np.percentile(d, 50)),
            "intra_radius_p90": float(np.percentile(d, 90)),
            "intra_radius_p95": float(np.percentile(d, 95)),
            "intra_radius_max": float(np.max(d)),
            "intra_diameter_p90_approx": float(2.0 * np.percentile(d, 90)),
        }
    return out


def _split_class_rows(
    *,
    split: str,
    ids: Sequence[int],
    proto: Mapping[int, np.ndarray],
    groups: Mapping[int, Sequence[np.ndarray]],
    names: Mapping[int, str],
    k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mat, valid = _matrix_for_ids(proto, ids)
    sim = _cos(np.nan_to_num(mat, nan=0.0))
    stats = _intra_stats(groups, proto)
    rows: List[Dict[str, Any]] = []
    for i, rid in enumerate(ids):
        if not valid[i] or int(rid) not in stats:
            continue
        nn, nearest_dist, nearest_sim = _nearest_from_sim(sim, ids, i)
        mean_inter = float(np.nanmean(1.0 - np.delete(sim[i], i))) if len(ids) > 1 else float("nan")
        person_dist = float("nan")
        person_sim = float("nan")
        if 773 in ids:
            pi = list(ids).index(773)
            person_sim = float(sim[i, pi]) if np.isfinite(sim[i, pi]) else float("nan")
            person_dist = float(1.0 - person_sim) if math.isfinite(person_sim) else float("nan")
        neigh = _topk(sim, ids, i, k)
        st = stats[int(rid)]
        nn_radius = stats.get(int(nn), {}).get("intra_radius_p90") if nn is not None else None
        radius = _safe_float(st.get("intra_radius_p90"))
        boundary_gap = float("nan")
        if nn_radius is not None and math.isfinite(radius):
            boundary_gap = float(nearest_dist - radius - float(nn_radius))
        sep_ratio = float(nearest_dist / radius) if math.isfinite(nearest_dist) and math.isfinite(radius) and radius > 1e-12 else float("nan")
        rows.append({
            "split": split,
            "raw_id": int(rid),
            "class_name": names.get(int(rid), f"raw_id_{rid}"),
            **st,
            "nearest_neighbor_raw_id": nn,
            "nearest_neighbor_name": names.get(int(nn), f"raw_id_{nn}") if nn is not None else "",
            "nearest_inter_distance": nearest_dist,
            "nearest_inter_similarity": nearest_sim,
            "mean_inter_distance": mean_inter,
            "separation_ratio_nn_dist_over_intra_p90": sep_ratio,
            "boundary_gap_nn_minus_two_p90": boundary_gap,
            "person_distance": person_dist,
            "person_similarity": person_sim,
            "topk_neighbors_raw_ids": " ".join(map(str, neigh)),
            "topk_neighbors_names": " | ".join(names.get(int(x), f"raw_id_{x}") for x in neigh),
        })
    summary = {
        "split": split,
        "class_count": int(len(rows)),
        **_summary_numeric(rows, "intra_radius_p90"),
        **_summary_numeric(rows, "intra_diameter_p90_approx"),
        **_summary_numeric(rows, "nearest_inter_distance"),
        **_summary_numeric(rows, "separation_ratio_nn_dist_over_intra_p90"),
        **_summary_numeric(rows, "boundary_gap_nn_minus_two_p90"),
        "boundary_gap_negative_rate": float(np.mean([_safe_float(r.get("boundary_gap_nn_minus_two_p90")) < 0 for r in rows])) if rows else None,
        "separation_ratio_lt_1_rate": float(np.mean([_safe_float(r.get("separation_ratio_nn_dist_over_intra_p90")) < 1 for r in rows])) if rows else None,
        "separation_ratio_lt_2_rate": float(np.mean([_safe_float(r.get("separation_ratio_nn_dist_over_intra_p90")) < 2 for r in rows])) if rows else None,
    }
    return rows, summary


def _train_val_shift_rows(
    *,
    ids: Sequence[int],
    train_proto: Mapping[int, np.ndarray],
    val_proto: Mapping[int, np.ndarray],
    train_groups: Mapping[int, Sequence[np.ndarray]],
    val_groups: Mapping[int, Sequence[np.ndarray]],
    names: Mapping[int, str],
    k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    train_mat, train_valid = _matrix_for_ids(train_proto, ids)
    val_mat, val_valid = _matrix_for_ids(val_proto, ids)
    both = train_valid & val_valid
    train_sim = _cos(np.nan_to_num(train_mat, nan=0.0))
    val_sim = _cos(np.nan_to_num(val_mat, nan=0.0))
    graph = _graph_metrics(train_sim, val_sim, ids, both, k)
    train_intra = _intra_stats(train_groups, train_proto)
    val_intra = _intra_stats(val_groups, val_proto)
    rows: List[Dict[str, Any]] = []
    for i, rid in enumerate(ids):
        if not both[i]:
            continue
        rid = int(rid)
        center_sim = float(np.dot(train_mat[i], val_mat[i]))
        center_shift = float(1.0 - center_sim)
        train_n = _topk(train_sim, ids, i, k)
        val_n = _topk(val_sim, ids, i, k)
        train_nn, train_nn_dist, _ = _nearest_from_sim(train_sim, ids, i)
        val_nn, val_nn_dist, _ = _nearest_from_sim(val_sim, ids, i)
        train_person_dist = float("nan")
        val_person_dist = float("nan")
        if 773 in ids:
            pi = list(ids).index(773)
            train_person_dist = float(1.0 - train_sim[i, pi])
            val_person_dist = float(1.0 - val_sim[i, pi])
        ti = train_intra.get(rid, {})
        vi = val_intra.get(rid, {})
        rows.append({
            "raw_id": rid,
            "class_name": names.get(rid, f"raw_id_{rid}"),
            "train_count": len(train_groups.get(rid, [])),
            "val_count": len(val_groups.get(rid, [])),
            "center_cosine_train_val": center_sim,
            "center_shift_distance_1_minus_cos": center_shift,
            "topk_neighbor_jaccard_train_val": _jaccard(train_n, val_n),
            "train_nearest_neighbor_raw_id": train_nn,
            "train_nearest_neighbor_name": names.get(int(train_nn), f"raw_id_{train_nn}") if train_nn is not None else "",
            "train_nearest_inter_distance": train_nn_dist,
            "val_nearest_neighbor_raw_id": val_nn,
            "val_nearest_neighbor_name": names.get(int(val_nn), f"raw_id_{val_nn}") if val_nn is not None else "",
            "val_nearest_inter_distance": val_nn_dist,
            "nearest_neighbor_changed": bool(train_nn != val_nn),
            "train_person_distance": train_person_dist,
            "val_person_distance": val_person_dist,
            "val_minus_train_person_distance": float(val_person_dist - train_person_dist) if math.isfinite(train_person_dist) and math.isfinite(val_person_dist) else float("nan"),
            "train_intra_radius_p90": ti.get("intra_radius_p90"),
            "val_intra_radius_p90": vi.get("intra_radius_p90"),
            "val_minus_train_intra_radius_p90": float(vi.get("intra_radius_p90", float("nan")) - ti.get("intra_radius_p90", float("nan"))) if ti and vi else float("nan"),
            "train_topk_neighbors_raw_ids": " ".join(map(str, train_n)),
            "val_topk_neighbors_raw_ids": " ".join(map(str, val_n)),
            "train_topk_neighbors_names": " | ".join(names.get(int(x), f"raw_id_{x}") for x in train_n),
            "val_topk_neighbors_names": " | ".join(names.get(int(x), f"raw_id_{x}") for x in val_n),
        })
    shift_summary = {
        "comparison": "train_visual_proto_vs_val_visual_proto",
        **graph,
        "visible_class_count": int(len(ids)),
        "train_proto_class_count": int(train_valid.sum()),
        "val_proto_class_count": int(val_valid.sum()),
        "intersection_class_count": int(both.sum()),
        **_summary_numeric(rows, "center_shift_distance_1_minus_cos"),
        **_summary_numeric(rows, "topk_neighbor_jaccard_train_val"),
        "nearest_neighbor_changed_rate": float(np.mean([bool(r.get("nearest_neighbor_changed")) for r in rows])) if rows else None,
        **_summary_numeric(rows, "val_minus_train_intra_radius_p90"),
        **_summary_numeric(rows, "val_minus_train_person_distance"),
    }
    return rows, shift_summary


def _row_boundary_audit(
    *,
    rows: Sequence[Mapping[str, Any]],
    carrier: np.ndarray,
    proto: Mapping[int, np.ndarray],
    candidate_ids: Sequence[int],
    names: Mapping[int, str],
    label: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates = [int(x) for x in candidate_ids if int(x) in proto]
    if not candidates:
        raise RuntimeError(f"no prototype candidates for {label}")
    pmat = _l2(np.stack([proto[int(x)] for x in candidates], axis=0))
    cand_to_idx = {int(r): i for i, r in enumerate(candidates)}
    ranks: List[int] = []
    margins: List[float] = []
    person_margins: List[float] = []
    gt_sims: List[float] = []
    topwrong_sims: List[float] = []
    person_top1 = 0
    person_closer = 0
    pair_counter: Counter[Tuple[int, int]] = Counter()
    class_buf: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "ranks": [], "margins": [], "person_margins": [], "top1_person": 0, "wrong_pairs": Counter()})
    for i, row in enumerate(rows):
        gt = _as_int(row.get("raw_category_id"))
        if gt is None or int(gt) not in cand_to_idx or i >= carrier.shape[0]:
            continue
        gt = int(gt)
        z = np.asarray(carrier[i], dtype=np.float32)
        sims = pmat @ z
        order = np.argsort(-sims, kind="mergesort")
        gt_idx = cand_to_idx[gt]
        where = np.where(order == gt_idx)[0]
        if where.size == 0:
            continue
        rank = int(where[0]) + 1
        ranks.append(rank)
        gt_sim = float(sims[gt_idx])
        wrong_order = [int(j) for j in order if int(j) != gt_idx]
        top_wrong_idx = wrong_order[0] if wrong_order else gt_idx
        top_wrong_raw = int(candidates[top_wrong_idx])
        top1_raw = int(candidates[int(order[0])])
        margin = float(gt_sim - float(sims[top_wrong_idx]))
        margins.append(margin)
        gt_sims.append(gt_sim)
        topwrong_sims.append(float(sims[top_wrong_idx]))
        pm = float("nan")
        if 773 in cand_to_idx:
            person_idx = cand_to_idx[773]
            pm = float(gt_sim - float(sims[person_idx]))
            person_margins.append(pm)
            if pm < 0:
                person_closer += 1
        if top1_raw == 773:
            person_top1 += 1
        if rank > 1:
            pair_counter[(gt, top_wrong_raw)] += 1
        b = class_buf[gt]
        b["count"] += 1
        b["ranks"].append(rank)
        b["margins"].append(margin)
        if math.isfinite(pm):
            b["person_margins"].append(pm)
        if top1_raw == 773:
            b["top1_person"] += 1
        if rank > 1:
            b["wrong_pairs"][(gt, top_wrong_raw)] += 1
    n = len(ranks)
    summary: Dict[str, Any] = {
        "boundary_label": label,
        "candidate_count": int(len(candidates)),
        "class_count": int(len(class_buf)),
        **_summary_ranks(ranks, prefix=""),
        "mean_margin_gt_vs_top_wrong": _mean(margins),
        "median_margin_gt_vs_top_wrong": _median(margins),
        "positive_margin_gt_vs_top_wrong_rate": float(np.mean(np.asarray(margins) > 0)) if margins else None,
        "negative_margin_gt_vs_top_wrong_rate": float(np.mean(np.asarray(margins) <= 0)) if margins else None,
        "mean_margin_gt_vs_person": _mean(person_margins),
        "positive_margin_gt_vs_person_rate": float(np.mean(np.asarray(person_margins) > 0)) if person_margins else None,
        "person_closer_than_gt_rate": float(person_closer / max(n, 1)),
        "top1_person_rate": float(person_top1 / max(n, 1)),
        "top1_max_wrong_pair": "",
        "top1_max_wrong_pair_count": 0,
    }
    if pair_counter:
        (gt, wrong), cnt = pair_counter.most_common(1)[0]
        summary["top1_max_wrong_pair"] = f"{gt}:{names.get(gt, gt)} -> {wrong}:{names.get(wrong, wrong)}"
        summary["top1_max_wrong_pair_count"] = int(cnt)
    per_class: List[Dict[str, Any]] = []
    for rid, b in sorted(class_buf.items()):
        rr = b["ranks"]
        mm = b["margins"]
        pm = b["person_margins"]
        wrong_pair = ""
        wrong_cnt = 0
        if b["wrong_pairs"]:
            (gt, wrong), wrong_cnt = b["wrong_pairs"].most_common(1)[0]
            wrong_pair = f"{gt}:{names.get(gt, gt)} -> {wrong}:{names.get(wrong, wrong)}"
        per_class.append({
            "boundary_label": label,
            "raw_id": int(rid),
            "class_name": names.get(int(rid), f"raw_id_{rid}"),
            "count": int(b["count"]),
            "rank@1": float(np.mean(np.asarray(rr) <= 1)) if rr else None,
            "mean_rank": float(np.mean(rr)) if rr else None,
            "median_rank": float(np.median(rr)) if rr else None,
            "mean_margin_gt_vs_top_wrong": _mean(mm),
            "positive_margin_gt_vs_top_wrong_rate": float(np.mean(np.asarray(mm) > 0)) if mm else None,
            "mean_margin_gt_vs_person": _mean(pm),
            "positive_margin_gt_vs_person_rate": float(np.mean(np.asarray(pm) > 0)) if pm else None,
            "top1_person_rate": float(b["top1_person"] / max(int(b["count"]), 1)),
            "top_wrong_pair": wrong_pair,
            "top_wrong_pair_count": int(wrong_cnt),
        })
    pair_rows: List[Dict[str, Any]] = []
    for (gt, wrong), cnt in pair_counter.most_common(200):
        pair_rows.append({
            "boundary_label": label,
            "gt_raw_id": int(gt),
            "gt_name": names.get(int(gt), f"raw_id_{gt}"),
            "wrong_raw_id": int(wrong),
            "wrong_name": names.get(int(wrong), f"raw_id_{wrong}"),
            "count": int(cnt),
        })
    return summary, per_class, pair_rows


def _merge_class_tables(
    *,
    shift_rows: Sequence[Mapping[str, Any]],
    train_class_rows: Sequence[Mapping[str, Any]],
    val_class_rows: Sequence[Mapping[str, Any]],
    boundary_per_class: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_id: Dict[int, Dict[str, Any]] = {}
    for r in shift_rows:
        rid = _as_int(r.get("raw_id"))
        if rid is None:
            continue
        by_id.setdefault(int(rid), {}).update({f"shift_{k}": v for k, v in r.items() if k not in {"raw_id", "class_name"}})
        by_id[int(rid)]["raw_id"] = int(rid)
        by_id[int(rid)]["class_name"] = r.get("class_name")
    for prefix, rows in [("train", train_class_rows), ("val", val_class_rows)]:
        for r in rows:
            rid = _as_int(r.get("raw_id"))
            if rid is None:
                continue
            by_id.setdefault(int(rid), {"raw_id": int(rid), "class_name": r.get("class_name")})
            for k, v in r.items():
                if k not in {"raw_id", "class_name", "split"}:
                    by_id[int(rid)][f"{prefix}_{k}"] = v
    for r in boundary_per_class:
        rid = _as_int(r.get("raw_id"))
        if rid is None:
            continue
        label = str(r.get("boundary_label", "boundary")).replace(" ", "_")
        by_id.setdefault(int(rid), {"raw_id": int(rid), "class_name": r.get("class_name")})
        for k, v in r.items():
            if k not in {"raw_id", "class_name", "boundary_label"}:
                by_id[int(rid)][f"{label}_{k}"] = v
    return [by_id[k] for k in sorted(by_id)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo_root", default=str(_repo_root_default()))
    p.add_argument("--asset_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--train_dataset_name", default="lvvis_train_base")
    p.add_argument("--val_dataset_name", default="lvvis_val")
    p.add_argument("--train_annotation_json", required=True)
    p.add_argument("--val_annotation_json", required=True)
    p.add_argument("--visible_csv", required=True)
    p.add_argument("--neighbor_k", type=int, default=10)
    p.add_argument("--max_rows", type=int, default=0, help="Optional smoke cap. 0 means full.")
    p.add_argument("--write_per_class_merged", action="store_true", default=True)
    args = p.parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _ensure_repo(repo_root)
    gtceil = _load_gtceil(repo_root)
    visible = _load_visible_ids(Path(args.visible_csv).expanduser().resolve())
    visible_ids = sorted(visible)
    names = _load_class_names(repo_root, asset_root, args.train_dataset_name)

    train_rows, train_carrier, train_meta = _rows_and_carriers(
        gtceil,
        asset_root=asset_root,
        dataset_name=args.train_dataset_name,
        ann=Path(args.train_annotation_json).expanduser().resolve(),
        max_rows=int(args.max_rows),
    )
    val_rows, val_carrier, val_meta = _rows_and_carriers(
        gtceil,
        asset_root=asset_root,
        dataset_name=args.val_dataset_name,
        ann=Path(args.val_annotation_json).expanduser().resolve(),
        max_rows=int(args.max_rows),
    )
    train_groups = _group_vectors(train_rows, train_carrier, visible)
    val_groups = _group_vectors(val_rows, val_carrier, visible)
    train_proto = _prototypes(train_groups)
    val_proto = _prototypes(val_groups)

    shift_rows, graph_summary = _train_val_shift_rows(
        ids=visible_ids,
        train_proto=train_proto,
        val_proto=val_proto,
        train_groups=train_groups,
        val_groups=val_groups,
        names=names,
        k=int(args.neighbor_k),
    )
    train_class_rows, train_class_summary = _split_class_rows(
        split="train",
        ids=visible_ids,
        proto=train_proto,
        groups=train_groups,
        names=names,
        k=int(args.neighbor_k),
    )
    val_class_rows, val_class_summary = _split_class_rows(
        split="val",
        ids=visible_ids,
        proto=val_proto,
        groups=val_groups,
        names=names,
        k=int(args.neighbor_k),
    )

    boundary_summaries: List[Dict[str, Any]] = []
    boundary_per_class_all: List[Dict[str, Any]] = []
    boundary_pair_all: List[Dict[str, Any]] = []
    boundary_jobs = [
        ("train_rows_vs_train_proto", train_rows, train_carrier, train_proto),
        ("val_rows_vs_train_proto", val_rows, val_carrier, train_proto),
        ("val_rows_vs_val_proto", val_rows, val_carrier, val_proto),
    ]
    for label, rows, carrier, proto in boundary_jobs:
        summary, per_class, pair_rows = _row_boundary_audit(
            rows=rows,
            carrier=carrier,
            proto=proto,
            candidate_ids=visible_ids,
            names=names,
            label=label,
        )
        boundary_summaries.append(summary)
        boundary_per_class_all.extend(per_class)
        boundary_pair_all.extend(pair_rows)

    merged = _merge_class_tables(
        shift_rows=shift_rows,
        train_class_rows=train_class_rows,
        val_class_rows=val_class_rows,
        boundary_per_class=boundary_per_class_all,
    )

    # Sort helpful diagnostic views.
    biggest_shift = sorted(shift_rows, key=lambda r: _safe_float(r.get("center_shift_distance_1_minus_cos")), reverse=True)[:50]
    lowest_neighbor_overlap = sorted(shift_rows, key=lambda r: _safe_float(r.get("topk_neighbor_jaccard_train_val"), 1.0))[:50]
    worst_boundary_gap = sorted(
        [r for r in val_class_rows if math.isfinite(_safe_float(r.get("boundary_gap_nn_minus_two_p90")))],
        key=lambda r: _safe_float(r.get("boundary_gap_nn_minus_two_p90")),
    )[:50]
    worst_val_margin = sorted(
        [r for r in boundary_per_class_all if r.get("boundary_label") == "val_rows_vs_train_proto"],
        key=lambda r: _safe_float(r.get("positive_margin_gt_vs_top_wrong_rate"), 1.0),
    )[:50]

    artifacts = {
        "graph_summary_csv": output_root / "train_val_structure_graph_summary.csv",
        "class_shift_per_class_csv": output_root / "train_val_class_shift_per_class.csv",
        "intra_inter_per_class_csv": output_root / "train_val_intra_inter_per_class.csv",
        "intra_inter_summary_csv": output_root / "train_val_intra_inter_summary.csv",
        "visual_row_boundary_summary_csv": output_root / "train_val_visual_row_boundary_summary.csv",
        "visual_row_boundary_per_class_csv": output_root / "train_val_visual_row_boundary_per_class.csv",
        "boundary_pair_top_failures_csv": output_root / "train_val_boundary_pair_top_failures.csv",
        "merged_per_class_csv": output_root / "train_val_structure_merged_per_class.csv",
        "biggest_shift_csv": output_root / "top50_center_shift_classes.csv",
        "lowest_neighbor_overlap_csv": output_root / "top50_neighbor_drift_classes.csv",
        "worst_boundary_gap_csv": output_root / "top50_worst_val_boundary_gap_classes.csv",
        "worst_val_margin_csv": output_root / "top50_worst_val_margin_classes.csv",
        "report_json": output_root / "train_val_structure_shift_report.json",
        "takeover_md": output_root / "train_val_structure_shift_takeover.md",
    }

    _write_csv(artifacts["graph_summary_csv"], [graph_summary])
    _write_csv(artifacts["class_shift_per_class_csv"], shift_rows)
    _write_csv(artifacts["intra_inter_per_class_csv"], train_class_rows + val_class_rows)
    _write_csv(artifacts["intra_inter_summary_csv"], [train_class_summary, val_class_summary])
    _write_csv(artifacts["visual_row_boundary_summary_csv"], boundary_summaries)
    _write_csv(artifacts["visual_row_boundary_per_class_csv"], boundary_per_class_all)
    _write_csv(artifacts["boundary_pair_top_failures_csv"], boundary_pair_all)
    _write_csv(artifacts["merged_per_class_csv"], merged)
    _write_csv(artifacts["biggest_shift_csv"], biggest_shift)
    _write_csv(artifacts["lowest_neighbor_overlap_csv"], lowest_neighbor_overlap)
    _write_csv(artifacts["worst_boundary_gap_csv"], worst_boundary_gap)
    _write_csv(artifacts["worst_val_margin_csv"], worst_val_margin)

    report = {
        "status": "PASS",
        "definition": "read-only audit of train/val visible525 visual structure shift, intra/inter separation, and visual row-level boundary separability",
        "output_root": str(output_root),
        "visible_class_count": int(len(visible_ids)),
        "train_dataset_name": args.train_dataset_name,
        "val_dataset_name": args.val_dataset_name,
        "neighbor_k": int(args.neighbor_k),
        "visual_source_meta": {"train": train_meta, "val": val_meta},
        "class_counts": {
            "train_visible_proto_count": int(len(train_proto)),
            "val_visible_proto_count": int(len(val_proto)),
            "train_val_intersection_count": int(len(set(train_proto) & set(val_proto) & visible)),
        },
        "train_val_graph_summary": graph_summary,
        "intra_inter_summary": {"train": train_class_summary, "val": val_class_summary},
        "visual_row_boundary_summary": boundary_summaries,
        "top_diagnostic_classes": {
            "biggest_center_shift": biggest_shift[:10],
            "lowest_neighbor_overlap": lowest_neighbor_overlap[:10],
            "worst_val_boundary_gap": worst_boundary_gap[:10],
            "worst_val_margin_vs_train_proto": worst_val_margin[:10],
        },
        "artifacts": {k: str(v) for k, v in artifacts.items()},
    }
    _write_json(artifacts["report_json"], report)

    # Compact takeover for control-plane review.
    lines = [
        "# A8 Train/Val Class Structure Shift Audit TAKEOVER",
        "",
        "## Status",
        "",
        "- overall_status: `PASS`",
        f"- output_root: `{output_root}`",
        f"- visible_class_count: `{len(visible_ids)}`",
        f"- train_visible_proto_count: `{len(train_proto)}`",
        f"- val_visible_proto_count: `{len(val_proto)}`",
        f"- train_val_intersection_count: `{len(set(train_proto) & set(val_proto) & visible)}`",
        "",
        "## Key aggregate diagnostics",
        "",
        f"- train_vs_val_graph_spearman: `{graph_summary.get('spearman_pairwise')}`",
        f"- train_vs_val_topK_neighbor_jaccard: `{graph_summary.get('mean_topk_jaccard')}`",
        f"- center_shift_mean: `{graph_summary.get('center_shift_distance_1_minus_cos_mean')}`",
        f"- center_shift_p90: `{graph_summary.get('center_shift_distance_1_minus_cos_p90')}`",
        f"- nearest_neighbor_changed_rate: `{graph_summary.get('nearest_neighbor_changed_rate')}`",
        "",
        "## Intra/inter separation",
        "",
        f"- train_boundary_gap_negative_rate: `{train_class_summary.get('boundary_gap_negative_rate')}`",
        f"- val_boundary_gap_negative_rate: `{val_class_summary.get('boundary_gap_negative_rate')}`",
        f"- train_separation_ratio_lt_1_rate: `{train_class_summary.get('separation_ratio_lt_1_rate')}`",
        f"- val_separation_ratio_lt_1_rate: `{val_class_summary.get('separation_ratio_lt_1_rate')}`",
        "",
        "## Visual row-level boundary summaries",
        "",
    ]
    for s in boundary_summaries:
        lines.append(f"- `{s.get('boundary_label')}`: rank@1=`{s.get('rank@1')}`, mean_rank=`{s.get('mean_rank')}`, negative_margin_rate=`{s.get('negative_margin_gt_vs_top_wrong_rate')}`, top1_person_rate=`{s.get('top1_person_rate')}`")
    lines += [
        "",
        "## Artifacts",
        "",
    ]
    for k, v in artifacts.items():
        if k.endswith("csv") or k.endswith("json"):
            lines.append(f"- {k}: `{v}`")
    lines += [
        "",
        "## Interpretation checklist",
        "",
        "- If train_vs_val_graph_spearman / topK Jaccard are low, train/val visual class geometry shifts substantially.",
        "- If val boundary_gap_negative_rate is high, intra-class spread overlaps nearest inter-class separation.",
        "- If val_rows_vs_train_proto rank@1 is far below train_rows_vs_train_proto, the drop is explained by visual distribution shift / instance spread before text projection.",
        "- If top1_person_rate remains high in visual-only prototype scoring, person hub is a carrier-geometry problem, not only a text-bank problem.",
    ]
    artifacts["takeover_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
