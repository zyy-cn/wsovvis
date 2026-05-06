#!/usr/bin/env python3
"""A8 text/vision prototype graph isomorphism audit.

Read-only audit for comparing the category graph induced by raw CLIP text
prototypes against the category graphs induced by DINOv2 GT-trajectory visual
prototypes. It also tries to build a projected-text graph from the D-J3
projector if the checkpoint bundle exposes a callable projector.

The script writes CSV/JSON/TAKEOVER artifacts only. It does not train, mutate
checkpoints, or write control-plane files.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

SENTINEL_RANK = 10**9


def _repo_root_from_arg(repo_root: Optional[str]) -> Path:
    return Path(repo_root).resolve() if repo_root else Path.cwd().resolve()


def _ensure_repo_on_path(repo_root: Path) -> None:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _load_gtceil_module(repo_root: Path):
    path = repo_root / "tools" / "run_a8_gt_trajectory_semantic_ceiling_eval.py"
    if not path.exists():
        raise FileNotFoundError(f"missing helper module: {path}")
    spec = importlib.util.spec_from_file_location("_a8_gtceil_helper", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fields:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _norm_id(x: Any) -> str:
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def _fnum(x: Any, default: float = float("nan")) -> float:
    try:
        if x in (None, "", "None", "nan", "NaN", "NA"):
            return default
        return float(x)
    except Exception:
        return default


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def _mean(vals: Iterable[float]) -> float:
    xs = [float(x) for x in vals if x == x and not math.isinf(float(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _median(vals: Iterable[float]) -> float:
    xs = [float(x) for x in vals if x == x and not math.isinf(float(x))]
    return float(np.median(xs)) if xs else float("nan")


def _std(vals: Iterable[float]) -> float:
    xs = [float(x) for x in vals if x == x and not math.isinf(float(x))]
    return float(np.std(xs)) if xs else float("nan")


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(map(str, a)), set(map(str, b))
    if not sa and not sb:
        return float("nan")
    return float(len(sa & sb) / max(len(sa | sb), 1))


def _rank_of(target: str, ordered: Sequence[str]) -> int:
    t = str(target)
    if not t:
        return SENTINEL_RANK
    for i, x in enumerate(ordered, start=1):
        if str(x) == t:
            return i
    return SENTINEL_RANK


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Small scipy-free rankdata(method='average')."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    if len(x) == 0:
        return ranks
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        avg_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return float("nan")
    ra = _rankdata_average(a[m])
    rb = _rankdata_average(b[m])
    sa = float(np.std(ra))
    sb = float(np.std(rb))
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _upper_tri_values(mat: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    n = int(mat.shape[0])
    iu = np.triu_indices(n, k=1)
    vals = np.asarray(mat[iu], dtype=np.float64)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        pair_valid = vm[iu[0]] & vm[iu[1]]
        vals = vals[pair_valid]
    return vals


def _cosine_matrix(mat: np.ndarray) -> np.ndarray:
    return _l2_normalize(np.asarray(mat, dtype=np.float32)) @ _l2_normalize(np.asarray(mat, dtype=np.float32)).T


def _valid_cosine_matrix(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(mat, dtype=np.float32)
    valid = np.isfinite(mat).all(axis=1)
    sim = np.full((mat.shape[0], mat.shape[0]), np.nan, dtype=np.float32)
    if valid.any():
        normed = _l2_normalize(mat[valid])
        sub = normed @ normed.T
        idx = np.where(valid)[0]
        sim[np.ix_(idx, idx)] = sub
    return sim, valid


def _topk_neighbors(sim: np.ndarray, ids: Sequence[str], idx: int, k: int) -> List[str]:
    row = np.asarray(sim[idx], dtype=np.float64).copy()
    if idx < len(row):
        row[idx] = -np.inf
    row[~np.isfinite(row)] = -np.inf
    if not np.isfinite(row).any():
        return []
    order = np.argsort(-row, kind="mergesort")[:k]
    return [str(ids[int(i)]) for i in order if np.isfinite(row[int(i)])]


def _neighbor_rank_map(sim: np.ndarray, ids: Sequence[str], idx: int) -> Dict[str, int]:
    row = np.asarray(sim[idx], dtype=np.float64).copy()
    if idx < len(row):
        row[idx] = -np.inf
    row[~np.isfinite(row)] = -np.inf
    order = np.argsort(-row, kind="mergesort")
    out: Dict[str, int] = {}
    rank = 1
    for j in order:
        if not np.isfinite(row[int(j)]):
            continue
        out[str(ids[int(j)])] = rank
        rank += 1
    return out


def _rank_distortion(source_neighbors: Sequence[str], target_rank: Mapping[str, int]) -> float:
    vals = [float(target_rank.get(str(x), SENTINEL_RANK)) for x in source_neighbors]
    vals = [v for v in vals if v < SENTINEL_RANK]
    return float(np.mean(vals)) if vals else float("nan")


def _in_degree(sim: np.ndarray, ids: Sequence[str], k: int) -> Counter[str]:
    deg: Counter[str] = Counter()
    for i in range(len(ids)):
        for nb in _topk_neighbors(sim, ids, i, k):
            deg[str(nb)] += 1
    return deg


def _knn_adjacency(sim: np.ndarray, k: int) -> np.ndarray:
    n = int(sim.shape[0])
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        row = np.asarray(sim[i], dtype=np.float64).copy()
        row[i] = -np.inf
        row[~np.isfinite(row)] = -np.inf
        if not np.isfinite(row).any():
            continue
        order = np.argsort(-row, kind="mergesort")[:k]
        for j in order:
            if np.isfinite(row[int(j)]):
                adj[i, int(j)] = 1.0
    adj = np.maximum(adj, adj.T)
    return adj


def _laplacian_spectrum(sim: np.ndarray, k: int, m: int) -> np.ndarray:
    adj = _knn_adjacency(sim, k)
    deg = adj.sum(axis=1)
    keep = deg > 0
    if int(keep.sum()) <= 2:
        return np.full((m,), np.nan, dtype=np.float64)
    a = adj[np.ix_(keep, keep)]
    d = a.sum(axis=1)
    inv = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-12)))
    lap = np.eye(a.shape[0]) - inv @ a @ inv
    vals = np.linalg.eigvalsh(lap)
    vals = np.sort(np.real(vals))[:m]
    if len(vals) < m:
        vals = np.pad(vals, (0, m - len(vals)), constant_values=np.nan)
    return vals.astype(np.float64)


def _spectral_distance(sim_a: np.ndarray, sim_b: np.ndarray, k: int, m: int) -> float:
    ea = _laplacian_spectrum(sim_a, k, m)
    eb = _laplacian_spectrum(sim_b, k, m)
    mask = np.isfinite(ea) & np.isfinite(eb)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.linalg.norm(ea[mask] - eb[mask]) / math.sqrt(int(mask.sum())))


def _class_name(raw_id: str, class_name_map: Mapping[Any, Any]) -> str:
    for key in (raw_id, int(raw_id) if str(raw_id).isdigit() else raw_id):
        if key in class_name_map:
            return str(class_name_map[key])
    return f"raw_id_{raw_id}"


def _load_rows_and_carriers(
    *,
    gtceil: Any,
    gt_carrier_path: Path,
    gt_identity_path: Path,
    gt_trajectory_path: Path,
    annotation_json: Path,
    max_rows: int = 0,
) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any], Dict[str, Any]]:
    rows0, source_meta = gtceil._candidate_source_rows(
        gt_carrier_path=gt_carrier_path,
        gt_identity_path=gt_identity_path,
        gt_trajectory_path=gt_trajectory_path,
        annotation_json=annotation_json,
        max_rows=int(max_rows or 0),
    )
    carrier_matrix, keep_indices, vector_counters = gtceil._load_carrier_matrix(
        gt_carrier_path=gt_carrier_path,
        rows=rows0,
    )
    rows = [rows0[int(i)] for i in keep_indices]
    return rows, np.asarray(carrier_matrix, dtype=np.float32), dict(source_meta), dict(vector_counters)


def _build_visual_prototypes_and_members(rows: Sequence[Mapping[str, Any]], carrier: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, List[int]]]:
    sums: Dict[str, np.ndarray] = {}
    members: Dict[str, List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        rid = _norm_id(row.get("raw_category_id", row.get("gt_raw_id", row.get("category_id", ""))))
        if not rid:
            continue
        vec = np.asarray(carrier[i], dtype=np.float32)
        if rid not in sums:
            sums[rid] = np.zeros_like(vec, dtype=np.float32)
        sums[rid] += vec
        members[rid].append(i)
    proto: Dict[str, np.ndarray] = {}
    for rid, idxs in members.items():
        if idxs:
            proto[rid] = _l2_normalize((sums[rid] / float(len(idxs)))[None, :])[0]
    return proto, dict(members)


def _matrix_for_ids(proto: Mapping[str, np.ndarray], ids: Sequence[str]) -> np.ndarray:
    dim = None
    for v in proto.values():
        dim = int(np.asarray(v).shape[-1])
        break
    if dim is None:
        raise RuntimeError("empty prototype dictionary")
    vecs: List[np.ndarray] = []
    for rid in ids:
        if rid in proto:
            vecs.append(np.asarray(proto[rid], dtype=np.float32))
        else:
            vecs.append(np.full((dim,), np.nan, dtype=np.float32))
    return np.stack(vecs, axis=0)


def _try_project_text(bundle: Any, text_sub: np.ndarray, device: torch.device) -> Tuple[Optional[np.ndarray], str]:
    """Best-effort text projection. Returns (projected, status).

    The canonical G8 inference path does not expose projected text by calling the
    ProjectorBundle itself.  It stores the actual nn.Module at bundle.projector
    and projects candidate text through _project_candidate_matrix(...).  The
    first branch below mirrors that canonical scoring path exactly.  The
    remaining branches are fallback probes for older or future checkpoint bundle
    shapes.
    """
    text_np = np.asarray(text_sub, dtype=np.float32)

    projector = getattr(bundle, "projector", None)
    if projector is None and isinstance(bundle, Mapping):
        projector = bundle.get("projector") or bundle.get("text_projector")

    if projector is not None:
        try:
            from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import _project_candidate_matrix  # type: ignore

            with torch.no_grad():
                if hasattr(projector, "eval"):
                    projector.eval()
                y = _project_candidate_matrix(projector=projector, candidate_matrix=text_np, device=device)
                if torch.is_tensor(y):
                    arr = y.detach().float().cpu().numpy()
                    if arr.ndim == 2 and arr.shape[0] == text_np.shape[0]:
                        return _l2_normalize(arr), "available:g8_bridge_canonical_project_candidate_matrix"
        except Exception as exc:
            canonical_error = f"canonical_project_candidate_matrix_failed:{type(exc).__name__}:{exc}"
        else:
            canonical_error = "canonical_project_candidate_matrix_failed:unknown"
    else:
        canonical_error = "missing_bundle_projector"

    candidates: List[Tuple[str, Any]] = []
    if callable(bundle):
        candidates.append(("bundle_callable", bundle))
    if hasattr(bundle, "project_text"):
        candidates.append(("bundle.project_text", getattr(bundle, "project_text")))
    if hasattr(bundle, "text_projector"):
        candidates.append(("bundle.text_projector", getattr(bundle, "text_projector")))
    if hasattr(bundle, "projector") and callable(getattr(bundle, "projector")):
        candidates.append(("bundle.projector", getattr(bundle, "projector")))
    if isinstance(bundle, Mapping):
        for key in ["projector", "text_projector", "model", "module", "net"]:
            obj = bundle.get(key)
            if obj is None:
                continue
            if hasattr(obj, "project_text"):
                candidates.append((f"{key}.project_text", getattr(obj, "project_text")))
            if hasattr(obj, "text_projector"):
                candidates.append((f"{key}.text_projector", getattr(obj, "text_projector")))
            if callable(obj):
                candidates.append((key, obj))
    x = torch.as_tensor(text_np, device=device)
    for name, fn in candidates:
        try:
            with torch.no_grad():
                if hasattr(fn, "eval"):
                    fn.eval()
                y = fn(x)
                if isinstance(y, (tuple, list)):
                    y = y[0]
                if isinstance(y, Mapping):
                    for kk in ["projected_text", "text", "emb", "embedding", "features", "out"]:
                        if kk in y:
                            y = y[kk]
                            break
                if not torch.is_tensor(y):
                    continue
                arr = y.detach().float().cpu().numpy()
                if arr.ndim == 2 and arr.shape[0] == text_np.shape[0]:
                    return _l2_normalize(arr), f"available:fallback:{name}"
        except Exception:
            continue
    return None, f"unavailable:no_projected_text:{canonical_error}"


def _resolve_inputs(args: argparse.Namespace, run_root: Path) -> Dict[str, Path]:
    return {
        "per_class_join": Path(args.per_class_join) if args.per_class_join else run_root / "analysis/a8_dj3_semantic_boundary_bottleneck_audit/per_class_train_val_525_join.csv",
        "val_visible_per_row": Path(args.val_visible_per_row) if args.val_visible_per_row else run_root / "analysis/a8_visible525_candidate_rankk_audit/lvvis_val/D-J3_train_time_dynamic_ep10_val_target525_candidate525/visible525_candidate_rankk_per_row.csv",
        "checkpoint": Path(args.checkpoint_path) if args.checkpoint_path else run_root / "outputs/a8_joint_train_time_dynamic_hungarian/lvvis_train_base/D-J3_pre1_dyn1_ep10/train/joint_train_time_dynamic_hungarian/a8_joint_train_time_dynamic_last.pth",
    }


def _summary_by_group(rows: Sequence[Mapping[str, Any]], group_key: str, numeric_keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(group_key, "NA"))].append(r)
    out: List[Dict[str, Any]] = []
    for g, items in sorted(groups.items()):
        rec: Dict[str, Any] = {group_key: g, "class_count": len(items)}
        for k in numeric_keys:
            vals = [_fnum(x.get(k)) for x in items]
            rec[f"mean_{k}"] = _mean(vals)
            rec[f"median_{k}"] = _median(vals)
            rec[f"std_{k}"] = _std(vals)
        rec["person_suppressed_class_count"] = sum(str(x.get("is_person_suppressed")) == "True" for x in items)
        rec["has_train_visual_proto_count"] = sum(str(x.get("has_train_visual_proto")) == "True" for x in items)
        rec["has_val_visual_proto_count"] = sum(str(x.get("has_val_visual_proto")) == "True" for x in items)
        out.append(rec)
    return out


def _sim_dict(sim_mats: Mapping[str, np.ndarray], a: str, b: str, valid: Optional[np.ndarray] = None) -> Dict[str, Any]:
    if a not in sim_mats or b not in sim_mats:
        return {"graph_a": a, "graph_b": b, "spearman_pairwise": float("nan"), "note": "missing_graph"}
    va = _upper_tri_values(sim_mats[a], valid)
    vb = _upper_tri_values(sim_mats[b], valid)
    return {"graph_a": a, "graph_b": b, "spearman_pairwise": _spearman_corr(va, vb), "pair_count": int(np.isfinite(va + vb).sum())}


def _permutation_control(
    sim_text: np.ndarray,
    sim_vis: np.ndarray,
    ids: Sequence[str],
    k: int,
    rounds: int,
    seed: int,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    real_corr = _spearman_corr(_upper_tri_values(sim_text, valid_mask), _upper_tri_values(sim_vis, valid_mask))
    real_j = []
    for i in range(len(ids)):
        if valid_mask is not None and not bool(valid_mask[i]):
            continue
        real_j.append(_jaccard(_topk_neighbors(sim_text, ids, i, k), _topk_neighbors(sim_vis, ids, i, k)))
    rand_corrs = []
    rand_js = []
    n = len(ids)
    for _ in range(max(0, int(rounds))):
        perm = rng.permutation(n)
        sim_perm = sim_vis[np.ix_(perm, perm)]
        vm = valid_mask[perm] if valid_mask is not None else None
        rand_corrs.append(_spearman_corr(_upper_tri_values(sim_text, valid_mask), _upper_tri_values(sim_perm, vm)))
        js = []
        perm_ids = [ids[int(j)] for j in perm]
        for i in range(n):
            if valid_mask is not None and not bool(valid_mask[i]):
                continue
            js.append(_jaccard(_topk_neighbors(sim_text, ids, i, k), _topk_neighbors(sim_perm, perm_ids, i, k)))
        rand_js.append(_mean(js))
    return {
        "real_spearman": real_corr,
        "random_spearman_mean": _mean(rand_corrs),
        "random_spearman_std": _std(rand_corrs),
        "real_mean_jaccard": _mean(real_j),
        "random_mean_jaccard_mean": _mean(rand_js),
        "random_mean_jaccard_std": _std(rand_js),
        "rounds": int(rounds),
    }


def _bootstrap_stability(
    ids: Sequence[str],
    members: Mapping[str, Sequence[int]],
    carrier: np.ndarray,
    full_sim: np.ndarray,
    k: int,
    rounds: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    id_to_pos = {rid: i for i, rid in enumerate(ids)}
    for rid in ids:
        idxs = list(members.get(rid, []))
        pos = id_to_pos[rid]
        full_neighbors = _topk_neighbors(full_sim, ids, pos, k)
        cos_vals = []
        jac_vals = []
        if not idxs or not np.isfinite(full_sim[pos]).any():
            rows.append({"raw_id": rid, "sample_count": len(idxs), "bootstrap_rounds": 0, "mean_proto_cos_to_full": float("nan"), "mean_neighbor_jaccard": float("nan")})
            continue
        full_proto = _l2_normalize(np.mean(carrier[idxs], axis=0, keepdims=True))[0]
        for _ in range(max(1, int(rounds))):
            sample = rng.choice(idxs, size=len(idxs), replace=True)
            proto = _l2_normalize(np.mean(carrier[sample], axis=0, keepdims=True))[0]
            cos_vals.append(float(np.dot(proto, full_proto)))
            # For neighbor stability, replace only this class prototype is insufficient to rebuild a true graph.
            # Instead use cosine-to-all against full class prototypes implicitly from full_sim row scale is unavailable.
            # We report prototype cosine as the primary stability statistic and leave neighbor Jaccard as NA.
        rows.append({
            "raw_id": rid,
            "sample_count": len(idxs),
            "bootstrap_rounds": int(rounds),
            "mean_proto_cos_to_full": _mean(cos_vals),
            "std_proto_cos_to_full": _std(cos_vals),
            "mean_neighbor_jaccard": float("nan"),
            "note": "n=1 bootstrap is deterministic and should not be over-interpreted" if len(idxs) <= 1 else "prototype bootstrap cosine only",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="A8 text/vision prototype graph isomorphism audit")
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    ap.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--checkpoint_path", default="")
    ap.add_argument("--per_class_join", default="")
    ap.add_argument("--val_visible_per_row", default="")
    ap.add_argument("--train_dataset_name", default="lvvis_train_base")
    ap.add_argument("--val_dataset_name", default="lvvis_val")
    ap.add_argument("--train_annotation_json", default="/mnt/sda/zyy/code/wsovvis/videocutler/datasets/LV-VIS/annotations/train_instances.json")
    ap.add_argument("--val_annotation_json", default="/mnt/sda/zyy/code/wsovvis/videocutler/datasets/LV-VIS/annotations/val_instances.json")
    ap.add_argument("--neighbor_k", type=int, default=10)
    ap.add_argument("--spectral_m", type=int, default=32)
    ap.add_argument("--bootstrap_rounds", type=int, default=20)
    ap.add_argument("--random_perm_rounds", type=int, default=50)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--person_raw_id", default="773")
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    repo_root = _repo_root_from_arg(args.repo_root)
    asset_root = Path(args.asset_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    _ensure_repo_on_path(repo_root)
    gtceil = _load_gtceil_module(repo_root)

    inputs = _resolve_inputs(args, run_root)
    for name, path in inputs.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"missing input {name}: {path}")
    per_class = _read_csv(inputs["per_class_join"])
    val_per_rows = _read_csv(inputs["val_visible_per_row"])

    from videocutler.ext_stageb_ovvis.eval.g8_bridge import load_projector_bundle, load_text_vocab_with_names  # type: ignore

    text_vocab_ids_raw, _text_records, text_matrix, class_name_map = load_text_vocab_with_names(asset_root, args.train_dataset_name)
    text_ids_all = [_norm_id(x) for x in text_vocab_ids_raw]
    text_id_to_idx = {rid: i for i, rid in enumerate(text_ids_all)}
    text_matrix_np = _l2_normalize(np.asarray(text_matrix, dtype=np.float32))

    # Keep exactly the 525 audit classes that have text prototypes.
    class_ids = []
    per_class_by_id: Dict[str, Dict[str, str]] = {}
    for row in per_class:
        rid = _norm_id(row.get("raw_id"))
        if rid in text_id_to_idx and rid not in per_class_by_id:
            class_ids.append(rid)
            per_class_by_id[rid] = row
    if not class_ids:
        raise RuntimeError("no class ids in per_class_join matched text vocab")
    id_to_pos = {rid: i for i, rid in enumerate(class_ids)}
    class_idx = np.asarray([text_id_to_idx[rid] for rid in class_ids], dtype=np.int64)
    text_sub = text_matrix_np[class_idx]
    text_sim = _cosine_matrix(text_sub)

    # Try D-J3 projected-text graph.
    device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    projection_status = "not_attempted"
    proj_sim = None
    try:
        bundle = load_projector_bundle(inputs["checkpoint"], device=device)
        proj_text, projection_status = _try_project_text(bundle, text_sub, device)
        if proj_text is not None:
            proj_sim = _cosine_matrix(proj_text)
    except Exception as e:
        projection_status = f"unavailable:{type(e).__name__}:{e}"

    # Load visual GT carrier rows and prototypes.
    train_rows, train_carrier, train_meta, train_vec_counters = _load_rows_and_carriers(
        gtceil=gtceil,
        gt_carrier_path=asset_root / "carrier_bank_gt" / args.train_dataset_name / "carrier_records.jsonl",
        gt_identity_path=asset_root / "carrier_bank_gt" / args.train_dataset_name / "gt_carrier_identity_binding.jsonl",
        gt_trajectory_path=asset_root / "exports_gt" / args.train_dataset_name / "trajectory_records.jsonl",
        annotation_json=Path(args.train_annotation_json),
        max_rows=args.max_rows,
    )
    val_rows, val_carrier, val_meta, val_vec_counters = _load_rows_and_carriers(
        gtceil=gtceil,
        gt_carrier_path=asset_root / "carrier_bank_gt" / args.val_dataset_name / "carrier_records.jsonl",
        gt_identity_path=asset_root / "carrier_bank_gt" / args.val_dataset_name / "gt_carrier_identity_binding.jsonl",
        gt_trajectory_path=asset_root / "exports_gt" / args.val_dataset_name / "trajectory_records.jsonl",
        annotation_json=Path(args.val_annotation_json),
        max_rows=args.max_rows,
    )
    train_proto, train_members = _build_visual_prototypes_and_members(train_rows, train_carrier)
    val_proto, val_members = _build_visual_prototypes_and_members(val_rows, val_carrier)
    train_proto_mat = _matrix_for_ids(train_proto, class_ids)
    val_proto_mat = _matrix_for_ids(val_proto, class_ids)
    train_sim, train_valid = _valid_cosine_matrix(train_proto_mat)
    val_sim, val_valid = _valid_cosine_matrix(val_proto_mat)
    train_val_valid = train_valid & val_valid

    sim_mats: Dict[str, np.ndarray] = {
        "raw_text": text_sim,
        "vision_train": train_sim,
        "vision_val": val_sim,
    }
    if proj_sim is not None:
        sim_mats["projected_text"] = proj_sim

    k = int(args.neighbor_k)
    spectral_m = int(args.spectral_m)

    # Val suppressor by GT class.
    suppressor_by_gt: Dict[str, Counter[str]] = defaultdict(Counter)
    suppressor_margin: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in val_per_rows:
        gt = _norm_id(row.get("gt_raw_id"))
        top1 = _norm_id(row.get("top1_raw_id"))
        rank = _fnum(row.get("restricted_rank"))
        if rank > 1 and gt and top1:
            suppressor_by_gt[gt][top1] += 1
            m = _fnum(row.get("margin_gt_minus_top1"))
            if m == m:
                suppressor_margin[(gt, top1)].append(m)

    # Per-class local graph metrics.
    metric_rows: List[Dict[str, Any]] = []
    graph_names = ["raw_text", "vision_train", "vision_val"] + (["projected_text"] if proj_sim is not None else [])
    neighbor_cache: Dict[Tuple[str, str], List[str]] = {}
    rank_cache: Dict[Tuple[str, str], Dict[str, int]] = {}
    for g in graph_names:
        sim = sim_mats[g]
        for rid, pos in id_to_pos.items():
            neighbor_cache[(g, rid)] = _topk_neighbors(sim, class_ids, pos, k)
            rank_cache[(g, rid)] = _neighbor_rank_map(sim, class_ids, pos)

    for rid in class_ids:
        pos = id_to_pos[rid]
        cls = per_class_by_id[rid]
        text_n = neighbor_cache[("raw_text", rid)]
        vtrain_n = neighbor_cache[("vision_train", rid)]
        vval_n = neighbor_cache[("vision_val", rid)]
        proj_n = neighbor_cache.get(("projected_text", rid), [])

        drift = float("nan")
        if rid in train_proto and rid in val_proto:
            drift = float(1.0 - float(np.dot(train_proto[rid], val_proto[rid])))

        top_supp = ""
        top_supp_count = 0
        if rid in suppressor_by_gt and suppressor_by_gt[rid]:
            top_supp, top_supp_count = suppressor_by_gt[rid].most_common(1)[0]
        margins = suppressor_margin.get((rid, top_supp), []) if top_supp else []

        rec: Dict[str, Any] = {
            "raw_id": rid,
            "class_name": cls.get("class_name", _class_name(rid, class_name_map)),
            "quadrant": cls.get("quadrant", ""),
            "group": cls.get("quadrant", ""),
            "support_bucket": cls.get("support_bucket", ""),
            "train_count": cls.get("train_count", ""),
            "val_count": cls.get("val_count", ""),
            "train_rank@1": cls.get("train_rank@1", ""),
            "val_rank@1": cls.get("val_rank@1", ""),
            "has_train_visual_proto": bool(train_valid[pos]),
            "has_val_visual_proto": bool(val_valid[pos]),
            "train_val_visual_proto_drift": drift,
            "text_vtrain_jaccard@k": _jaccard(text_n, vtrain_n),
            "text_vval_jaccard@k": _jaccard(text_n, vval_n),
            "vtrain_vval_jaccard@k": _jaccard(vtrain_n, vval_n),
            "text_topk_rank_in_vtrain_mean": _rank_distortion(text_n, rank_cache[("vision_train", rid)]),
            "text_topk_rank_in_vval_mean": _rank_distortion(text_n, rank_cache[("vision_val", rid)]),
            "vtrain_topk_rank_in_vval_mean": _rank_distortion(vtrain_n, rank_cache[("vision_val", rid)]),
            "text_neighbors@k": ";".join(text_n),
            "vision_train_neighbors@k": ";".join(vtrain_n),
            "vision_val_neighbors@k": ";".join(vval_n),
            "top_suppressor_raw_id": top_supp,
            "top_suppressor_name": _class_name(top_supp, class_name_map) if top_supp else "",
            "top_suppressor_count": top_supp_count,
            "is_person_suppressed": top_supp == _norm_id(args.person_raw_id),
            "mean_gt_vs_top_suppressor_margin": _mean(margins),
            "suppressor_text_rank": rank_cache[("raw_text", rid)].get(top_supp, SENTINEL_RANK) if top_supp else "",
            "suppressor_vtrain_rank": rank_cache[("vision_train", rid)].get(top_supp, SENTINEL_RANK) if top_supp else "",
            "suppressor_vval_rank": rank_cache[("vision_val", rid)].get(top_supp, SENTINEL_RANK) if top_supp else "",
        }
        if proj_sim is not None:
            rec.update({
                "projected_text_neighbors@k": ";".join(proj_n),
                "proj_vtrain_jaccard@k": _jaccard(proj_n, vtrain_n),
                "proj_vval_jaccard@k": _jaccard(proj_n, vval_n),
                "text_proj_jaccard@k": _jaccard(text_n, proj_n),
                "proj_topk_rank_in_vtrain_mean": _rank_distortion(proj_n, rank_cache[("vision_train", rid)]),
                "proj_topk_rank_in_vval_mean": _rank_distortion(proj_n, rank_cache[("vision_val", rid)]),
                "suppressor_proj_rank": rank_cache[("projected_text", rid)].get(top_supp, SENTINEL_RANK) if top_supp else "",
            })
        metric_rows.append(rec)

    numeric_keys = [
        "train_val_visual_proto_drift",
        "text_vtrain_jaccard@k",
        "text_vval_jaccard@k",
        "vtrain_vval_jaccard@k",
        "text_topk_rank_in_vtrain_mean",
        "text_topk_rank_in_vval_mean",
        "vtrain_topk_rank_in_vval_mean",
        "suppressor_text_rank",
        "suppressor_vtrain_rank",
        "suppressor_vval_rank",
        "mean_gt_vs_top_suppressor_margin",
    ]
    if proj_sim is not None:
        numeric_keys.extend([
            "proj_vtrain_jaccard@k",
            "proj_vval_jaccard@k",
            "text_proj_jaccard@k",
            "proj_topk_rank_in_vtrain_mean",
            "proj_topk_rank_in_vval_mean",
            "suppressor_proj_rank",
        ])

    group_summary = _summary_by_group(metric_rows, "group", numeric_keys)
    support_summary = _summary_by_group(metric_rows, "support_bucket", numeric_keys)

    # Global graph comparisons.
    global_rows: List[Dict[str, Any]] = []
    comparisons = [("raw_text", "vision_train", train_valid), ("raw_text", "vision_val", val_valid), ("vision_train", "vision_val", train_val_valid)]
    if proj_sim is not None:
        comparisons.extend([("projected_text", "vision_train", train_valid), ("projected_text", "vision_val", val_valid), ("raw_text", "projected_text", None)])
    for a, b, vm in comparisons:
        rec = _sim_dict(sim_mats, a, b, vm)
        rec["spectral_distance@k"] = _spectral_distance(sim_mats[a], sim_mats[b], k, spectral_m) if a in sim_mats and b in sim_mats else float("nan")
        global_rows.append(rec)

    # Hub structure.
    hub_rows: List[Dict[str, Any]] = []
    degs = {g: _in_degree(sim_mats[g], class_ids, k) for g in graph_names}
    for rid in class_ids:
        rec: Dict[str, Any] = {"raw_id": rid, "class_name": _class_name(rid, class_name_map), "is_person": rid == _norm_id(args.person_raw_id)}
        for g in graph_names:
            rec[f"{g}_indegree@k"] = degs[g].get(rid, 0)
        hub_rows.append(rec)
    hub_summary: List[Dict[str, Any]] = []
    for g in graph_names:
        top = degs[g].most_common(20)
        for rank, (rid, deg) in enumerate(top, start=1):
            hub_summary.append({"graph": g, "hub_rank": rank, "raw_id": rid, "class_name": _class_name(rid, class_name_map), "indegree@k": deg, "is_person": rid == _norm_id(args.person_raw_id)})

    # Suppressor edge graph diagnosis.
    suppressor_rows: List[Dict[str, Any]] = []
    for r in metric_rows:
        if not r.get("top_suppressor_raw_id"):
            continue
        suppressor_rows.append({
            "raw_id": r.get("raw_id"),
            "class_name": r.get("class_name"),
            "group": r.get("group"),
            "support_bucket": r.get("support_bucket"),
            "top_suppressor_raw_id": r.get("top_suppressor_raw_id"),
            "top_suppressor_name": r.get("top_suppressor_name"),
            "top_suppressor_count": r.get("top_suppressor_count"),
            "is_person_suppressed": r.get("is_person_suppressed"),
            "suppressor_text_rank": r.get("suppressor_text_rank"),
            "suppressor_vtrain_rank": r.get("suppressor_vtrain_rank"),
            "suppressor_vval_rank": r.get("suppressor_vval_rank"),
            "suppressor_proj_rank": r.get("suppressor_proj_rank", "NA"),
            "mean_gt_vs_top_suppressor_margin": r.get("mean_gt_vs_top_suppressor_margin"),
        })

    # Permutation controls.
    perm_rows: List[Dict[str, Any]] = []
    for name, sim, vm in [("text_vs_vtrain", train_sim, train_valid), ("text_vs_vval", val_sim, val_valid), ("vtrain_vs_vval", val_sim, train_val_valid)]:
        if name == "vtrain_vs_vval":
            res = _permutation_control(train_sim, val_sim, class_ids, k, args.random_perm_rounds, args.seed, train_val_valid)
            graph_a, graph_b = "vision_train", "vision_val"
        else:
            res = _permutation_control(text_sim, sim, class_ids, k, args.random_perm_rounds, args.seed, vm)
            graph_a, graph_b = "raw_text", "vision_train" if name.endswith("vtrain") else "vision_val"
        res.update({"comparison": name, "graph_a": graph_a, "graph_b": graph_b})
        perm_rows.append(res)

    # Bootstrap prototype stability.
    boot_train = _bootstrap_stability(class_ids, train_members, train_carrier, train_sim, k, args.bootstrap_rounds, args.seed)
    boot_val = _bootstrap_stability(class_ids, val_members, val_carrier, val_sim, k, args.bootstrap_rounds, args.seed + 13)
    for r in boot_train:
        r["split"] = "train"
    for r in boot_val:
        r["split"] = "val"
    bootstrap_rows = boot_train + boot_val

    _write_csv(out_root / "prototype_inventory.csv", [{
        "raw_id": rid,
        "class_name": _class_name(rid, class_name_map),
        "train_count_from_per_class": per_class_by_id[rid].get("train_count", ""),
        "val_count_from_per_class": per_class_by_id[rid].get("val_count", ""),
        "support_bucket": per_class_by_id[rid].get("support_bucket", ""),
        "quadrant": per_class_by_id[rid].get("quadrant", ""),
        "has_raw_text": True,
        "has_projected_text": proj_sim is not None,
        "has_train_visual_proto": bool(train_valid[id_to_pos[rid]]),
        "has_val_visual_proto": bool(val_valid[id_to_pos[rid]]),
    } for rid in class_ids])
    _write_csv(out_root / "graph_global_isomorphism_summary.csv", global_rows)
    _write_csv(out_root / "class_graph_isomorphism_metrics.csv", metric_rows)
    _write_csv(out_root / "group_graph_isomorphism_summary.csv", group_summary)
    _write_csv(out_root / "support_bucket_graph_summary.csv", support_summary)
    _write_csv(out_root / "hub_structure_by_class.csv", hub_rows)
    _write_csv(out_root / "hub_structure_summary.csv", hub_summary)
    _write_csv(out_root / "suppressor_edge_graph_diagnosis.csv", suppressor_rows)
    _write_csv(out_root / "random_permutation_control.csv", perm_rows)
    _write_csv(out_root / "bootstrap_prototype_stability.csv", bootstrap_rows)

    def _find_group(name: str) -> Dict[str, Any]:
        for r in group_summary:
            if r.get("group") == name:
                return dict(r)
        return {"group": name, "class_count": 0}

    learned = _find_group("learned_stable")
    overfit = _find_group("overfit_context_fail")
    underlearned = _find_group("underlearned")
    headline = {
        "class_count": len(class_ids),
        "projection_status": projection_status,
        "raw_text_vs_vtrain_spearman": next((r.get("spearman_pairwise") for r in global_rows if r.get("graph_a") == "raw_text" and r.get("graph_b") == "vision_train"), float("nan")),
        "raw_text_vs_vval_spearman": next((r.get("spearman_pairwise") for r in global_rows if r.get("graph_a") == "raw_text" and r.get("graph_b") == "vision_val"), float("nan")),
        "vtrain_vs_vval_spearman": next((r.get("spearman_pairwise") for r in global_rows if r.get("graph_a") == "vision_train" and r.get("graph_b") == "vision_val"), float("nan")),
        "learned_stable": learned,
        "overfit_context_fail": overfit,
        "underlearned": underlearned,
        "overfit_minus_learned_text_vtrain_jaccard": _fnum(overfit.get("mean_text_vtrain_jaccard@k")) - _fnum(learned.get("mean_text_vtrain_jaccard@k")),
        "overfit_minus_learned_text_vval_jaccard": _fnum(overfit.get("mean_text_vval_jaccard@k")) - _fnum(learned.get("mean_text_vval_jaccard@k")),
        "overfit_minus_learned_vtrain_vval_jaccard": _fnum(overfit.get("mean_vtrain_vval_jaccard@k")) - _fnum(learned.get("mean_vtrain_vval_jaccard@k")),
        "overfit_minus_learned_train_val_visual_proto_drift": _fnum(overfit.get("mean_train_val_visual_proto_drift")) - _fnum(learned.get("mean_train_val_visual_proto_drift")),
    }
    if proj_sim is not None:
        headline.update({
            "projected_text_vs_vtrain_spearman": next((r.get("spearman_pairwise") for r in global_rows if r.get("graph_a") == "projected_text" and r.get("graph_b") == "vision_train"), float("nan")),
            "projected_text_vs_vval_spearman": next((r.get("spearman_pairwise") for r in global_rows if r.get("graph_a") == "projected_text" and r.get("graph_b") == "vision_val"), float("nan")),
            "raw_text_vs_projected_text_spearman": next((r.get("spearman_pairwise") for r in global_rows if r.get("graph_a") == "raw_text" and r.get("graph_b") == "projected_text"), float("nan")),
        })

    payload = {
        "status": "PASS",
        "output_root": str(out_root),
        "inputs": {k0: str(v0) for k0, v0 in inputs.items()},
        "asset_root": str(asset_root),
        "neighbor_k": k,
        "spectral_m": spectral_m,
        "projection_status": projection_status,
        "source_meta": {
            "train": train_meta,
            "val": val_meta,
            "train_vector_counters": train_vec_counters,
            "val_vector_counters": val_vec_counters,
        },
        "headline": headline,
        "artifacts": {
            "prototype_inventory": str(out_root / "prototype_inventory.csv"),
            "graph_global_isomorphism_summary": str(out_root / "graph_global_isomorphism_summary.csv"),
            "class_graph_isomorphism_metrics": str(out_root / "class_graph_isomorphism_metrics.csv"),
            "group_graph_isomorphism_summary": str(out_root / "group_graph_isomorphism_summary.csv"),
            "support_bucket_graph_summary": str(out_root / "support_bucket_graph_summary.csv"),
            "hub_structure_summary": str(out_root / "hub_structure_summary.csv"),
            "hub_structure_by_class": str(out_root / "hub_structure_by_class.csv"),
            "suppressor_edge_graph_diagnosis": str(out_root / "suppressor_edge_graph_diagnosis.csv"),
            "random_permutation_control": str(out_root / "random_permutation_control.csv"),
            "bootstrap_prototype_stability": str(out_root / "bootstrap_prototype_stability.csv"),
        },
    }
    (out_root / "text_vision_graph_isomorphism_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# A8 Text/Vision Prototype Graph Isomorphism TAKEOVER",
        "",
        "- status: PASS",
        f"- output_root: {out_root}",
        f"- class_count: {len(class_ids)}",
        f"- neighbor_k: {k}",
        f"- projection_status: {projection_status}",
        "",
        "## Headline",
        f"- raw_text_vs_vtrain_spearman: {headline.get('raw_text_vs_vtrain_spearman')}",
        f"- raw_text_vs_vval_spearman: {headline.get('raw_text_vs_vval_spearman')}",
        f"- vtrain_vs_vval_spearman: {headline.get('vtrain_vs_vval_spearman')}",
        f"- overfit_minus_learned_text_vtrain_jaccard: {headline.get('overfit_minus_learned_text_vtrain_jaccard')}",
        f"- overfit_minus_learned_text_vval_jaccard: {headline.get('overfit_minus_learned_text_vval_jaccard')}",
        f"- overfit_minus_learned_vtrain_vval_jaccard: {headline.get('overfit_minus_learned_vtrain_vval_jaccard')}",
        f"- overfit_minus_learned_train_val_visual_proto_drift: {headline.get('overfit_minus_learned_train_val_visual_proto_drift')}",
        "",
        "## Notes",
        "- This audit compares graph structures over the same 525 raw_id nodes; it does not compare raw vector dimensions directly.",
        "- raw_text graph is built from text prototypes returned by load_text_vocab_with_names.",
        "- vision_train/vision_val graphs are built from GT trajectory carrier class means.",
        "- projected_text graph is included only if the checkpoint bundle exposes a callable projector; otherwise the audit remains valid for raw text vs raw visual graphs.",
        "- Bootstrap stability is a prototype-level sanity check; n=1 classes are deterministic and should not be over-interpreted as stable.",
        "",
        "## Artifacts",
    ]
    for p in payload["artifacts"].values():
        lines.append(f"- {p}")
    (out_root / "TEXT_VISION_GRAPH_ISOMORPHISM_TAKEOVER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
