#!/usr/bin/env python3
"""A10 simulated-manifold extrapolation ceiling audit.

Read-only diagnostic overlay.  It constructs vision-derived synthetic text
prototypes, fits text->vision mappings using class-level anchor splits, and
measures heldout extrapolation at class-prototype and row levels.

This script intentionally does not mutate training code, checkpoints, or data
assets.  It writes only under --output_root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_G8_ROOT = "codex/outputs/G8_inference_and_eval"
DEFAULT_RUN_NAME = "sinkhorn_safeneg_k32_b010_preaug_k10_repro_20260427"
DEFAULT_A10_NAME = "A10_SIMULATED_MANIFOLD_EXTRAPOLATION_CEILING"
DEFAULT_VARIANTS = "clip_current,clip_of_llm_mean,llama_hidden_mean,llama_direct_concept_mean"
DEFAULT_TRANSFORMS = "S0_identity,S1_orthogonal,S2_anisotropic_linear,S3_noisy_linear,S4_lowrank_linear"
DEFAULT_PROJECTORS = "identity,orthogonal_procrustes,ridge,least_squares,lowrank_ridge,oracle_inverse"
PERSON_RAW_ID = 773


def _repo_default() -> Path:
    return Path.cwd().resolve()


def _run_root_default(repo_root: Path) -> Path:
    preferred = repo_root / DEFAULT_G8_ROOT / DEFAULT_RUN_NAME
    return preferred if preferred.exists() else repo_root / DEFAULT_G8_ROOT


def _output_root_default(repo_root: Path) -> Path:
    return repo_root / DEFAULT_G8_ROOT / DEFAULT_A10_NAME


def _ensure_repo(repo_root: Path) -> None:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _load_a8_helper(repo_root: Path):
    path = repo_root / "tools" / "a8_manifold_alignment_diagnosis.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("_a8_manifold_helper", str(path))
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


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def _summary_ranks(ranks: Sequence[int], prefix: str = "") -> Dict[str, Any]:
    n = len(ranks)
    if n <= 0:
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
        prefix + "count": int(n),
        prefix + "rank@1": float(np.mean(arr <= 1)),
        prefix + "rank@5": float(np.mean(arr <= 5)),
        prefix + "rank@10": float(np.mean(arr <= 10)),
        prefix + "rank@20": float(np.mean(arr <= 20)),
        prefix + "rank@50": float(np.mean(arr <= 50)),
        prefix + "mean_rank": float(np.mean(arr)),
        prefix + "median_rank": float(np.median(arr)),
    }


def _mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def _median(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.median(vals)) if vals else float("nan")


def _read_csv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return next(csv.reader(f))
    except Exception:
        return []


def _count_visible_ids(path: Path) -> int:
    if not path.is_file():
        return 0
    header = _read_csv_header(path)
    if "raw_id" not in header:
        return 0
    try:
        raw_idx = header.index("raw_id")
        gap_idx = header.index("in_row_gap") if "in_row_gap" in header else None
        count = 0
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if raw_idx >= len(row) or not str(row[raw_idx]).strip():
                    continue
                if gap_idx is None or (gap_idx < len(row) and str(row[gap_idx]).strip() == "1"):
                    count += 1
        return count
    except Exception:
        return 0


def _find_visible_csv(repo_root: Path, run_root: Path) -> Optional[Path]:
    candidates = [
        run_root / "analysis/a8_base_116_visibility_audit/lvvis_train_base/base_641_visibility_by_class.csv",
        repo_root / DEFAULT_G8_ROOT / DEFAULT_RUN_NAME / "analysis/a8_base_116_visibility_audit/lvvis_train_base/base_641_visibility_by_class.csv",
    ]
    for p in candidates:
        if _count_visible_ids(p) == 525:
            return p
    roots = [run_root / "analysis", repo_root / DEFAULT_G8_ROOT]
    found: list[Path] = []
    for root in roots:
        if root.exists():
            found.extend(root.rglob("base_641_visibility_by_class.csv"))
            found.extend(root.rglob("*visibility_by_class.csv"))
    scored = []
    for p in found:
        n = _count_visible_ids(p)
        if n:
            scored.append((abs(n - 525), -p.stat().st_mtime, n, p))
    if scored:
        scored.sort()
        if scored[0][2] == 525:
            return scored[0][3]
    return None


def _find_per_class_join(repo_root: Path, run_root: Path) -> Optional[Path]:
    candidates = [
        run_root / "analysis/a8_dj3_semantic_boundary_bottleneck_audit/per_class_train_val_525_join.csv",
        repo_root / DEFAULT_G8_ROOT / DEFAULT_RUN_NAME / "analysis/a8_dj3_semantic_boundary_bottleneck_audit/per_class_train_val_525_join.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    found: list[Path] = []
    for root in [run_root / "analysis", repo_root / DEFAULT_G8_ROOT]:
        if root.exists():
            found.extend(root.rglob("per_class_train_val_525_join.csv"))
    if found:
        found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return found[0]
    return None


def _load_visible_ids(a8: Any, path: Path) -> set[int]:
    # Prefer the existing A8 semantics so visible525 remains identical.
    return set(int(x) for x in a8._load_visible_ids(path))


def _load_per_class_meta(path: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if path is None or not path.is_file():
        return out
    for row in _read_csv(path):
        rid = _as_int(row.get("raw_id"))
        if rid is None:
            continue
        out[int(rid)] = dict(row)
    return out


def _bucket_from_count(n: int) -> str:
    n = int(n)
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    if n <= 50:
        return "11-50"
    if n <= 200:
        return "51-200"
    return ">200"


def _support_bucket_for(rid: int, train_counts: Mapping[int, int], per_class: Mapping[int, Mapping[str, Any]]) -> str:
    row = per_class.get(int(rid), {})
    for key in ("support_bucket", "train_support_bucket", "bucket"):
        val = str(row.get(key, "")).strip()
        if val:
            return val
    return _bucket_from_count(int(train_counts.get(int(rid), 0)))


def _stratified_split(
    ids: Sequence[int],
    train_counts: Mapping[int, int],
    per_class: Mapping[int, Mapping[str, Any]],
    seed: int,
    train_frac: float,
    calib_frac: float,
    test_frac: float,
) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    by_bucket: Dict[str, List[int]] = defaultdict(list)
    for rid in ids:
        by_bucket[_support_bucket_for(int(rid), train_counts, per_class)].append(int(rid))
    train: List[int] = []
    calib: List[int] = []
    test: List[int] = []
    bucket_rows: List[Dict[str, Any]] = []
    for bucket, vals0 in sorted(by_bucket.items()):
        vals = list(vals0)
        rng.shuffle(vals)
        n = len(vals)
        if n <= 2:
            n_train = max(1, n - 1)
            n_calib = 0
        else:
            n_train = int(round(n * float(train_frac)))
            n_calib = int(round(n * float(calib_frac)))
            n_train = min(max(1, n_train), n - 2)
            n_calib = min(max(1, n_calib), n - n_train - 1)
        n_test = n - n_train - n_calib
        if n_test <= 0 and n > 1:
            n_train = max(1, n_train - 1)
            n_test = n - n_train - n_calib
        train.extend(vals[:n_train])
        calib.extend(vals[n_train:n_train + n_calib])
        test.extend(vals[n_train + n_calib:])
        bucket_rows.append({
            "support_bucket": bucket,
            "class_count": n,
            "anchor_train_count": n_train,
            "anchor_calib_count": n_calib,
            "heldout_test_count": n_test,
        })
    return sorted(train), sorted(calib), sorted(test), {"bucket_rows": bucket_rows}


def _orthogonal_matrix(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    # QR on a Gaussian matrix is deterministic for a fixed seed.  Normalize
    # signs so the result is stable across LAPACK sign conventions.
    a = rng.normal(size=(dim, dim)).astype(np.float64)
    q, r = np.linalg.qr(a)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    q = q * signs.reshape(1, -1)
    return q.astype(np.float64)


def _make_transform(name: str, dim: int, seed: int, noise_sigma: float, lowrank_dim: int) -> Dict[str, Any]:
    name = str(name)
    if name == "S0_identity":
        return {"name": name, "kind": "synthetic", "matrix": np.eye(dim, dtype=np.float64), "bias": None, "noise_sigma": 0.0, "rank": dim, "invertible": True}
    if name == "S1_orthogonal":
        q = _orthogonal_matrix(dim, 100_003 + int(seed))
        return {"name": name, "kind": "synthetic", "matrix": q, "bias": None, "noise_sigma": 0.0, "rank": dim, "invertible": True}
    if name == "S2_anisotropic_linear":
        q1 = _orthogonal_matrix(dim, 200_003 + int(seed))
        q2 = _orthogonal_matrix(dim, 300_007 + int(seed))
        scales = np.exp(np.linspace(math.log(0.35), math.log(2.25), dim)).astype(np.float64)
        a = q1 @ np.diag(scales) @ q2
        return {"name": name, "kind": "synthetic", "matrix": a, "bias": None, "noise_sigma": 0.0, "rank": dim, "invertible": True}
    if name == "S3_noisy_linear":
        base = _make_transform("S2_anisotropic_linear", dim, seed, noise_sigma, lowrank_dim)
        base = dict(base)
        base["name"] = name
        base["noise_sigma"] = float(noise_sigma)
        base["invertible"] = True
        return base
    if name == "S4_lowrank_linear":
        k = int(max(1, min(lowrank_dim, dim)))
        q = _orthogonal_matrix(dim, 400_009 + int(seed))
        scales = np.zeros(dim, dtype=np.float64)
        scales[:k] = np.linspace(1.0, 0.25, k, dtype=np.float64)
        a = q @ np.diag(scales) @ q.T
        return {"name": name, "kind": "synthetic", "matrix": a, "bias": None, "noise_sigma": 0.0, "rank": k, "invertible": False}
    raise ValueError(f"unsupported synthetic transform: {name}")


def _apply_transform(v: np.ndarray, transform: Mapping[str, Any], seed: int) -> np.ndarray:
    mat = np.asarray(transform["matrix"], dtype=np.float64)
    out = np.asarray(v, dtype=np.float64) @ mat
    if transform.get("bias") is not None:
        out = out + np.asarray(transform["bias"], dtype=np.float64)
    sigma = float(transform.get("noise_sigma", 0.0) or 0.0)
    if sigma > 0:
        rng = np.random.default_rng(500_021 + int(seed))
        out = out + rng.normal(scale=sigma, size=out.shape)
    return _l2(out.astype(np.float32))


def _oracle_inverse_project(t: np.ndarray, transform: Mapping[str, Any]) -> np.ndarray:
    name = str(transform.get("name"))
    if name == "S0_identity":
        return _l2(t)
    mat = np.asarray(transform["matrix"], dtype=np.float64)
    if transform.get("bias") is not None:
        t0 = np.asarray(t, dtype=np.float64) - np.asarray(transform["bias"], dtype=np.float64)
    else:
        t0 = np.asarray(t, dtype=np.float64)
    if bool(transform.get("invertible", False)):
        inv = np.linalg.inv(mat)
    else:
        inv = np.linalg.pinv(mat)
    return _l2((t0 @ inv).astype(np.float32))


def _augment_intercept(x: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(x, dtype=np.float64), np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)


def _fit_ridge_affine(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xa = _augment_intercept(x)
    y = np.asarray(y, dtype=np.float64)
    d = xa.shape[1]
    reg = float(alpha) * np.eye(d, dtype=np.float64)
    reg[-1, -1] = 0.0  # do not regularize bias
    if d > xa.shape[0]:
        # Primal on augmented features is still acceptable for these sizes, but
        # the solve can be singular for tiny anchor counts; fall back to pinv.
        try:
            w = np.linalg.solve(xa.T @ xa + reg, xa.T @ y)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(xa.T @ xa + reg) @ xa.T @ y
    else:
        try:
            w = np.linalg.solve(xa.T @ xa + reg, xa.T @ y)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(xa.T @ xa + reg) @ xa.T @ y
    return np.asarray(w, dtype=np.float32)


def _fit_lstsq_affine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xa = _augment_intercept(x)
    w, *_ = np.linalg.lstsq(xa, np.asarray(y, dtype=np.float64), rcond=None)
    return np.asarray(w, dtype=np.float32)


def _predict_affine(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return _l2((_augment_intercept(x) @ np.asarray(w, dtype=np.float64)).astype(np.float32))


def _fit_procrustes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = x.T @ y
    u, _s, vt = np.linalg.svd(m, full_matrices=False)
    r = u @ vt
    return np.asarray(r, dtype=np.float32)


def _fit_lowrank_ridge(x: np.ndarray, y: np.ndarray, alpha: float, rank: int) -> Dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    x_mean = x.mean(axis=0, keepdims=True)
    xc = x - x_mean
    # Basis learned from anchor_train only; this avoids heldout leakage.
    _u, _s, vt = np.linalg.svd(xc, full_matrices=False)
    k = int(max(1, min(rank, vt.shape[0])))
    basis = vt[:k].T.astype(np.float32)
    xr = xc @ basis
    w = _fit_ridge_affine(xr, y, alpha)
    return {"mean": x_mean.astype(np.float32), "basis": basis, "w": w}


def _predict_lowrank(x: np.ndarray, model: Mapping[str, np.ndarray]) -> np.ndarray:
    xr = (np.asarray(x, dtype=np.float64) - np.asarray(model["mean"], dtype=np.float64)) @ np.asarray(model["basis"], dtype=np.float64)
    return _predict_affine(xr, np.asarray(model["w"]))


def _fit_projector(
    projector: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alpha: float,
    lowrank_dim: int,
    transform: Optional[Mapping[str, Any]] = None,
) -> Any:
    projector = str(projector)
    if projector == "identity":
        return {"type": projector}
    if projector == "orthogonal_procrustes":
        return {"type": projector, "r": _fit_procrustes(x_train, y_train)}
    if projector == "ridge":
        return {"type": projector, "w": _fit_ridge_affine(x_train, y_train, alpha), "alpha": float(alpha)}
    if projector == "least_squares":
        return {"type": projector, "w": _fit_lstsq_affine(x_train, y_train)}
    if projector == "lowrank_ridge":
        return {"type": projector, "model": _fit_lowrank_ridge(x_train, y_train, alpha, lowrank_dim), "alpha": float(alpha), "lowrank_dim": int(lowrank_dim)}
    if projector == "oracle_inverse":
        if transform is None or str(transform.get("kind")) != "synthetic":
            raise ValueError("oracle_inverse is only valid for synthetic transforms")
        return {"type": projector, "transform": transform}
    raise ValueError(f"unsupported projector: {projector}")


def _apply_projector(x: np.ndarray, model: Any) -> np.ndarray:
    typ = str(model["type"])
    if typ == "identity":
        return _l2(x)
    if typ == "orthogonal_procrustes":
        return _l2(np.asarray(x, dtype=np.float64) @ np.asarray(model["r"], dtype=np.float64))
    if typ in {"ridge", "least_squares"}:
        return _predict_affine(x, np.asarray(model["w"]))
    if typ == "lowrank_ridge":
        return _predict_lowrank(x, model["model"])
    if typ == "oracle_inverse":
        return _oracle_inverse_project(x, model["transform"])
    raise ValueError(f"bad projector model type: {typ}")


def _compatible_projectors(feature_kind: str, transform_name: str, requested: Sequence[str]) -> List[str]:
    req = [str(x) for x in requested]
    if feature_kind == "real":
        allowed = {"ridge", "least_squares", "lowrank_ridge"}
    elif transform_name == "S0_identity":
        allowed = {"identity", "ridge", "least_squares", "lowrank_ridge", "oracle_inverse"}
    elif transform_name == "S1_orthogonal":
        allowed = {"orthogonal_procrustes", "ridge", "least_squares", "lowrank_ridge", "oracle_inverse"}
    else:
        allowed = {"ridge", "least_squares", "lowrank_ridge", "oracle_inverse"}
    return [p for p in req if p in allowed]


def _rank_against_candidates(scores: np.ndarray, cand_idx: Sequence[int], target_idx: int) -> Optional[int]:
    cand_idx = list(cand_idx)
    if target_idx not in cand_idx:
        return None
    vals = np.asarray(scores[cand_idx], dtype=np.float64)
    vals[~np.isfinite(vals)] = -np.inf
    order = np.argsort(-vals, kind="mergesort")
    local_target = cand_idx.index(target_idx)
    where = np.where(order == local_target)[0]
    if len(where) == 0:
        return None
    return int(where[0]) + 1


def _evaluate_class_retrieval(
    projected: np.ndarray,
    visual: np.ndarray,
    ids: Sequence[int],
    eval_ids: Sequence[int],
    candidate_ids: Sequence[int],
) -> Dict[str, Any]:
    id_to_idx = {int(r): i for i, r in enumerate(ids)}
    valid_visual = np.isfinite(visual).all(axis=1)
    cand_idx = [id_to_idx[int(r)] for r in candidate_ids if int(r) in id_to_idx and bool(valid_visual[id_to_idx[int(r)]])]
    eval_idx = [id_to_idx[int(r)] for r in eval_ids if int(r) in id_to_idx and bool(valid_visual[id_to_idx[int(r)]])]
    if not cand_idx or not eval_idx:
        return {"eval_count": 0, "candidate_count": len(cand_idx)}
    pt = _l2(np.nan_to_num(projected, nan=0.0))
    vv = _l2(np.nan_to_num(visual, nan=0.0))
    t2v_ranks: List[int] = []
    v2t_ranks: List[int] = []
    recovery_errors: List[float] = []
    for qi in eval_idx:
        target_rid = int(ids[qi])
        if target_rid not in candidate_ids:
            continue
        target_idx = qi
        sims = vv @ pt[qi]
        r = _rank_against_candidates(sims, cand_idx, target_idx)
        if r is not None:
            t2v_ranks.append(r)
        sims2 = pt @ vv[qi]
        r2 = _rank_against_candidates(sims2, cand_idx, target_idx)
        if r2 is not None:
            v2t_ranks.append(r2)
        recovery_errors.append(float(1.0 - float(np.dot(pt[qi], vv[qi]))))
    return {
        "eval_count": int(len(t2v_ranks)),
        "candidate_count": int(len(cand_idx)),
        **_summary_ranks(t2v_ranks, prefix="t2v_"),
        **_summary_ranks(v2t_ranks, prefix="v2t_"),
        "mean_cosine_recovery_error": _mean(recovery_errors),
        "median_cosine_recovery_error": _median(recovery_errors),
    }


def _select_score(met: Mapping[str, Any]) -> Tuple[float, float, float]:
    return (
        _safe_float(met.get("t2v_rank@1"), -1.0),
        _safe_float(met.get("t2v_rank@5"), -1.0),
        -_safe_float(met.get("t2v_mean_rank"), 1e12),
    )


def _candidate_ids_for_scope(scope: str, ids: Sequence[int], heldout_ids: Sequence[int], forced_hubs: Sequence[int]) -> List[int]:
    if scope == "heldout_only":
        return sorted(set(int(x) for x in heldout_ids))
    if scope == "visible525_all":
        return list(map(int, ids))
    if scope == "visible525_plus_forced_hubs":
        return sorted(set(map(int, ids)) | set(map(int, forced_hubs)))
    raise ValueError(f"unsupported candidate scope: {scope}")


def _evaluate_row_level(
    projected: np.ndarray,
    ids: Sequence[int],
    val_rows: Sequence[Mapping[str, Any]],
    val_carrier: np.ndarray,
    heldout_ids: Sequence[int],
    candidate_ids: Sequence[int],
    names: Mapping[int, str],
    row_limit: int,
    case_meta: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    id_to_idx = {int(r): i for i, r in enumerate(ids)}
    cand_ids = [int(r) for r in candidate_ids if int(r) in id_to_idx]
    cand_idx = [id_to_idx[int(r)] for r in cand_ids]
    if not cand_idx:
        return {"row_count": 0, "candidate_count": 0}, [], []
    pt = _l2(np.nan_to_num(projected, nan=0.0))
    cand_mat = pt[cand_idx]
    heldout_set = set(int(x) for x in heldout_ids)
    person_pos = cand_ids.index(PERSON_RAW_ID) if PERSON_RAW_ID in cand_ids else None
    ranks: List[int] = []
    margins_top_wrong: List[float] = []
    margins_person: List[float] = []
    top1_ids: List[int] = []
    per_row: List[Dict[str, Any]] = []
    by_class: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    scanned = 0
    for i, row in enumerate(val_rows):
        rid = _as_int(row.get("raw_category_id"))
        if rid is None or int(rid) not in heldout_set or int(rid) not in id_to_idx:
            continue
        if i >= int(val_carrier.shape[0]):
            continue
        z = np.asarray(val_carrier[i], dtype=np.float32)
        if not np.isfinite(z).all():
            continue
        if row_limit > 0 and scanned >= row_limit:
            break
        scanned += 1
        scores = cand_mat @ _l2(z.reshape(1, -1))[0]
        scores = np.asarray(scores, dtype=np.float64)
        order = np.argsort(-scores, kind="mergesort")
        target_pos = cand_ids.index(int(rid)) if int(rid) in cand_ids else None
        if target_pos is None:
            continue
        rank = int(np.where(order == target_pos)[0][0]) + 1
        top1_pos = int(order[0])
        top1_rid = int(cand_ids[top1_pos])
        nearest_wrong = float(scores[int(order[1])]) if len(order) > 1 and int(order[0]) == target_pos else float(scores[int(order[0])])
        if len(order) > 1 and int(order[0]) != target_pos:
            nearest_wrong = float(scores[int(order[0])])
        elif len(order) > 1:
            nearest_wrong = float(scores[int(order[1])])
        else:
            nearest_wrong = float("nan")
        gt_score = float(scores[target_pos])
        m_wrong = gt_score - nearest_wrong if math.isfinite(nearest_wrong) else float("nan")
        person_score = float(scores[person_pos]) if person_pos is not None else float("nan")
        m_person = gt_score - person_score if math.isfinite(person_score) else float("nan")
        top1_ids.append(top1_rid)
        ranks.append(rank)
        if math.isfinite(m_wrong):
            margins_top_wrong.append(m_wrong)
        if math.isfinite(m_person):
            margins_person.append(m_person)
        rec = {
            **case_meta,
            "trajectory_id": str(row.get("trajectory_id", row.get("track_id", ""))),
            "video_id": row.get("video_id", ""),
            "clip_id": row.get("clip_id", row.get("video_id", "")),
            "gt_raw_id": int(rid),
            "gt_name": names.get(int(rid), f"raw_id_{rid}"),
            "rank": rank,
            "top1_raw_id": top1_rid,
            "top1_name": names.get(top1_rid, f"raw_id_{top1_rid}"),
            "top1_is_gt": int(top1_rid == int(rid)),
            "top1_is_person": int(top1_rid == PERSON_RAW_ID),
            "gt_score": gt_score,
            "top1_score": float(scores[top1_pos]),
            "nearest_wrong_score": nearest_wrong,
            "person_score": person_score,
            "margin_gt_vs_top_wrong": m_wrong,
            "margin_gt_vs_person": m_person,
        }
        per_row.append(rec)
        by_class[int(rid)].append(rec)
    top_counter = Counter(top1_ids)
    summary = {
        **case_meta,
        "row_count": int(len(ranks)),
        "candidate_count": int(len(cand_ids)),
        "class_count": int(len(by_class)),
        **_summary_ranks(ranks, prefix="row_"),
        "mean_margin_gt_vs_top_wrong": _mean(margins_top_wrong),
        "median_margin_gt_vs_top_wrong": _median(margins_top_wrong),
        "positive_margin_gt_vs_top_wrong_rate": float(np.mean(np.asarray(margins_top_wrong) > 0)) if margins_top_wrong else 0.0,
        "mean_margin_gt_vs_person": _mean(margins_person),
        "positive_margin_gt_vs_person_rate": float(np.mean(np.asarray(margins_person) > 0)) if margins_person else 0.0,
        "top1_person_rate": float(top_counter.get(PERSON_RAW_ID, 0) / max(len(ranks), 1)),
        "top1_max_hub_raw_id": int(top_counter.most_common(1)[0][0]) if top_counter else None,
        "top1_max_hub_count": int(top_counter.most_common(1)[0][1]) if top_counter else 0,
    }
    per_class: List[Dict[str, Any]] = []
    for rid, rs in sorted(by_class.items()):
        rr = [int(r["rank"]) for r in rs]
        per_class.append({
            **{k: v for k, v in case_meta.items() if k not in {"candidate_scope"}},
            "candidate_scope": case_meta.get("candidate_scope"),
            "gt_raw_id": int(rid),
            "gt_name": names.get(int(rid), f"raw_id_{rid}"),
            "count": len(rs),
            **_summary_ranks(rr, prefix="row_"),
            "mean_margin_gt_vs_top_wrong": _mean([_safe_float(r.get("margin_gt_vs_top_wrong")) for r in rs]),
            "top1_person_rate": float(np.mean([int(r.get("top1_is_person", 0)) for r in rs])) if rs else 0.0,
        })
    return summary, per_row, per_class


def _build_real_feature_cases(a8: Any, asset_root: Path, dataset_name: str, visual_root: Path, direct_root: Path, variants: Sequence[str], ids: Sequence[int]) -> List[Dict[str, Any]]:
    text_banks = a8._load_all_text_banks(asset_root, dataset_name, visual_root, direct_root, variants)
    cases: List[Dict[str, Any]] = []
    for variant, (t_ids, t_mat, names, meta) in text_banks.items():
        mat = a8._submatrix_for_ids(t_ids, t_mat, ids)
        cases.append({
            "feature_kind": "real",
            "feature_name": variant,
            "transform": "real_text",
            "text_matrix": _l2(np.asarray(mat, dtype=np.float32)),
            "names": names,
            "meta": dict(meta),
        })
    return cases


def _select_best_configs(
    case: Mapping[str, Any],
    train_ids: Sequence[int],
    calib_ids: Sequence[int],
    ids: Sequence[int],
    visual_train: np.ndarray,
    projector_names: Sequence[str],
    ridge_alphas: Sequence[float],
    lowrank_dims: Sequence[int],
    candidate_ids: Sequence[int],
    transform_obj: Optional[Mapping[str, Any]],
    max_projector_errors: int = 5,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    id_to_idx = {int(r): i for i, r in enumerate(ids)}
    train_idx = [id_to_idx[int(r)] for r in train_ids if int(r) in id_to_idx and np.isfinite(visual_train[id_to_idx[int(r)]]).all()]
    calib_ids_valid = [int(r) for r in calib_ids if int(r) in id_to_idx and np.isfinite(visual_train[id_to_idx[int(r)]]).all()]
    if not calib_ids_valid:
        calib_ids_valid = [int(r) for r in train_ids[: max(1, min(10, len(train_ids)))] if int(r) in id_to_idx]
    x_all = np.asarray(case["text_matrix"], dtype=np.float32)
    y_all = np.asarray(visual_train, dtype=np.float32)
    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    errors = 0
    for pname in _compatible_projectors(str(case["feature_kind"]), str(case.get("transform", "")), projector_names):
        alpha_list = list(ridge_alphas) if pname in {"ridge", "lowrank_ridge"} else [float("nan")]
        rank_list = list(lowrank_dims) if pname == "lowrank_ridge" else [0]
        for alpha in alpha_list:
            for lr_dim in rank_list:
                try:
                    model = _fit_projector(pname, x_all[train_idx], y_all[train_idx], alpha=float(alpha) if math.isfinite(float(alpha)) else 0.0, lowrank_dim=int(lr_dim), transform=transform_obj)
                    projected = _apply_projector(x_all, model)
                    calib_met = _evaluate_class_retrieval(projected, visual_train, ids, calib_ids_valid, candidate_ids)
                    row = {
                        "feature_kind": case["feature_kind"],
                        "feature_name": case["feature_name"],
                        "transform": case.get("transform"),
                        "projector": pname,
                        "ridge_alpha": alpha if math.isfinite(float(alpha)) else "",
                        "lowrank_dim": int(lr_dim) if lr_dim else "",
                        "selection_split": "anchor_calib",
                        **calib_met,
                    }
                    rows.append(row)
                    score = _select_score(calib_met)
                    if best is None or score > best["score"]:
                        best = {"score": score, "model": model, "projected": projected, "row": row, "projector": pname, "ridge_alpha": alpha, "lowrank_dim": lr_dim}
                except Exception as exc:
                    errors += 1
                    rows.append({
                        "feature_kind": case["feature_kind"],
                        "feature_name": case["feature_name"],
                        "transform": case.get("transform"),
                        "projector": pname,
                        "ridge_alpha": alpha if math.isfinite(float(alpha)) else "",
                        "lowrank_dim": int(lr_dim) if lr_dim else "",
                        "selection_split": "anchor_calib",
                        "status": "FAIL",
                        "error": str(exc),
                    })
                    if errors > max_projector_errors and best is None:
                        continue
    if best is None:
        raise RuntimeError(f"no valid projector for case={case.get('feature_name')}")
    return best, rows


def run_fixed_split_ceiling(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    args = ctx["args"]
    ids: List[int] = list(ctx["ids"])
    names: Dict[int, str] = dict(ctx["names"])
    visual_train = np.asarray(ctx["visual_train_mat"], dtype=np.float32)
    visual_val = np.asarray(ctx["visual_val_mat"], dtype=np.float32)
    val_valid = np.asarray(ctx["val_valid"], dtype=bool)
    val_rows = ctx["val_rows"]
    val_carrier = np.asarray(ctx["val_carrier"], dtype=np.float32)
    per_class = ctx["per_class_meta"]
    train_counts = ctx["train_counts"]
    cases = ctx["feature_cases"]
    forced_hubs = [int(x) for x in str(args.forced_hubs).split(",") if str(x).strip()]
    candidate_scopes = [x.strip() for x in str(args.candidate_scopes).split(",") if x.strip()]
    target_visuals = [x.strip() for x in str(args.target_visuals).split(",") if x.strip()]
    projectors = [x.strip() for x in str(args.projectors).split(",") if x.strip()]
    ridge_alphas = [float(x) for x in str(args.ridge_alphas).split(",") if x.strip()]
    lowrank_dims = [int(x) for x in str(args.lowrank_dims).split(",") if x.strip()]
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    out_root: Path = ctx["analysis_root"] / "E10_1_3_fixed_split_ceiling"
    out_root.mkdir(parents=True, exist_ok=True)
    class_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    row_summary_rows: List[Dict[str, Any]] = []
    per_row_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    sanity_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        train_ids, calib_ids, heldout_ids, split_meta = _stratified_split(ids, train_counts, per_class, seed, float(args.anchor_train_fraction), float(args.anchor_calib_fraction), float(args.heldout_fraction))
        for br in split_meta["bucket_rows"]:
            split_rows.append({"seed": seed, **br})
        for case in cases:
            transform_obj = case.get("transform_obj")
            visible_candidate_ids = _candidate_ids_for_scope("visible525_all", ids, heldout_ids, forced_hubs)
            best, sel_rows = _select_best_configs(case, train_ids, calib_ids, ids, visual_train, projectors, ridge_alphas, lowrank_dims, visible_candidate_ids, transform_obj)
            for sr in sel_rows:
                selection_rows.append({"seed": seed, "anchor_train_count": len(train_ids), "anchor_calib_count": len(calib_ids), "heldout_test_count": len(heldout_ids), **sr})
            projected = np.asarray(best["projected"], dtype=np.float32)
            best_meta = {
                "seed": seed,
                "feature_kind": case["feature_kind"],
                "feature_name": case["feature_name"],
                "transform": case.get("transform"),
                "selected_projector": best["projector"],
                "selected_ridge_alpha": best["ridge_alpha"] if math.isfinite(float(best["ridge_alpha"])) else "",
                "selected_lowrank_dim": int(best["lowrank_dim"]) if best["lowrank_dim"] else "",
                "anchor_train_count": len(train_ids),
                "anchor_calib_count": len(calib_ids),
                "heldout_test_count": len(heldout_ids),
            }
            for target_name in target_visuals:
                if target_name == "train_proto":
                    vis = visual_train
                    eval_ids = heldout_ids
                elif target_name == "val_proto":
                    vis = visual_val
                    eval_ids = [int(r) for r in heldout_ids if int(r) in set(np.asarray(ids)[val_valid].tolist())]
                else:
                    raise ValueError(f"unsupported target_visual: {target_name}")
                for scope in candidate_scopes:
                    cand = _candidate_ids_for_scope(scope, ids, heldout_ids, forced_hubs)
                    if target_name == "val_proto":
                        # Candidate classes without val prototypes are ignored by evaluator.
                        pass
                    met = _evaluate_class_retrieval(projected, vis, ids, eval_ids, cand)
                    class_rows.append({**best_meta, "target_visual": target_name, "candidate_scope": scope, **met})
            if not args.skip_row_level:
                for scope in candidate_scopes:
                    cand = _candidate_ids_for_scope(scope, ids, heldout_ids, forced_hubs)
                    case_meta = {**best_meta, "candidate_scope": scope}
                    summ, per_row, per_class = _evaluate_row_level(projected, ids, val_rows, val_carrier, heldout_ids, cand, names, int(args.row_max_rows), case_meta)
                    row_summary_rows.append(summ)
                    if not args.no_per_row:
                        per_row_rows.extend(per_row)
                    per_class_rows.extend(per_class)
            # Explicit sanity rows for canonical expected-success cases.
            if case["feature_kind"] == "synthetic" and case.get("transform") in {"S0_identity", "S1_orthogonal"}:
                expected = "identity" if case.get("transform") == "S0_identity" else "orthogonal_procrustes"
                if expected in _compatible_projectors("synthetic", str(case.get("transform")), projectors):
                    try:
                        id_to_idx = {int(r): i for i, r in enumerate(ids)}
                        train_idx = [id_to_idx[int(r)] for r in train_ids]
                        model = _fit_projector(expected, np.asarray(case["text_matrix"])[train_idx], visual_train[train_idx], alpha=float(ridge_alphas[0]), lowrank_dim=int(lowrank_dims[0]), transform=transform_obj)
                        proj = _apply_projector(np.asarray(case["text_matrix"]), model)
                        met = _evaluate_class_retrieval(proj, visual_train, ids, heldout_ids, visible_candidate_ids)
                        sanity_rows.append({"seed": seed, "transform": case.get("transform"), "expected_projector": expected, **met})
                    except Exception as exc:
                        sanity_rows.append({"seed": seed, "transform": case.get("transform"), "expected_projector": expected, "status": "FAIL", "error": str(exc)})
    _write_csv(out_root / "split_inventory.csv", split_rows)
    _write_csv(out_root / "projector_selection_by_calib.csv", selection_rows)
    _write_csv(out_root / "class_proto_heldout_ceiling_summary.csv", class_rows)
    _write_csv(out_root / "row_level_heldout_ceiling_summary.csv", row_summary_rows)
    _write_csv(out_root / "row_level_heldout_ceiling_per_class.csv", per_class_rows)
    if not args.no_per_row:
        _write_csv(out_root / "row_level_heldout_ceiling_per_row.csv", per_row_rows)
    _write_csv(out_root / "sanity_gate_summary.csv", sanity_rows)
    sanity_pass = True
    for r in sanity_rows:
        if str(r.get("status", "PASS")) == "FAIL" or _safe_float(r.get("t2v_rank@1"), 0.0) < float(args.sanity_min_rank1):
            sanity_pass = False
    payload = {
        "status": "PASS",
        "sanity_gate_status": "PASS" if sanity_pass else "WARN",
        "summary_rows": len(class_rows),
        "row_summary_rows": len(row_summary_rows),
        "selection_rows": len(selection_rows),
        "artifacts": {
            "class_summary_csv": str(out_root / "class_proto_heldout_ceiling_summary.csv"),
            "row_summary_csv": str(out_root / "row_level_heldout_ceiling_summary.csv"),
            "sanity_csv": str(out_root / "sanity_gate_summary.csv"),
            "selection_csv": str(out_root / "projector_selection_by_calib.csv"),
            "per_class_csv": str(out_root / "row_level_heldout_ceiling_per_class.csv"),
            "per_row_csv": str(out_root / "row_level_heldout_ceiling_per_row.csv") if not args.no_per_row else "",
        },
    }
    _write_json(out_root / "fixed_split_ceiling_summary.json", payload)
    return payload


def run_anchor_curve(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    args = ctx["args"]
    ids: List[int] = list(ctx["ids"])
    visual_train = np.asarray(ctx["visual_train_mat"], dtype=np.float32)
    cases = ctx["feature_cases"]
    per_class = ctx["per_class_meta"]
    train_counts = ctx["train_counts"]
    projectors = [x.strip() for x in str(args.projectors).split(",") if x.strip()]
    ridge_alphas = [float(x) for x in str(args.ridge_alphas).split(",") if x.strip()]
    lowrank_dims = [int(x) for x in str(args.lowrank_dims).split(",") if x.strip()]
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    anchor_counts = [int(x) for x in str(args.anchor_counts).split(",") if x.strip()]
    forced_hubs = [int(x) for x in str(args.forced_hubs).split(",") if str(x).strip()]
    out_root: Path = ctx["analysis_root"] / "E10_2_anchor_count_curve"
    out_root.mkdir(parents=True, exist_ok=True)
    id_to_idx = {int(r): i for i, r in enumerate(ids)}
    rows: List[Dict[str, Any]] = []
    # For the curve, use all classes not sampled as anchors as the heldout set.
    # Hyperparameters are fixed to first alpha/rank to avoid using heldout for selection.
    alpha = float(ridge_alphas[0])
    lr_dim = int(lowrank_dims[0])
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        # Stratify approximately by sampling per bucket proportional to bucket size.
        by_bucket: Dict[str, List[int]] = defaultdict(list)
        for rid in ids:
            by_bucket[_support_bucket_for(int(rid), train_counts, per_class)].append(int(rid))
        for anchor_count in anchor_counts:
            if anchor_count >= len(ids):
                continue
            anchor_ids: List[int] = []
            for bucket, vals0 in sorted(by_bucket.items()):
                vals = list(vals0)
                rng.shuffle(vals)
                take = int(round(anchor_count * len(vals) / max(len(ids), 1)))
                take = min(max(1 if anchor_count >= len(by_bucket) else 0, take), len(vals))
                anchor_ids.extend(vals[:take])
            if len(anchor_ids) > anchor_count:
                rng.shuffle(anchor_ids)
                anchor_ids = anchor_ids[:anchor_count]
            elif len(anchor_ids) < anchor_count:
                rem = [int(r) for r in ids if int(r) not in set(anchor_ids)]
                rng.shuffle(rem)
                anchor_ids.extend(rem[: anchor_count - len(anchor_ids)])
            anchor_ids = sorted(set(anchor_ids))
            heldout_ids = sorted(set(map(int, ids)) - set(anchor_ids))
            train_idx = [id_to_idx[int(r)] for r in anchor_ids]
            cand = _candidate_ids_for_scope("visible525_all", ids, heldout_ids, forced_hubs)
            for case in cases:
                # Keep the curve compact: use the expected canonical projector for
                # synthetic positives and ridge for real controls unless explicitly
                # requested projectors exclude them.
                default_p = "ridge"
                if case["feature_kind"] == "synthetic":
                    if case.get("transform") == "S0_identity":
                        default_p = "identity"
                    elif case.get("transform") == "S1_orthogonal":
                        default_p = "orthogonal_procrustes"
                    else:
                        default_p = "ridge"
                if default_p not in projectors or default_p not in _compatible_projectors(str(case["feature_kind"]), str(case.get("transform", "")), projectors):
                    continue
                try:
                    model = _fit_projector(default_p, np.asarray(case["text_matrix"])[train_idx], visual_train[train_idx], alpha=alpha, lowrank_dim=lr_dim, transform=case.get("transform_obj"))
                    projected = _apply_projector(np.asarray(case["text_matrix"]), model)
                    met = _evaluate_class_retrieval(projected, visual_train, ids, heldout_ids, cand)
                    rows.append({
                        "seed": seed,
                        "feature_kind": case["feature_kind"],
                        "feature_name": case["feature_name"],
                        "transform": case.get("transform"),
                        "projector": default_p,
                        "anchor_count": len(anchor_ids),
                        "heldout_count": len(heldout_ids),
                        "target_visual": "train_proto",
                        "candidate_scope": "visible525_all",
                        **met,
                    })
                except Exception as exc:
                    rows.append({
                        "seed": seed,
                        "feature_kind": case["feature_kind"],
                        "feature_name": case["feature_name"],
                        "transform": case.get("transform"),
                        "projector": default_p,
                        "anchor_count": len(anchor_ids),
                        "heldout_count": len(heldout_ids),
                        "status": "FAIL",
                        "error": str(exc),
                    })
    # Compact aggregate.
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r.get("feature_kind"), r.get("feature_name"), r.get("transform"), r.get("projector"), r.get("anchor_count"), r.get("candidate_scope"))
        groups[key].append(r)
    summary: List[Dict[str, Any]] = []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        summary.append({
            "feature_kind": key[0],
            "feature_name": key[1],
            "transform": key[2],
            "projector": key[3],
            "anchor_count": key[4],
            "candidate_scope": key[5],
            "seed_count": len(rs),
            "t2v_rank@1_mean": _mean([_safe_float(r.get("t2v_rank@1")) for r in rs]),
            "t2v_rank@5_mean": _mean([_safe_float(r.get("t2v_rank@5")) for r in rs]),
            "t2v_mean_rank_mean": _mean([_safe_float(r.get("t2v_mean_rank")) for r in rs]),
            "v2t_rank@1_mean": _mean([_safe_float(r.get("v2t_rank@1")) for r in rs]),
            "v2t_rank@5_mean": _mean([_safe_float(r.get("v2t_rank@5")) for r in rs]),
            "v2t_mean_rank_mean": _mean([_safe_float(r.get("v2t_mean_rank")) for r in rs]),
        })
    _write_csv(out_root / "anchor_count_curve_rows.csv", rows)
    _write_csv(out_root / "anchor_count_curve_summary.csv", summary)
    payload = {"status": "PASS", "row_count": len(rows), "summary_count": len(summary), "artifacts": {"rows_csv": str(out_root / "anchor_count_curve_rows.csv"), "summary_csv": str(out_root / "anchor_count_curve_summary.csv")}}
    _write_json(out_root / "anchor_count_curve_summary.json", payload)
    return payload


def run_oracle_decomposition(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    analysis_root: Path = ctx["analysis_root"]
    out_root = analysis_root / "E10_4_oracle_decomposition"
    out_root.mkdir(parents=True, exist_ok=True)
    class_csv = analysis_root / "E10_1_3_fixed_split_ceiling" / "class_proto_heldout_ceiling_summary.csv"
    row_csv = analysis_root / "E10_1_3_fixed_split_ceiling" / "row_level_heldout_ceiling_summary.csv"
    rows: List[Dict[str, Any]] = []
    if class_csv.is_file():
        data = _read_csv(class_csv)
        groups: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
        for r in data:
            if str(r.get("target_visual")) != "train_proto" or str(r.get("candidate_scope")) != "visible525_all":
                continue
            groups[(str(r.get("seed")), str(r.get("feature_kind")), str(r.get("transform")))] .append(r)
        for key, rs in groups.items():
            best = max(rs, key=lambda r: (_safe_float(r.get("t2v_rank@1"), -1), _safe_float(r.get("t2v_rank@5"), -1), -_safe_float(r.get("t2v_mean_rank"), 1e12)))
            rows.append({"oracle_type": "best_projector_by_calib_then_test", "seed": key[0], "feature_kind": key[1], "transform": key[2], **{f"class_{k}": v for k, v in best.items() if k in {"selected_projector", "t2v_rank@1", "t2v_rank@5", "t2v_mean_rank", "v2t_rank@1", "v2t_rank@5", "v2t_mean_rank"}}})
    if row_csv.is_file():
        data = _read_csv(row_csv)
        for r in data:
            rk5 = _safe_float(r.get("row_rank@5"), 0.0)
            rk1 = _safe_float(r.get("row_rank@1"), 0.0)
            rows.append({
                "oracle_type": "top5_rerank_oracle",
                "seed": r.get("seed"),
                "feature_kind": r.get("feature_kind"),
                "transform": r.get("transform"),
                "candidate_scope": r.get("candidate_scope"),
                "selected_projector": r.get("selected_projector"),
                "row_rank@1": rk1,
                "row_rank@5": rk5,
                "top5_rerank_oracle_row_rank@1": rk5,
                "top5_to_top1_gap": rk5 - rk1,
                "top1_person_rate": r.get("top1_person_rate"),
            })
    _write_csv(out_root / "oracle_decomposition_summary.csv", rows)
    payload = {"status": "PASS", "row_count": len(rows), "artifacts": {"summary_csv": str(out_root / "oracle_decomposition_summary.csv")}}
    _write_json(out_root / "oracle_decomposition_summary.json", payload)
    return payload


def collect_summary(output_root: Path, analysis_root: Path) -> Dict[str, Any]:
    manifest: List[Dict[str, Any]] = []
    compact: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    for p in sorted(analysis_root.rglob("*.csv")):
        rel = str(p.relative_to(analysis_root))
        try:
            rows = _read_csv(p)
        except Exception:
            rows = []
        manifest.append({"artifact": rel, "row_count": len(rows)})
        if "per_row" in p.name:
            continue
        for idx, row in enumerate(rows):
            crow = {"artifact": rel, "row_index": idx, **row}
            compact.append(crow)
            for k, v in row.items():
                try:
                    fv = float(v)
                    if math.isfinite(fv):
                        long_rows.append({"artifact": rel, "row_index": idx, "metric": k, "value": fv, **{kk: row.get(kk, "") for kk in ("feature_kind", "feature_name", "transform", "projector", "selected_projector", "candidate_scope", "target_visual", "anchor_count", "seed")}})
                except Exception:
                    continue
    _write_csv(analysis_root / "A10_artifact_manifest.csv", manifest)
    _write_csv(analysis_root / "A10_simulated_manifold_summary.csv", compact)
    _write_csv(analysis_root / "A10_simulated_manifold_long_metrics.csv", long_rows)
    payload = {
        "status": "PASS",
        "output_root": str(output_root),
        "analysis_root": str(analysis_root),
        "artifact_count": len(manifest),
        "compact_summary_rows": len(compact),
        "long_metric_rows": len(long_rows),
        "artifacts": {
            "manifest_csv": str(analysis_root / "A10_artifact_manifest.csv"),
            "summary_csv": str(analysis_root / "A10_simulated_manifold_summary.csv"),
            "long_metrics_csv": str(analysis_root / "A10_simulated_manifold_long_metrics.csv"),
        },
    }
    _write_json(analysis_root / "A10_simulated_manifold_summary.json", payload)
    return payload


def make_takeover(output_root: Path, result: Mapping[str, Any]) -> None:
    analysis_root = Path(str(result.get("analysis_root", output_root / "analysis")))
    sanity_csv = analysis_root / "E10_1_3_fixed_split_ceiling" / "sanity_gate_summary.csv"
    class_csv = analysis_root / "E10_1_3_fixed_split_ceiling" / "class_proto_heldout_ceiling_summary.csv"
    row_csv = analysis_root / "E10_1_3_fixed_split_ceiling" / "row_level_heldout_ceiling_summary.csv"
    lines: List[str] = []
    lines.append("# A10 Simulated Manifold Extrapolation Ceiling TAKEOVER\n")
    lines.append("## Status")
    lines.append(f"- overall_status: `{result.get('status')}`")
    lines.append(f"- output_root: `{output_root}`")
    lines.append(f"- analysis_root: `{analysis_root}`")
    lines.append("\n## Required artifacts")
    for p in [
        analysis_root / "A10_simulated_manifold_summary.csv",
        analysis_root / "A10_simulated_manifold_summary.json",
        analysis_root / "A10_simulated_manifold_long_metrics.csv",
        analysis_root / "A10_artifact_manifest.csv",
        sanity_csv,
        class_csv,
        row_csv,
    ]:
        lines.append(f"- `{p}`: {'FOUND' if p.is_file() else 'MISSING'}")
    lines.append("\n## Sanity gate quick view")
    if sanity_csv.is_file():
        rows = _read_csv(sanity_csv)[:12]
        for r in rows:
            lines.append(f"- seed={r.get('seed')} transform={r.get('transform')} projector={r.get('expected_projector')} t2v@1={r.get('t2v_rank@1')} t2v@5={r.get('t2v_rank@5')} mean_rank={r.get('t2v_mean_rank')}")
    else:
        lines.append("- sanity gate file missing")
    lines.append("\n## Interpretation checklist")
    lines.append("- Synthetic class-level high and real-text class-level low => real text/vision manifold mismatch evidence.")
    lines.append("- Synthetic class-level high but synthetic row-level low => row-level semantic boundary / hub competition evidence.")
    lines.append("- Synthetic heldout-only row high but visible525 row low => global distractor/hub competition evidence.")
    lines.append("- Synthetic S0/S1 sanity failure => do not draw scientific conclusions; inspect normalization/projector/prototype pipeline first.")
    lines.append("\n## Scope")
    lines.append("- Read-only audit. No training code, model code, loss, text bank, or carrier bank is modified.")
    lines.append("- Large per-row files may remain remote; use summary/manifest/TAKEOVER for local lightweight review.\n")
    (output_root / "A10_SIMULATED_MANIFOLD_EXTRAPOLATION_CEILING_TAKEOVER.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run A10 vision-derived synthetic text manifold extrapolation ceiling audit.")
    p.add_argument("--repo_root", default=str(_repo_default()))
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--run_root", default="")
    p.add_argument("--output_root", default="")
    p.add_argument("--train_dataset_name", default="lvvis_train_base")
    p.add_argument("--val_dataset_name", default="lvvis_val")
    p.add_argument("--train_annotation_json", default="")
    p.add_argument("--val_annotation_json", default="")
    p.add_argument("--visible_csv", default="")
    p.add_argument("--per_class_join", default="")
    p.add_argument("--visual_only_root", default="/home/zyy/code/wsovvis_asserts/text_bank_llama3/lvvis/lvvis_visual_only_v1")
    p.add_argument("--direct_concept_root", default="/home/zyy/code/wsovvis_asserts/text_bank_llama3/lvvis/llama3_direct_concept_v1")
    p.add_argument("--variants", default=DEFAULT_VARIANTS)
    p.add_argument("--synthetic_transforms", default=DEFAULT_TRANSFORMS)
    p.add_argument("--projectors", default=DEFAULT_PROJECTORS)
    p.add_argument("--ridge_alphas", default="0.01")
    p.add_argument("--lowrank_dims", default="128")
    p.add_argument("--synthetic_noise_sigma", type=float, default=0.02)
    p.add_argument("--synthetic_lowrank_dim", type=int, default=128)
    p.add_argument("--anchor_train_fraction", type=float, default=0.70)
    p.add_argument("--anchor_calib_fraction", type=float, default=0.10)
    p.add_argument("--heldout_fraction", type=float, default=0.20)
    p.add_argument("--anchor_counts", default="32,64,128,256,384,420,450")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--candidate_scopes", default="heldout_only,visible525_all,visible525_plus_forced_hubs")
    p.add_argument("--target_visuals", default="train_proto,val_proto")
    p.add_argument("--forced_hubs", default=str(PERSON_RAW_ID))
    p.add_argument("--max_rows", type=int, default=0, help="Limit source rows loaded from GT carrier helpers; 0 means full.")
    p.add_argument("--row_max_rows", type=int, default=0, help="Limit evaluated heldout val rows per config; 0 means full.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_max_rows", type=int, default=2000)
    p.add_argument("--smoke_row_max_rows", type=int, default=500)
    p.add_argument("--device", default="cuda:0", help="Accepted for workflow consistency; this audit is numpy/CPU-bound.")
    p.add_argument("--skip_row_level", action="store_true")
    p.add_argument("--skip_anchor_curve", action="store_true")
    p.add_argument("--skip_oracle", action="store_true")
    p.add_argument("--no_per_row", action="store_true", help="Do not write row-level per-row CSV.")
    p.add_argument("--sanity_min_rank1", type=float, default=0.95)
    p.add_argument("--continue_on_error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve() if args.run_root else _run_root_default(repo_root)
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else _output_root_default(repo_root)
    analysis_root = output_root / "analysis"
    logs_root = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        if int(args.max_rows) <= 0:
            args.max_rows = int(args.smoke_max_rows)
        if int(args.row_max_rows) <= 0:
            args.row_max_rows = int(args.smoke_row_max_rows)
        # Keep smoke fast while still validating every code path.
        args.seeds = ",".join(str(x) for x in str(args.seeds).split(",")[:2])
        args.anchor_counts = "32,128"
        args.ridge_alphas = "0.01"
        args.lowrank_dims = "64"
    _ensure_repo(repo_root)
    a8 = _load_a8_helper(repo_root)
    visible_csv = Path(args.visible_csv).expanduser().resolve() if args.visible_csv else _find_visible_csv(repo_root, run_root)
    if not visible_csv or not visible_csv.is_file():
        raise RuntimeError("Could not locate visible525 CSV. Pass --visible_csv explicitly.")
    per_class_join = Path(args.per_class_join).expanduser().resolve() if args.per_class_join else _find_per_class_join(repo_root, run_root)
    visible_ids = _load_visible_ids(a8, visible_csv)
    train_ann = Path(args.train_annotation_json).expanduser().resolve() if args.train_annotation_json else repo_root / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "train_instances.json"
    val_ann = Path(args.val_annotation_json).expanduser().resolve() if args.val_annotation_json else repo_root / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "val_instances.json"
    gtceil = a8._load_gtceil(repo_root)
    train_rows, train_carrier, train_meta = a8._rows_and_carriers(gtceil, asset_root=asset_root, dataset_name=args.train_dataset_name, ann=train_ann, max_rows=int(args.max_rows))
    val_rows, val_carrier, val_meta = a8._rows_and_carriers(gtceil, asset_root=asset_root, dataset_name=args.val_dataset_name, ann=val_ann, max_rows=int(args.max_rows))
    train_proto, train_counts = a8._visual_prototypes(train_rows, train_carrier)
    val_proto, val_counts = a8._visual_prototypes(val_rows, val_carrier)
    # Start from canonical CLIP ids and visible ids; require train prototype for fit scope.
    clip_ids, _clip_mat, clip_names, _clip_meta = a8._load_current_clip_text_bank(asset_root, args.train_dataset_name)
    ids = sorted(set(map(int, clip_ids)) & set(map(int, visible_ids)) & set(map(int, train_proto.keys())))
    if len(ids) < 10:
        raise RuntimeError(f"too few scoped ids: {len(ids)}")
    visual_train_mat, train_valid = a8._matrix_for_ids(train_proto, ids)
    visual_val_mat, val_valid = a8._matrix_for_ids(val_proto, ids)
    visual_train_mat = _l2(np.nan_to_num(visual_train_mat, nan=0.0))
    # Keep NaNs for val validity, but normalize finite rows.
    finite_val = np.asarray(val_valid, dtype=bool)
    vt = np.asarray(visual_val_mat, dtype=np.float32)
    vt[finite_val] = _l2(vt[finite_val])
    visual_val_mat = vt
    per_class_meta = _load_per_class_meta(per_class_join)
    variants = [x.strip() for x in str(args.variants).split(",") if x.strip()]
    real_cases = _build_real_feature_cases(a8, asset_root, args.train_dataset_name, Path(args.visual_only_root).expanduser().resolve(), Path(args.direct_concept_root).expanduser().resolve(), variants, ids)
    synthetic_names = [x.strip() for x in str(args.synthetic_transforms).split(",") if x.strip()]
    synth_cases: List[Dict[str, Any]] = []
    # Transform seed is fixed here; split seeds are separate. This makes the
    # synthetic manifold stable across split repeats.
    transform_seed = 20260512
    dim = int(visual_train_mat.shape[1])
    for sname in synthetic_names:
        tobj = _make_transform(sname, dim, transform_seed, float(args.synthetic_noise_sigma), int(args.synthetic_lowrank_dim))
        tmat = _apply_transform(visual_train_mat, tobj, transform_seed)
        synth_cases.append({
            "feature_kind": "synthetic",
            "feature_name": sname,
            "transform": sname,
            "transform_obj": tobj,
            "text_matrix": tmat,
            "names": clip_names,
            "meta": {"source": "vision_derived_synthetic_text", "feature_dim": dim, "transform": sname, "transform_rank": tobj.get("rank"), "noise_sigma": tobj.get("noise_sigma", 0.0)},
        })
    feature_cases = synth_cases + real_cases
    ctx = {
        "args": args,
        "repo_root": repo_root,
        "asset_root": asset_root,
        "run_root": run_root,
        "output_root": output_root,
        "analysis_root": analysis_root,
        "ids": ids,
        "names": clip_names,
        "visual_train_mat": visual_train_mat,
        "visual_val_mat": visual_val_mat,
        "train_valid": train_valid,
        "val_valid": val_valid,
        "train_rows": train_rows,
        "train_carrier": train_carrier,
        "val_rows": val_rows,
        "val_carrier": val_carrier,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "per_class_meta": per_class_meta,
        "feature_cases": feature_cases,
        "visible_csv": visible_csv,
        "per_class_join": per_class_join,
        "train_meta": train_meta,
        "val_meta": val_meta,
    }
    result: Dict[str, Any] = {
        "status": "PASS",
        "repo_root": str(repo_root),
        "asset_root": str(asset_root),
        "run_root": str(run_root),
        "output_root": str(output_root),
        "analysis_root": str(analysis_root),
        "visible_csv": str(visible_csv),
        "per_class_join": str(per_class_join) if per_class_join else "",
        "class_scope": "visible525_with_train_gt_visual_proto",
        "class_count": len(ids),
        "synthetic_transforms": synthetic_names,
        "real_variants": variants,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        result["fixed_split_ceiling"] = run_fixed_split_ceiling(ctx)
        if not args.skip_anchor_curve:
            result["anchor_count_curve"] = run_anchor_curve(ctx)
        if not args.skip_oracle:
            result["oracle_decomposition"] = run_oracle_decomposition(ctx)
        result["collector"] = collect_summary(output_root, analysis_root)
        make_takeover(output_root, {**result, "analysis_root": str(analysis_root)})
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = str(exc)
        _write_json(analysis_root / "A10_simulated_manifold_summary.json", result)
        make_takeover(output_root, {**result, "analysis_root": str(analysis_root)})
        if not args.continue_on_error:
            raise
    result["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(analysis_root / "A10_run_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
