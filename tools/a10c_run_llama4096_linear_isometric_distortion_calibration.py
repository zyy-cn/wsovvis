#!/usr/bin/env python3
"""A10C LLaMA4096 real-residual linear-isometric manifold calibration.

Read-only diagnostic overlay.

This audit constructs an ideal raw LLaMA4096 text manifold whose pairwise class
geometry is exactly inherited from DINOv2 visual class prototypes, then moves
from that ideal manifold toward real LLaMA hidden-mean text features with
row-wise spherical interpolation.  For every interpolation alpha, it refits a
single linear text->vision projector using visible525 train anchors only, and
measures class-prototype extrapolation on val-base and novel-val under
full-available open-vocabulary competition.

It intentionally does not mutate training code, checkpoints, text banks, carrier
banks, annotations, package specs, or G7/G8 pipeline files.  It writes only under
--output_root.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_G8_ROOT = "codex/outputs/G8_inference_and_eval"
DEFAULT_RUN_NAME = "sinkhorn_safeneg_k32_b010_preaug_k10_repro_20260427"
DEFAULT_OUT_NAME = "A10C_REAL_RESIDUAL_LLAMA4096_LINEAR_ISOMETRIC_CALIBRATION"
DEFAULT_VISUAL_ONLY_ROOT = "/home/zyy/code/wsovvis_asserts/text_bank_llama3/lvvis/lvvis_visual_only_v1"
PERSON_RAW_ID = 773


def _import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


def _resolve_device(device_arg: str):
    torch = _import_torch()
    if torch is None:
        return None, "numpy"
    dev = str(device_arg or "auto").strip().lower()
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev.startswith("cuda") and not torch.cuda.is_available():
        dev = "cpu"
    try:
        device = torch.device(dev)
    except Exception:
        device = torch.device("cpu")
    return device, str(device)


def _torch_l2(x, eps: float = 1e-12):
    torch = _import_torch()
    return x / torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=eps)


def _np_to_torch(x: np.ndarray, device):
    torch = _import_torch()
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


def _repo_default() -> Path:
    return Path.cwd().resolve()


def _run_root_default(repo_root: Path) -> Path:
    p = repo_root / DEFAULT_G8_ROOT / DEFAULT_RUN_NAME
    return p if p.exists() else repo_root / DEFAULT_G8_ROOT


def _output_root_default(repo_root: Path) -> Path:
    return repo_root / DEFAULT_G8_ROOT / DEFAULT_OUT_NAME


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_a10(repo_root: Path):
    path = repo_root / "tools" / "a10_run_simulated_manifold_extrapolation_ceiling.py"
    if not path.is_file():
        raise FileNotFoundError(f"A10 helper missing: {path}")
    return _load_module(path, "_a10_helper_for_a10c")


def _load_a10b(repo_root: Path):
    path = repo_root / "tools" / "a10b_run_cross_scope_manifold_extrapolation_audit.py"
    if not path.is_file():
        raise FileNotFoundError(f"A10B helper missing: {path}")
    return _load_module(path, "_a10b_helper_for_a10c")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


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


def _finite_rows(x: np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(x)).all(axis=1)


def _upper_triangle_values(sim: np.ndarray) -> np.ndarray:
    n = sim.shape[0]
    if n < 2:
        return np.asarray([], dtype=np.float64)
    idx = np.triu_indices(n, k=1)
    vals = np.asarray(sim[idx], dtype=np.float64)
    return vals[np.isfinite(vals)]


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Small scipy-free rankdata(method='average') for Spearman."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return float("nan")
    ra = _rankdata_average(a[m])
    rb = _rankdata_average(b[m])
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(np.dot(ra, rb) / den) if den > 0 else float("nan")


def _topk_indices(sim: np.ndarray, k: int) -> np.ndarray:
    n = int(sim.shape[0])
    kk = int(max(0, min(k, max(n - 1, 0))))
    if kk <= 0:
        return np.zeros((n, 0), dtype=np.int64)
    s = np.asarray(sim, dtype=np.float64).copy()
    np.fill_diagonal(s, -np.inf)
    # argpartition is faster but argsort is deterministic and small enough here.
    return np.argsort(-s, axis=1, kind="mergesort")[:, :kk]


def _mean_knn_overlap(sim_a: np.ndarray, sim_b: np.ndarray, k: int) -> float:
    if sim_a.shape[0] <= 1 or sim_b.shape[0] <= 1:
        return float("nan")
    ia = _topk_indices(sim_a, k)
    ib = _topk_indices(sim_b, k)
    if ia.shape[1] == 0:
        return float("nan")
    vals = []
    for x, y in zip(ia, ib):
        vals.append(len(set(map(int, x)).intersection(set(map(int, y)))) / float(ia.shape[1]))
    return float(np.mean(vals)) if vals else float("nan")


def _triplet_agreement(sim_x: np.ndarray, sim_v: np.ndarray, seed: int, samples: int) -> float:
    n = int(sim_x.shape[0])
    if n < 3 or samples <= 0:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    agree = 0
    total = 0
    for _ in range(int(samples)):
        i, j, k = rng.choice(n, size=3, replace=False)
        dx = float(sim_x[i, j] - sim_x[i, k])
        dv = float(sim_v[i, j] - sim_v[i, k])
        if not (math.isfinite(dx) and math.isfinite(dv)):
            continue
        if dx == 0 or dv == 0:
            continue
        total += 1
        if (dx > 0) == (dv > 0):
            agree += 1
    return float(agree / total) if total else float("nan")


def _hubness_top1_concentration(sim: np.ndarray) -> Dict[str, Any]:
    n = int(sim.shape[0])
    if n <= 1:
        return {"hubness_top1_concentration": float("nan"), "hubness_top1_max_count": 0, "hubness_top1_max_local_index": None}
    s = np.asarray(sim, dtype=np.float64).copy()
    np.fill_diagonal(s, -np.inf)
    nn = np.argmax(s, axis=1)
    counts = np.bincount(nn, minlength=n)
    max_idx = int(np.argmax(counts)) if len(counts) else -1
    max_count = int(counts[max_idx]) if max_idx >= 0 else 0
    return {
        "hubness_top1_concentration": float(max_count / max(n, 1)),
        "hubness_top1_max_count": max_count,
        "hubness_top1_max_local_index": max_idx if max_idx >= 0 else None,
    }


def _mean_neighbor_rank_shift(sim_x: np.ndarray, sim_v: np.ndarray) -> float:
    n = int(sim_x.shape[0])
    if n <= 1:
        return float("nan")
    sv = np.asarray(sim_v, dtype=np.float64).copy()
    sx = np.asarray(sim_x, dtype=np.float64).copy()
    np.fill_diagonal(sv, -np.inf)
    np.fill_diagonal(sx, -np.inf)
    v_nn = np.argmax(sv, axis=1)
    order_x = np.argsort(-sx, axis=1, kind="mergesort")
    ranks = []
    for i in range(n):
        where = np.where(order_x[i] == v_nn[i])[0]
        if len(where):
            ranks.append(int(where[0]) + 1)
    return float(np.mean(ranks)) if ranks else float("nan")


def _structure_metrics(x: np.ndarray, v: np.ndarray, ids: Sequence[int], seed: int, triplets: int) -> Dict[str, Any]:
    x = _l2(np.asarray(x, dtype=np.float32))
    v = _l2(np.asarray(v, dtype=np.float32))
    valid = _finite_rows(x) & _finite_rows(v)
    x = x[valid]
    v = v[valid]
    ids_valid = [int(r) for r, ok in zip(ids, valid) if bool(ok)]
    n = int(x.shape[0])
    if n < 2:
        return {"structure_count": n}
    sim_x = x @ x.T
    sim_v = v @ v.T
    out: Dict[str, Any] = {
        "structure_count": n,
        "spearman_Xalpha_vs_V": _spearman(_upper_triangle_values(sim_x), _upper_triangle_values(sim_v)),
        "knn_overlap@5": _mean_knn_overlap(sim_x, sim_v, 5),
        "knn_overlap@10": _mean_knn_overlap(sim_x, sim_v, 10),
        "knn_overlap@20": _mean_knn_overlap(sim_x, sim_v, 20),
        "triplet_agreement": _triplet_agreement(sim_x, sim_v, seed=seed, samples=triplets),
        "mean_neighbor_rank_shift": _mean_neighbor_rank_shift(sim_x, sim_v),
    }
    out.update(_hubness_top1_concentration(sim_x))
    max_local = out.get("hubness_top1_max_local_index")
    if max_local is not None and 0 <= int(max_local) < len(ids_valid):
        out["hubness_top1_max_raw_id"] = int(ids_valid[int(max_local)])
    return out


def _fit_rectangular_procrustes(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Fit min ||src @ W - dst||_F with W^T W=I for src_dim>=dst_dim.

    Returns W with shape [src_dim, dst_dim].  This is the one-layer near-isometric
    projector used as the primary A10C projector.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    c = src.T @ dst
    u, _s, vt = np.linalg.svd(c, full_matrices=False)
    w = u @ vt
    return np.asarray(w, dtype=np.float32)


def _fit_row_orthogonal_embedding_basis(v_anchor: np.ndarray, r_anchor: np.ndarray) -> np.ndarray:
    """Fit B: R^768 -> R^4096 with B B^T=I using visible anchors only.

    This directly constructs the ideal LLaMA-coordinate DINO-compatible manifold:
        X_ideal = V @ B
    Because B B^T = I, pairwise dot products in X_ideal equal those in V.
    """
    v_anchor = np.asarray(v_anchor, dtype=np.float64)
    r_anchor = np.asarray(r_anchor, dtype=np.float64)
    c = v_anchor.T @ r_anchor  # [768, 4096]
    u, _s, vt = np.linalg.svd(c, full_matrices=False)  # U [768,768], Vt [768,4096]
    b = u @ vt
    return np.asarray(b, dtype=np.float32)


def _fit_ridge_linear_dual(src: np.ndarray, dst: np.ndarray, alpha: float) -> np.ndarray:
    """Bias-free one-layer ridge map src->dst using dual solve.

    W = X^T (X X^T + alpha I)^-1 Y.  This avoids a 4096x4096 solve.
    """
    x = np.asarray(src, dtype=np.float64)
    y = np.asarray(dst, dtype=np.float64)
    n = int(x.shape[0])
    a = x @ x.T + float(alpha) * np.eye(n, dtype=np.float64)
    try:
        coef = np.linalg.solve(a, y)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(a) @ y
    w = x.T @ coef
    return np.asarray(w, dtype=np.float32)


def _project_linear(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return _l2((np.asarray(x, dtype=np.float64) @ np.asarray(w, dtype=np.float64)).astype(np.float32))


def _orthogonality_error_w(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64)
    gram = w.T @ w
    ident = np.eye(gram.shape[0], dtype=np.float64)
    return float(np.linalg.norm(gram - ident, ord="fro") / max(1, gram.shape[0]))


def _condition_number_w(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64)
    gram = w.T @ w
    try:
        vals = np.linalg.eigvalsh(gram)
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 1e-12]
        if len(vals) == 0:
            return float("nan")
        return float(math.sqrt(float(vals.max() / vals.min())))
    except Exception:
        return float("nan")


def _mean_cos(a: np.ndarray, b: np.ndarray) -> float:
    a = _l2(np.asarray(a, dtype=np.float32))
    b = _l2(np.asarray(b, dtype=np.float32))
    valid = _finite_rows(a) & _finite_rows(b)
    if int(valid.sum()) <= 0:
        return float("nan")
    return float(np.mean(np.sum(a[valid] * b[valid], axis=1)))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    valid = _finite_rows(a) & _finite_rows(b)
    if int(valid.sum()) <= 0:
        return float("nan")
    return float(np.mean((a[valid] - b[valid]) ** 2))


def _slerp_rows(a: np.ndarray, b: np.ndarray, alpha: float, eps: float = 1e-7) -> np.ndarray:
    a0 = _l2(np.asarray(a, dtype=np.float32))
    b0 = _l2(np.asarray(b, dtype=np.float32))
    out = np.full_like(a0, np.nan, dtype=np.float32)
    valid = _finite_rows(a0) & _finite_rows(b0)
    if int(valid.sum()) <= 0:
        return out
    av = a0[valid].astype(np.float64)
    bv = b0[valid].astype(np.float64)
    dot = np.sum(av * bv, axis=1, keepdims=True)
    dot = np.clip(dot, -1.0 + eps, 1.0 - eps)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    # For nearly identical/opposite rows, normalized lerp is more stable.
    near = np.abs(sin_theta[:, 0]) < eps
    res = (np.sin((1.0 - float(alpha)) * theta) / sin_theta) * av + (np.sin(float(alpha) * theta) / sin_theta) * bv
    if np.any(near):
        res[near] = (1.0 - float(alpha)) * av[near] + float(alpha) * bv[near]
    out[valid] = _l2(res.astype(np.float32))
    return out


def _ids_to_indices(ids: Sequence[int], id_to_idx: Mapping[int, int], mat: np.ndarray) -> List[int]:
    return [id_to_idx[int(r)] for r in ids if int(r) in id_to_idx and np.isfinite(mat[id_to_idx[int(r)]]).all()]


def _aggregate_rows(rows: Sequence[Mapping[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k, "") for k in group_keys)].append(r)
    out: List[Dict[str, Any]] = []
    for key, vals in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row: Dict[str, Any] = {k: v for k, v in zip(group_keys, key)}
        row["n"] = len(vals)
        numeric_keys = []
        for r in vals:
            for k, v in r.items():
                if k in group_keys:
                    continue
                fv = _safe_float(v)
                if math.isfinite(fv) and k not in numeric_keys:
                    numeric_keys.append(k)
        for k in numeric_keys:
            xs = [_safe_float(r.get(k)) for r in vals]
            xs = [x for x in xs if math.isfinite(x)]
            if xs:
                row[f"{k}_mean"] = float(np.mean(xs))
                row[f"{k}_std"] = float(np.std(xs))
        out.append(row)
    return out


def _threshold_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"usable_rule": "t2v_rank@5 >= 0.80 and mean_normalized_rank <= 0.08", "collapse_rule": "t2v_rank@5 < 0.60 or mean_normalized_rank > 0.15", "groups": []}
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r.get("projector")), str(r.get("test_scope")))].append(r)
    for (projector, test_scope), vals0 in sorted(groups.items()):
        vals = sorted(vals0, key=lambda r: _safe_float(r.get("alpha"), 0.0))
        usable = []
        collapse = []
        for r in vals:
            a = _safe_float(r.get("alpha"))
            t5 = _safe_float(r.get("t2v_rank@5"))
            mnr = _safe_float(r.get("mean_normalized_rank"))
            if math.isfinite(a) and math.isfinite(t5) and math.isfinite(mnr):
                if t5 >= 0.80 and mnr <= 0.08:
                    usable.append(a)
                if t5 < 0.60 or mnr > 0.15:
                    collapse.append(a)
        report["groups"].append({
            "projector": projector,
            "test_scope": test_scope,
            "alpha_usable_max": max(usable) if usable else None,
            "alpha_collapse_start": min(collapse) if collapse else None,
        })
    return report


def _maybe_write_plots(analysis_root: Path, aggregate_rows: Sequence[Mapping[str, Any]]) -> List[str]:
    written: List[str] = []
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return written
    plot_root = analysis_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    # Plot only novel_val for compactness.
    rows = [r for r in aggregate_rows if str(r.get("test_scope")) == "novel_val"]
    for projector in sorted(set(str(r.get("projector")) for r in rows)):
        rr = sorted([r for r in rows if str(r.get("projector")) == projector], key=lambda r: _safe_float(r.get("alpha")))
        if not rr:
            continue
        x = [_safe_float(r.get("alpha")) for r in rr]
        y = [_safe_float(r.get("t2v_rank@5_mean")) for r in rr]
        if all(math.isfinite(v) for v in x + y):
            plt.figure()
            plt.plot(x, y, marker="o")
            plt.xlabel("alpha")
            plt.ylabel("novel_val full_available t2v@5")
            plt.title(f"A10C {projector}: alpha vs novel t2v@5")
            p = plot_root / f"alpha_vs_novel_t2v5_{projector}.png"
            plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
            written.append(str(p))
        y2 = [_safe_float(r.get("mean_normalized_rank_mean")) for r in rr]
        if all(math.isfinite(v) for v in x + y2):
            plt.figure()
            plt.plot(x, y2, marker="o")
            plt.xlabel("alpha")
            plt.ylabel("novel_val mean normalized rank")
            plt.title(f"A10C {projector}: alpha vs mean rank")
            p = plot_root / f"alpha_vs_mean_rank_{projector}.png"
            plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
            written.append(str(p))
        for metric, stem in [("knn_overlap@10_mean", "knn10_vs_novel_t2v5"), ("spearman_Xalpha_vs_V_mean", "spearman_vs_novel_t2v5"), ("hubness_top1_concentration_mean", "hubness_vs_novel_t2v5")]:
            xx = [_safe_float(r.get(metric)) for r in rr]
            yy = [_safe_float(r.get("t2v_rank@5_mean")) for r in rr]
            if all(math.isfinite(v) for v in xx + yy):
                plt.figure()
                plt.scatter(xx, yy)
                plt.xlabel(metric.replace("_mean", ""))
                plt.ylabel("novel_val full_available t2v@5")
                plt.title(f"A10C {projector}: {metric} vs t2v@5")
                p = plot_root / f"{stem}_{projector}.png"
                plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
                written.append(str(p))
    return written


def _build_takeover(output_root: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# A10C LLaMA4096 Linear-Isometric Distortion Calibration TAKEOVER",
        "",
        "## Status",
        f"- overall_status: `{result.get('status')}`",
        f"- output_root: `{result.get('output_root')}`",
        f"- analysis_root: `{result.get('analysis_root')}`",
        "",
        "## Scope",
        "- Read-only class-prototype audit; no formal training/inference pipeline files are mutated.",
        "- Text source: real `llama_hidden_mean` raw feature, expected dimension 4096.",
        "- Control variable: `X_alpha = slerp(X_ideal, R_llama, alpha)` only.",
        "- Projectors: one-layer bias-free linear maps only (`orthogonal_linear`, `ridge_linear`).",
        "- No MLP, no nonlinear activation, no LayerNorm/Dropout, no row-level eval.",
        "",
        "## Required artifacts",
    ]
    for rel in [
        "analysis/a10c_llama4096_linear_isometric_summary.csv",
        "analysis/a10c_llama4096_alpha_aggregate.csv",
        "analysis/a10c_llama4096_threshold_report.json",
        "analysis/A10C_run_result.json",
    ]:
        p = output_root / rel
        lines.append(f"- `{p}`: {'FOUND' if p.is_file() else 'MISSING'}")
    lines += [
        "",
        "## Interpretation checklist",
        "- `alpha=0` is the DINO-compatible ideal endpoint in LLaMA4096 coordinates.",
        "- `alpha=1` is the real LLaMA hidden-mean endpoint.",
        "- If alpha=0 is high and alpha=1 is low, the failure is consistent with real text-side manifold distortion.",
        "- The collapse interval should be read from `a10c_llama4096_threshold_report.json` and the alpha aggregate CSV.",
    ]
    (output_root / "TAKEOVER_A10C_LLAMA4096_LINEAR_ISOMETRIC.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def _fit_rectangular_procrustes_gpu(src: np.ndarray, dst: np.ndarray, device) -> np.ndarray:
    torch = _import_torch()
    if torch is None or device is None:
        return _fit_rectangular_procrustes(src, dst)
    with torch.no_grad():
        xs = _np_to_torch(src, device)
        ys = _np_to_torch(dst, device)
        c = xs.transpose(0, 1).matmul(ys)
        u, _s, vh = torch.linalg.svd(c, full_matrices=False)
        w = u.matmul(vh)
        return w.detach().cpu().numpy().astype(np.float32)


def _fit_row_orthogonal_embedding_basis_gpu(v_anchor: np.ndarray, r_anchor: np.ndarray, device) -> np.ndarray:
    torch = _import_torch()
    if torch is None or device is None:
        return _fit_row_orthogonal_embedding_basis(v_anchor, r_anchor)
    with torch.no_grad():
        v = _np_to_torch(v_anchor, device)
        r = _np_to_torch(r_anchor, device)
        c = v.transpose(0, 1).matmul(r)  # [768,4096]
        u, _s, vh = torch.linalg.svd(c, full_matrices=False)
        b = u.matmul(vh)
        return b.detach().cpu().numpy().astype(np.float32)


def _fit_ridge_linear_dual_gpu(src: np.ndarray, dst: np.ndarray, alpha: float, device) -> np.ndarray:
    torch = _import_torch()
    if torch is None or device is None:
        return _fit_ridge_linear_dual(src, dst, alpha)
    with torch.no_grad():
        x = _np_to_torch(src, device)
        y = _np_to_torch(dst, device)
        n = int(x.shape[0])
        a = x.matmul(x.transpose(0, 1)) + float(alpha) * torch.eye(n, device=device, dtype=x.dtype)
        try:
            coef = torch.linalg.solve(a, y)
        except Exception:
            coef = torch.linalg.pinv(a).matmul(y)
        w = x.transpose(0, 1).matmul(coef)
        return w.detach().cpu().numpy().astype(np.float32)


def _slerp_rows_gpu(a: np.ndarray, b: np.ndarray, alpha: float, device, eps: float = 1e-7) -> np.ndarray:
    torch = _import_torch()
    if torch is None or device is None:
        return _slerp_rows(a, b, alpha, eps=eps)
    with torch.no_grad():
        aa = _torch_l2(_np_to_torch(a, device))
        bb = _torch_l2(_np_to_torch(b, device))
        valid = torch.isfinite(aa).all(dim=1) & torch.isfinite(bb).all(dim=1)
        out = torch.full_like(aa, float('nan'))
        if int(valid.sum().item()) <= 0:
            return out.detach().cpu().numpy().astype(np.float32)
        av = aa[valid]
        bv = bb[valid]
        dot = torch.clamp((av * bv).sum(dim=1, keepdim=True), -1.0 + eps, 1.0 - eps)
        theta = torch.arccos(dot)
        sin_theta = torch.sin(theta)
        near = torch.abs(sin_theta[:, 0]) < eps
        res = (torch.sin((1.0 - float(alpha)) * theta) / sin_theta) * av + (torch.sin(float(alpha) * theta) / sin_theta) * bv
        if bool(near.any().item()):
            res[near] = (1.0 - float(alpha)) * av[near] + float(alpha) * bv[near]
        out[valid] = _torch_l2(res)
        return out.detach().cpu().numpy().astype(np.float32)


def _project_linear_gpu(x: np.ndarray, w: np.ndarray, device) -> np.ndarray:
    torch = _import_torch()
    if torch is None or device is None:
        return _project_linear(x, w)
    with torch.no_grad():
        xx = _np_to_torch(x, device)
        ww = _np_to_torch(w, device)
        out = _torch_l2(xx.matmul(ww))
        return out.detach().cpu().numpy().astype(np.float32)


def _evaluate_class_retrieval_gpu(projected: np.ndarray, visual: np.ndarray, ids: Sequence[int], eval_ids: Sequence[int], candidate_ids: Sequence[int], device) -> Dict[str, Any]:
    torch = _import_torch()
    if torch is None or device is None:
        # Use the existing A10 helper-equivalent path only when torch is unavailable.
        # The caller normally passes the original helper directly, so this fallback is rarely used.
        return {}
    id_to_idx = {int(r): i for i, r in enumerate(ids)}
    valid_visual = np.isfinite(visual).all(axis=1)
    cand_idx = [id_to_idx[int(r)] for r in candidate_ids if int(r) in id_to_idx and bool(valid_visual[id_to_idx[int(r)]])]
    eval_idx = [id_to_idx[int(r)] for r in eval_ids if int(r) in id_to_idx and bool(valid_visual[id_to_idx[int(r)]]) and int(r) in set(candidate_ids)]
    if not cand_idx or not eval_idx:
        return {"eval_count": 0, "candidate_count": len(cand_idx)}
    cand_pos = {int(ids[i]): j for j, i in enumerate(cand_idx)}
    target_cols = [cand_pos.get(int(ids[i]), None) for i in eval_idx]
    keep = [k for k, c in enumerate(target_cols) if c is not None]
    if not keep:
        return {"eval_count": 0, "candidate_count": len(cand_idx)}
    eval_idx = [eval_idx[k] for k in keep]
    target_cols = [int(target_cols[k]) for k in keep]
    with torch.no_grad():
        pt_all = _torch_l2(torch.nan_to_num(_np_to_torch(projected, device), nan=0.0))
        vv_all = _torch_l2(torch.nan_to_num(_np_to_torch(visual, device), nan=0.0))
        pt_eval = pt_all[torch.as_tensor(eval_idx, device=device, dtype=torch.long)]
        vv_eval = vv_all[torch.as_tensor(eval_idx, device=device, dtype=torch.long)]
        pt_cand = pt_all[torch.as_tensor(cand_idx, device=device, dtype=torch.long)]
        vv_cand = vv_all[torch.as_tensor(cand_idx, device=device, dtype=torch.long)]
        target_cols_t = torch.as_tensor(target_cols, device=device, dtype=torch.long)
        # t2v: query projected text of target against candidate visual prototypes.
        s1 = pt_eval.matmul(vv_cand.transpose(0, 1))
        target_s1 = s1.gather(1, target_cols_t[:, None])
        ranks1 = (s1 > target_s1).sum(dim=1) + 1
        # v2t: query visual target against candidate projected text prototypes.
        s2 = vv_eval.matmul(pt_cand.transpose(0, 1))
        target_s2 = s2.gather(1, target_cols_t[:, None])
        ranks2 = (s2 > target_s2).sum(dim=1) + 1
        rec = 1.0 - (pt_eval * vv_eval).sum(dim=1)
        r1 = ranks1.detach().cpu().numpy().astype(np.int64).tolist()
        r2 = ranks2.detach().cpu().numpy().astype(np.int64).tolist()
        recovery_errors = rec.detach().cpu().numpy().astype(np.float64).tolist()
    return {
        "eval_count": int(len(r1)),
        "candidate_count": int(len(cand_idx)),
        **_summary_ranks_local(r1, prefix="t2v_"),
        **_summary_ranks_local(r2, prefix="v2t_"),
        "mean_cosine_recovery_error": float(np.mean(recovery_errors)) if recovery_errors else float('nan'),
        "median_cosine_recovery_error": float(np.median(recovery_errors)) if recovery_errors else float('nan'),
    }


def _summary_ranks_local(ranks: Sequence[int], prefix: str) -> Dict[str, Any]:
    rs = [int(r) for r in ranks if int(r) > 0]
    if not rs:
        return {f"{prefix}rank@1": 0.0, f"{prefix}rank@5": 0.0, f"{prefix}mean_rank": float('nan'), f"{prefix}median_rank": float('nan')}
    arr = np.asarray(rs, dtype=np.float64)
    return {
        f"{prefix}rank@1": float(np.mean(arr <= 1)),
        f"{prefix}rank@5": float(np.mean(arr <= 5)),
        f"{prefix}mean_rank": float(np.mean(arr)),
        f"{prefix}median_rank": float(np.median(arr)),
    }


def _structure_metrics_gpu(x: np.ndarray, v: np.ndarray, ids: Sequence[int], seed: int, triplets: int, device) -> Dict[str, Any]:
    torch = _import_torch()
    if torch is None or device is None:
        return _structure_metrics(x, v, ids, seed, triplets)
    x_np = _l2(np.asarray(x, dtype=np.float32))
    v_np = _l2(np.asarray(v, dtype=np.float32))
    valid = _finite_rows(x_np) & _finite_rows(v_np)
    ids_valid = [int(r) for r, ok in zip(ids, valid) if bool(ok)]
    if int(valid.sum()) < 2:
        return {"structure_count": int(valid.sum())}
    x_np = x_np[valid]
    v_np = v_np[valid]
    n = int(x_np.shape[0])
    with torch.no_grad():
        xt = _np_to_torch(x_np, device)
        vt = _np_to_torch(v_np, device)
        sim_x_t = xt.matmul(xt.transpose(0, 1))
        sim_v_t = vt.matmul(vt.transpose(0, 1))
        eye = torch.eye(n, dtype=torch.bool, device=device)
        sim_x_masked = sim_x_t.masked_fill(eye, -float('inf'))
        sim_v_masked = sim_v_t.masked_fill(eye, -float('inf'))
        def knn(k: int) -> float:
            kk = int(max(0, min(k, n - 1)))
            if kk <= 0:
                return float('nan')
            ix = torch.topk(sim_x_masked, kk, dim=1).indices
            iv = torch.topk(sim_v_masked, kk, dim=1).indices
            # n is small enough; exact set overlap on CPU is acceptable.
            ixn = ix.detach().cpu().numpy(); ivn = iv.detach().cpu().numpy()
            return float(np.mean([len(set(a.tolist()).intersection(set(b.tolist()))) / kk for a, b in zip(ixn, ivn)]))
        nn = torch.argmax(sim_x_masked, dim=1)
        counts = torch.bincount(nn, minlength=n)
        max_count_t, max_idx_t = torch.max(counts, dim=0)
        max_count = int(max_count_t.item()); max_idx = int(max_idx_t.item())
        # rank shift: rank of visual NN inside text-neighbor ordering.
        v_nn = torch.argmax(sim_v_masked, dim=1)
        order_x = torch.argsort(sim_x_masked, dim=1, descending=True, stable=True)
        eq = order_x.eq(v_nn[:, None])
        rank_positions = torch.argmax(eq.to(torch.int32), dim=1).to(torch.float32) + 1.0
        mean_shift = float(rank_positions.mean().item())
        # vectorized triplet agreement.
        trip = float('nan')
        if n >= 3 and int(triplets) > 0:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))
            samples = int(triplets)
            i = torch.randint(0, n, (samples,), generator=gen, device=device)
            j = torch.randint(0, n, (samples,), generator=gen, device=device)
            k = torch.randint(0, n, (samples,), generator=gen, device=device)
            ok = (i != j) & (i != k) & (j != k)
            i, j, k = i[ok], j[ok], k[ok]
            if int(i.numel()) > 0:
                dx = sim_x_t[i, j] - sim_x_t[i, k]
                dv = sim_v_t[i, j] - sim_v_t[i, k]
                ok2 = torch.isfinite(dx) & torch.isfinite(dv) & (dx != 0) & (dv != 0)
                if int(ok2.sum().item()) > 0:
                    trip = float(((dx[ok2] > 0) == (dv[ok2] > 0)).to(torch.float32).mean().item())
        # Spearman still uses CPU ranks over upper triangles for exact parity with original output.
        sim_x = sim_x_t.detach().cpu().numpy()
        sim_v = sim_v_t.detach().cpu().numpy()
    out: Dict[str, Any] = {
        "structure_count": n,
        "spearman_Xalpha_vs_V": _spearman(_upper_triangle_values(sim_x), _upper_triangle_values(sim_v)),
        "knn_overlap@5": knn(5),
        "knn_overlap@10": knn(10),
        "knn_overlap@20": knn(20),
        "triplet_agreement": trip,
        "mean_neighbor_rank_shift": mean_shift,
        "hubness_top1_concentration": float(max_count / max(n, 1)),
        "hubness_top1_max_count": max_count,
        "hubness_top1_max_local_index": max_idx,
    }
    if 0 <= max_idx < len(ids_valid):
        out["hubness_top1_max_raw_id"] = int(ids_valid[max_idx])
    return out


def parse_args() -> argparse.Namespace:
    repo = _repo_default()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo_root", default=str(repo))
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--run_root", default="")
    p.add_argument("--output_root", default="")
    p.add_argument("--visible_csv", default="")
    p.add_argument("--per_class_join", default="")
    p.add_argument("--official_split_json", default="")
    p.add_argument("--train_dataset_name", default="lvvis_train_base")
    p.add_argument("--val_dataset_name", default="lvvis_val")
    p.add_argument("--train_annotation_json", default="")
    p.add_argument("--val_annotation_json", default="")
    p.add_argument("--visual_only_root", default=DEFAULT_VISUAL_ONLY_ROOT)
    p.add_argument("--text_variant", default="llama_hidden_mean")
    p.add_argument("--text_dim", type=int, default=4096)
    p.add_argument("--alphas", default="0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70,0.80,0.90,1.00")
    p.add_argument("--projectors", default="orthogonal_linear,ridge_linear")
    p.add_argument("--ridge_alpha", type=float, default=0.01)
    p.add_argument("--anchor_ratios", default="1.0")
    p.add_argument("--seeds", default="0")
    p.add_argument("--anchor_calib_fraction", type=float, default=0.1)
    p.add_argument("--test_scopes", default="novel_val,val_base_all")
    p.add_argument("--candidate_scope", default="full_available")
    p.add_argument("--triplet_samples", type=int, default=20000)
    p.add_argument("--max_rows", type=int, default=0, help="Max rows/carriers loaded per dataset for prototype construction; 0 means full.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--continue_on_error", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--device", default="auto", help="auto/cuda/cpu. GPU is used for SVD/solve/slerp/projection/retrieval/structure metrics when available.")
    p.add_argument("--progress", action="store_true", help="print stage progress lines before final JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve() if args.run_root else _run_root_default(repo_root)
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else _output_root_default(repo_root)
    analysis_root = output_root / "analysis"
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)
    device, device_name = _resolve_device(str(args.device))
    if args.smoke:
        args.alphas = "0,0.5,1.0"
        args.seeds = "0"
        args.test_scopes = "novel_val"
        args.triplet_samples = min(int(args.triplet_samples), 2000)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    result: Dict[str, Any] = {
        "status": "PASS",
        "repo_root": str(repo_root),
        "asset_root": str(asset_root),
        "run_root": str(run_root),
        "output_root": str(output_root),
        "analysis_root": str(analysis_root),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device_name,
        "gpu_accelerated": bool(device is not None and str(device_name).startswith("cuda")),
    }
    try:
        a10 = _load_a10(repo_root)
        a10b = _load_a10b(repo_root)
        a8 = a10._load_a8_helper(repo_root)
        base_ids, novel_ids, official_names = a10b._load_official_split(repo_root, args.official_split_json or None)
        visible_csv = Path(args.visible_csv).expanduser().resolve() if args.visible_csv else a10._find_visible_csv(repo_root, run_root)
        if not visible_csv or not Path(visible_csv).is_file():
            raise RuntimeError("visible525 csv not found; pass --visible_csv")
        visible_ids = set(int(x) for x in a10._load_visible_ids(a8, Path(visible_csv)))
        per_class_join = Path(args.per_class_join).expanduser().resolve() if args.per_class_join else a10._find_per_class_join(repo_root, run_root)
        per_class_meta = a10._load_per_class_meta(per_class_join)
        train_ann = Path(args.train_annotation_json).expanduser().resolve() if args.train_annotation_json else repo_root / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "train_instances.json"
        val_ann = Path(args.val_annotation_json).expanduser().resolve() if args.val_annotation_json else repo_root / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "val_instances.json"
        gtceil = a8._load_gtceil(repo_root)
        train_rows, train_carrier, _train_meta = a8._rows_and_carriers(gtceil, asset_root=asset_root, dataset_name=args.train_dataset_name, ann=train_ann, max_rows=int(args.max_rows))
        val_rows, val_carrier, _val_meta = a8._rows_and_carriers(gtceil, asset_root=asset_root, dataset_name=args.val_dataset_name, ann=val_ann, max_rows=int(args.max_rows))
        train_proto, train_counts = a8._visual_prototypes(train_rows, train_carrier)
        val_proto, _val_counts = a8._visual_prototypes(val_rows, val_carrier)
        # Load the real LLaMA hidden-mean bank only.  No CLIP branch is mixed into this A10C.
        real_banks, skipped = a10b._load_real_banks_available(
            a10,
            a8,
            asset_root,
            args.train_dataset_name,
            Path(args.visual_only_root).expanduser().resolve(),
            Path("/nonexistent/a10c_no_direct_concept_needed").resolve(),
            [str(args.text_variant)],
        )
        if str(args.text_variant) not in real_banks:
            raise RuntimeError(f"requested text_variant={args.text_variant} unavailable; skipped={skipped}")
        text_ids, text_mat, text_names, text_meta = real_banks[str(args.text_variant)]
        if int(text_mat.shape[1]) != int(args.text_dim):
            raise RuntimeError(f"text_dim mismatch: expected {args.text_dim}, got {text_mat.shape[1]} for {args.text_variant}")
        text_id_set = set(int(x) for x in text_ids)
        universe = sorted((text_id_set & (base_ids | novel_ids)) & (set(train_proto.keys()) | set(val_proto.keys())))
        id_to_idx = {int(r): i for i, r in enumerate(universe)}
        real_mat = _l2(np.asarray(a8._submatrix_for_ids(text_ids, text_mat, universe), dtype=np.float32))
        visual_train_mat, _train_valid = a10b._matrix_from_proto(train_proto, universe)
        visual_val_mat, _val_valid = a10b._matrix_from_proto(val_proto, universe)
        names = {**official_names, **{int(k): str(v) for k, v in text_names.items()}}
        anchor_pool = sorted(visible_ids & base_ids & set(train_proto.keys()) & text_id_set)
        if len(anchor_pool) < 10:
            raise RuntimeError(f"too few anchor_pool classes: {len(anchor_pool)}")
        target_defs: Dict[str, Dict[str, Any]] = {
            "val_base_all": {"target_ids": sorted(base_ids & set(val_proto.keys()) & text_id_set), "eval_visual": "val_proto", "source_policy": "val_prefer"},
            "novel_val": {"target_ids": sorted(novel_ids & set(val_proto.keys()) & text_id_set), "eval_visual": "val_proto", "source_policy": "val_prefer"},
            "val_base_outside_525": {"target_ids": sorted((base_ids - visible_ids) & set(val_proto.keys()) & text_id_set), "eval_visual": "val_proto", "source_policy": "val_prefer"},
            "train_base_outside_525": {"target_ids": sorted((base_ids - visible_ids) & set(train_proto.keys()) & text_id_set), "eval_visual": "train_proto", "source_policy": "train_prefer"},
        }
        test_scopes = [x.strip() for x in str(args.test_scopes).split(",") if x.strip()]
        projectors = [x.strip() for x in str(args.projectors).split(",") if x.strip()]
        allowed_projectors = {"orthogonal_linear", "ridge_linear"}
        bad = [p for p in projectors if p not in allowed_projectors]
        if bad:
            raise ValueError(f"unsupported projectors for A10C linear-only audit: {bad}")
        alphas = [float(x) for x in str(args.alphas).split(",") if x.strip()]
        seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
        anchor_ratios = [float(x) for x in str(args.anchor_ratios).split(",") if x.strip()]
        availability_rows: List[Dict[str, Any]] = []
        for ts in test_scopes:
            td = target_defs.get(ts)
            if not td:
                continue
            availability_rows.append({
                "test_scope": ts,
                "eval_visual": td["eval_visual"],
                "evaluable_class_count": len(td["target_ids"]),
                "anchor_pool_count": len(anchor_pool),
            })
        _write_csv(analysis_root / "a10c_llama4096_availability_inventory.csv", availability_rows)
        summary_rows: List[Dict[str, Any]] = []
        selfcheck_rows: List[Dict[str, Any]] = []
        for seed in seeds:
            for ratio in anchor_ratios:
                if args.progress:
                    print(f"[A10C] seed={seed} anchor_ratio={ratio} start", flush=True)
                fit_ids, calib_ids, split_meta = a10b._sample_anchors(a10, anchor_pool, train_counts, per_class_meta, ratio, float(args.anchor_calib_fraction), seed)
                fit_idx = _ids_to_indices(fit_ids, id_to_idx, visual_train_mat)
                fit_idx = [i for i in fit_idx if np.isfinite(real_mat[i]).all()]
                calib_idx = _ids_to_indices(calib_ids, id_to_idx, visual_train_mat)
                if len(fit_idx) < 10:
                    raise RuntimeError(f"too few fit anchors after filtering: {len(fit_idx)}")
                # B maps DINO768 -> LLaMA4096. It is fitted only from selected visible train anchors.
                if args.progress:
                    print(f"[A10C] seed={seed} ratio={ratio} fit DINO->LLaMA4096 ideal basis on {len(fit_idx)} anchors", flush=True)
                basis = _fit_row_orthogonal_embedding_basis_gpu(visual_train_mat[fit_idx], real_mat[fit_idx], device)
                basis_gram = basis @ basis.T
                basis_error = float(np.linalg.norm(basis_gram - np.eye(basis_gram.shape[0]), ord="fro") / basis_gram.shape[0])
                selfcheck_rows.append({
                    "seed": seed,
                    "anchor_ratio": ratio,
                    "anchor_train_count": len(fit_idx),
                    "basis_shape": str(tuple(basis.shape)),
                    "basis_row_orthogonality_error": basis_error,
                    "calib_anchor_count": len(calib_idx),
                })

                # Build each source visual matrix once. novel_val and val_base_all share
                # (source_policy=val_prefer, eval_visual=val_proto), so projector fitting is
                # cached and not repeated across those scopes.
                source_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
                scope_contexts: List[Dict[str, Any]] = []
                for ts in test_scopes:
                    if args.progress:
                        print(f"[A10C] seed={seed} ratio={ratio} scope={ts} prepare scope", flush=True)
                    td = target_defs.get(ts)
                    if not td:
                        continue
                    eval_visual = str(td["eval_visual"])
                    source_policy = str(td["source_policy"])
                    source_key = (source_policy, eval_visual)
                    eval_mat = visual_train_mat if eval_visual == "train_proto" else visual_val_mat
                    if source_key not in source_cache:
                        source_mat, source_valid = a10b._source_visual_matrix(universe, train_proto, val_proto, source_policy, set(anchor_pool))
                        source_mat = np.asarray(source_mat, dtype=np.float32)
                        x_ideal = np.full((len(universe), int(args.text_dim)), np.nan, dtype=np.float32)
                        valid_source_rows = source_valid & _finite_rows(source_mat)
                        if args.progress:
                            print(f"[A10C] seed={seed} ratio={ratio} source_key={source_key} construct X_ideal valid={int(valid_source_rows.sum())}", flush=True)
                        # GPU matmul for V@basis while preserving exact semantics.
                        torch = _import_torch()
                        if torch is not None and device is not None:
                            with torch.no_grad():
                                src_t = _np_to_torch(source_mat[valid_source_rows], device)
                                basis_t = _np_to_torch(basis, device)
                                xi = _torch_l2(src_t.matmul(basis_t)).detach().cpu().numpy().astype(np.float32)
                            x_ideal[valid_source_rows] = xi
                        else:
                            x_ideal[valid_source_rows] = _l2((source_mat[valid_source_rows].astype(np.float64) @ basis.astype(np.float64)).astype(np.float32))
                        valid_for_check = np.where(valid_source_rows & _finite_rows(real_mat))[0]
                        if len(valid_for_check) >= 2:
                            sim_v = source_mat[valid_for_check] @ source_mat[valid_for_check].T
                            sim_x0 = x_ideal[valid_for_check] @ x_ideal[valid_for_check].T
                            ideal_pairwise_mae = float(np.mean(np.abs(sim_v - sim_x0)))
                        else:
                            ideal_pairwise_mae = float("nan")
                        source_cache[source_key] = {
                            "source_mat": source_mat,
                            "source_valid": source_valid,
                            "x_ideal": x_ideal,
                            "ideal_pairwise_mae": ideal_pairwise_mae,
                            "xalpha_cache": {},
                            "projection_cache": {},
                        }
                    target_ids = [int(x) for x in td["target_ids"] if int(x) in id_to_idx]
                    cand_ids = a10b._candidate_ids(str(args.candidate_scope), target_ids, anchor_pool, base_ids, novel_ids, train_proto, val_proto, eval_visual, text_id_set)
                    cand_ids = [int(x) for x in cand_ids if int(x) in id_to_idx and np.isfinite(eval_mat[id_to_idx[int(x)]]).all() and np.isfinite(real_mat[id_to_idx[int(x)]]).all()]
                    source_mat_for_scope = source_cache[source_key]["source_mat"]
                    structure_idx = [id_to_idx[int(r)] for r in cand_ids if int(r) in id_to_idx and np.isfinite(source_mat_for_scope[id_to_idx[int(r)]]).all()]
                    scope_contexts.append({
                        "test_scope": ts,
                        "td": td,
                        "source_key": source_key,
                        "eval_visual": eval_visual,
                        "eval_mat": eval_mat,
                        "target_ids": target_ids,
                        "cand_ids": cand_ids,
                        "structure_idx": structure_idx,
                    })

                # Fit/project once per source_key x alpha x projector. This preserves
                # semantics but avoids repeating SVD/solve for novel_val and val_base_all.
                for source_key, scache in source_cache.items():
                    x_ideal = scache["x_ideal"]
                    for alpha in alphas:
                        if args.progress:
                            print(f"[A10C] seed={seed} ratio={ratio} source_key={source_key} alpha={alpha:.3f} slerp", flush=True)
                        x_alpha = _slerp_rows_gpu(x_ideal, real_mat, float(alpha), device)
                        scache["xalpha_cache"][float(alpha)] = x_alpha
                        train_idx = [i for i in fit_idx if np.isfinite(x_alpha[i]).all() and np.isfinite(visual_train_mat[i]).all()]
                        if len(train_idx) < 10:
                            raise RuntimeError(f"too few train_idx for alpha={alpha}: {len(train_idx)}")
                        for projector in projectors:
                            if args.progress:
                                print(f"[A10C] seed={seed} ratio={ratio} source_key={source_key} alpha={alpha:.3f} projector={projector} fit/project", flush=True)
                            if projector == "orthogonal_linear":
                                w = _fit_rectangular_procrustes_gpu(x_alpha[train_idx], visual_train_mat[train_idx], device)
                                selected_ridge_alpha = ""
                            elif projector == "ridge_linear":
                                w = _fit_ridge_linear_dual_gpu(x_alpha[train_idx], visual_train_mat[train_idx], float(args.ridge_alpha), device)
                                selected_ridge_alpha = float(args.ridge_alpha)
                            else:
                                raise ValueError(projector)
                            projected = _project_linear_gpu(x_alpha, w, device)
                            scache["projection_cache"][(float(alpha), projector)] = {
                                "w": w,
                                "projected": projected,
                                "train_idx": train_idx,
                                "selected_ridge_alpha": selected_ridge_alpha,
                                "projector_orthogonality_error": _orthogonality_error_w(w),
                                "linear_condition_number": _condition_number_w(w),
                                "anchor_reconstruction_cosine": _mean_cos(projected[train_idx], visual_train_mat[train_idx]),
                                "anchor_mse": _mse(projected[train_idx], visual_train_mat[train_idx]),
                            }

                # Evaluate every requested target scope using cached projections.
                for ctx in scope_contexts:
                    ts = str(ctx["test_scope"])
                    source_key = ctx["source_key"]
                    scache = source_cache[source_key]
                    eval_visual = str(ctx["eval_visual"])
                    eval_mat = ctx["eval_mat"]
                    target_ids = ctx["target_ids"]
                    cand_ids = ctx["cand_ids"]
                    structure_idx = ctx["structure_idx"]
                    source_mat = scache["source_mat"]
                    for alpha in alphas:
                        x_alpha = scache["xalpha_cache"][float(alpha)]
                        for projector in projectors:
                            if args.progress:
                                print(f"[A10C] seed={seed} ratio={ratio} scope={ts} alpha={alpha:.3f} projector={projector} evaluate", flush=True)
                            pcache = scache["projection_cache"][(float(alpha), projector)]
                            projected = pcache["projected"]
                            w = pcache["w"]
                            met = _evaluate_class_retrieval_gpu(projected, eval_mat, universe, target_ids, cand_ids, device) if device is not None else a10._evaluate_class_retrieval(projected, eval_mat, universe, target_ids, cand_ids)
                            eval_count = int(met.get("eval_count", 0) or 0)
                            candidate_count = int(met.get("candidate_count", 0) or 0)
                            mean_norm_rank = None
                            mr = _safe_float(met.get("t2v_mean_rank"))
                            if math.isfinite(mr) and candidate_count > 1:
                                mean_norm_rank = float((mr - 1.0) / max(candidate_count - 1, 1))
                            struct_met = _structure_metrics_gpu(x_alpha[structure_idx], source_mat[structure_idx], [universe[i] for i in structure_idx], seed=seed + int(round(float(alpha) * 10000)), triplets=int(args.triplet_samples), device=device) if structure_idx else {"structure_count": 0}
                            target_idx = _ids_to_indices(target_ids, id_to_idx, eval_mat)
                            target_cos = _mean_cos(projected[target_idx], eval_mat[target_idx]) if target_idx else float("nan")
                            target_mse = _mse(projected[target_idx], eval_mat[target_idx]) if target_idx else float("nan")
                            row: Dict[str, Any] = {
                                "seed": seed,
                                "anchor_ratio": ratio,
                                "anchor_train_count": len(pcache["train_idx"]),
                                "test_scope": ts,
                                "eval_visual": eval_visual,
                                "candidate_scope": str(args.candidate_scope),
                                "text_variant": str(args.text_variant),
                                "text_dim": int(args.text_dim),
                                "alpha": float(alpha),
                                "projector": projector,
                                "ridge_alpha": pcache["selected_ridge_alpha"],
                                "target_class_count": len(target_ids),
                                "candidate_count_requested": len(cand_ids),
                                "ideal_pairwise_mae_vs_visual": scache["ideal_pairwise_mae"],
                                "basis_row_orthogonality_error": basis_error,
                                "projector_orthogonality_error": pcache["projector_orthogonality_error"],
                                "linear_condition_number": pcache["linear_condition_number"],
                                "anchor_reconstruction_cosine": pcache["anchor_reconstruction_cosine"],
                                "anchor_mse": pcache["anchor_mse"],
                                "heldout_reconstruction_cosine": target_cos,
                                "heldout_mse": target_mse,
                                "mean_normalized_rank": mean_norm_rank if mean_norm_rank is not None else "",
                                **struct_met,
                                **met,
                            }
                            t5 = _safe_float(row.get("t2v_rank@5"))
                            mnr = _safe_float(row.get("mean_normalized_rank"))
                            if math.isfinite(t5) and math.isfinite(mnr):
                                if t5 >= 0.80 and mnr <= 0.08:
                                    row["collapse_zone"] = "usable"
                                elif t5 < 0.60 or mnr > 0.15:
                                    row["collapse_zone"] = "collapse"
                                else:
                                    row["collapse_zone"] = "decay"
                            summary_rows.append(row)
        _write_csv(analysis_root / "a10c_llama4096_linear_isometric_summary.csv", summary_rows)
        _write_csv(analysis_root / "a10c_llama4096_selfcheck.csv", selfcheck_rows)
        aggregate = _aggregate_rows(summary_rows, ["alpha", "projector", "test_scope", "candidate_scope", "text_variant"])
        _write_csv(analysis_root / "a10c_llama4096_alpha_aggregate.csv", aggregate)
        thresh = _threshold_report(summary_rows)
        _write_json(analysis_root / "a10c_llama4096_threshold_report.json", thresh)
        plot_paths = [] if args.no_plots else _maybe_write_plots(analysis_root, aggregate)
        final_summary = {
            "status": "PASS",
            "repo_root": str(repo_root),
            "asset_root": str(asset_root),
            "run_root": str(run_root),
            "output_root": str(output_root),
            "analysis_root": str(analysis_root),
            "visible_csv": str(visible_csv),
            "per_class_join": str(per_class_join) if per_class_join else "",
            "text_variant": str(args.text_variant),
            "text_dim": int(args.text_dim),
            "device": device_name,
            "gpu_accelerated": bool(device is not None and str(device_name).startswith("cuda")),
            "text_meta": text_meta,
            "official_base_count": len(base_ids),
            "official_novel_count": len(novel_ids),
            "anchor_pool_count": len(anchor_pool),
            "universe_count": len(universe),
            "alphas": alphas,
            "projectors": projectors,
            "test_scopes": test_scopes,
            "candidate_scope": str(args.candidate_scope),
            "seeds": seeds,
            "anchor_ratios": anchor_ratios,
            "summary_rows": len(summary_rows),
            "aggregate_rows": len(aggregate),
            "selfcheck_rows": len(selfcheck_rows),
            "skipped_text_banks": skipped,
            "plots": plot_paths,
            "artifacts": {
                "summary_csv": str(analysis_root / "a10c_llama4096_linear_isometric_summary.csv"),
                "aggregate_csv": str(analysis_root / "a10c_llama4096_alpha_aggregate.csv"),
                "threshold_report_json": str(analysis_root / "a10c_llama4096_threshold_report.json"),
                "selfcheck_csv": str(analysis_root / "a10c_llama4096_selfcheck.csv"),
            },
        }
        result.update(final_summary)
        _write_json(analysis_root / "A10C_run_result.json", final_summary)
        _build_takeover(output_root, result)
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        _write_json(analysis_root / "A10C_run_result.json", result)
        _build_takeover(output_root, result)
        if not args.continue_on_error:
            raise
    result["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(analysis_root / "A10C_final_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
