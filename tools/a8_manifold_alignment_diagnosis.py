#!/usr/bin/env python3
"""A8 manifold-alignment diagnosis for text-bank / trajectory-carrier alignment.

Read-only diagnostic tool. It does not train, mutate checkpoints, or change
existing metrics. It runs four audits:
  1. class-prototype holdout alignment;
  2. anchor-count curve;
  3. projector graph-distortion audit;
  4. row-level GT-margin audit.

The goal is to test whether text and vision class manifolds are actually
low-distortion alignable under finite anchors, and whether trained projectors
preserve or destroy that structure before row-level scoring.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _upper(sim: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    n = int(sim.shape[0])
    iu = np.triu_indices(n, k=1)
    vals = np.asarray(sim[iu], dtype=np.float64)
    if valid is not None:
        vm = np.asarray(valid, dtype=bool)
        vals = vals[vm[iu[0]] & vm[iu[1]]]
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


def _mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def _median(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(np.median(vals)) if vals else float("nan")


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


def _load_npz_first(path: Path) -> Tuple[np.ndarray, str]:
    z = np.load(path)
    keys = list(z.keys())
    if not keys:
        raise RuntimeError(f"empty npz: {path}")
    for key in ("protos", "features", "arr_0", "llama_hidden_mean", "clip_of_llm_mean", "llama_direct_concept_mean"):
        if key in z:
            return np.asarray(z[key]), key
    return np.asarray(z[keys[0]]), str(keys[0])


def _load_lvvis_classes(bank_root: Path) -> Tuple[List[int], Dict[int, str]]:
    path = bank_root / "lvvis_class_names.json"
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
        raise RuntimeError(f"invalid LV-VIS class list: {path}")
    return ids, names


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


def _load_external_text_bank(root: Path, variant: str) -> Tuple[List[int], np.ndarray, Dict[int, str], Dict[str, Any]]:
    ids, names = _load_lvvis_classes(root)
    p = _payload_for_variant(root, variant)
    arr, key = _load_npz_first(p)
    if arr.ndim != 2 or int(arr.shape[0]) != len(ids):
        raise RuntimeError(f"bad text-bank payload shape={arr.shape} ids={len(ids)}")
    arr = _l2(np.asarray(arr, dtype=np.float32))
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    meta = {
        "source": "external_text_bank",
        "variant": variant,
        "root": str(root),
        "payload_path": str(p),
        "payload_array_key": key,
        "payload_sha256": _sha256(p),
        "manifest_path": str(manifest_path) if manifest_path.is_file() else "",
        "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else "",
        "feature_dim": int(arr.shape[1]),
        "class_count": int(len(ids)),
        "profile_id": manifest.get("profile_id"),
        "profile_type": manifest.get("profile_type"),
        "token_feature_alignment": manifest.get("token_feature_alignment"),
        "uses_old_corr_feats": manifest.get("uses_old_corr_feats"),
    }
    return ids, arr, names, meta


def _load_current_clip_text_bank(asset_root: Path, dataset_name: str) -> Tuple[List[int], np.ndarray, Dict[int, str], Dict[str, Any]]:
    from videocutler.ext_stageb_ovvis.eval.g8_bridge import load_text_vocab_with_names  # type: ignore
    raw_ids, _records, mat, class_map = load_text_vocab_with_names(asset_root, dataset_name)
    ids = [int(x) for x in raw_ids]
    names = {int(k): str(v) for k, v in dict(class_map).items()}
    mat = _l2(np.asarray(mat, dtype=np.float32))
    return ids, mat, names, {
        "source": "current_clip_text_bank",
        "variant": "clip_current",
        "feature_dim": int(mat.shape[1]),
        "class_count": int(len(ids)),
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


def _visual_prototypes(rows: Sequence[Mapping[str, Any]], carrier: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    buf: Dict[int, List[np.ndarray]] = {}
    for i, row in enumerate(rows):
        rid = _as_int(row.get("raw_category_id"))
        if rid is None or i >= carrier.shape[0]:
            continue
        vec = np.asarray(carrier[i], dtype=np.float32)
        if np.isfinite(vec).all():
            buf.setdefault(int(rid), []).append(vec)
    proto: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    for rid, vecs in buf.items():
        proto[int(rid)] = _l2(np.mean(np.stack(vecs, axis=0), axis=0, keepdims=True))[0]
        counts[int(rid)] = int(len(vecs))
    return proto, counts


def _matrix_for_ids(proto: Mapping[int, np.ndarray], ids: Sequence[int], dim: int = 768) -> Tuple[np.ndarray, np.ndarray]:
    out = np.full((len(ids), dim), np.nan, dtype=np.float32)
    valid = np.zeros((len(ids),), dtype=bool)
    for i, rid in enumerate(ids):
        if int(rid) in proto:
            v = np.asarray(proto[int(rid)], dtype=np.float32).reshape(-1)
            out[i, : min(dim, v.shape[0])] = v[:dim]
            valid[i] = np.isfinite(out[i]).all()
    return out, valid


def _graph_metrics(a_sim: np.ndarray, b_sim: np.ndarray, ids: Sequence[int], valid: Optional[np.ndarray], k: int, prefix: str = "") -> Dict[str, Any]:
    if valid is None:
        valid = np.isfinite(a_sim).all(axis=1) & np.isfinite(b_sim).all(axis=1)
    valid = np.asarray(valid, dtype=bool)
    jac: List[float] = []
    person_in_a_topk = 0
    person = 773
    for i in range(len(ids)):
        an = _topk(a_sim, ids, i, k)
        bn = _topk(b_sim, ids, i, k)
        if valid[i]:
            jac.append(_jaccard(an, bn))
        if person in an:
            person_in_a_topk += 1
    deg_a = Counter()
    deg_b = Counter()
    for i in range(len(ids)):
        for nb in _topk(a_sim, ids, i, k):
            deg_a[int(nb)] += 1
        for nb in _topk(b_sim, ids, i, k):
            deg_b[int(nb)] += 1
    return {
        prefix + "valid_class_count": int(valid.sum()),
        prefix + "spearman_pairwise": _spearman(_upper(a_sim, valid), _upper(b_sim, valid)),
        prefix + "mean_topk_jaccard": _mean(jac),
        prefix + "a_person_in_topk_rate": float(person_in_a_topk / max(len(ids), 1)),
        prefix + "a_person_indegree@k": int(deg_a.get(person, 0)),
        prefix + "b_person_indegree@k": int(deg_b.get(person, 0)),
        prefix + "a_max_indegree@k": int(max(deg_a.values()) if deg_a else 0),
        prefix + "b_max_indegree@k": int(max(deg_b.values()) if deg_b else 0),
    }


def _fit_linear_map(x: np.ndarray, y: np.ndarray, *, method: str, alpha: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"bad fit shapes x={x.shape} y={y.shape}")
    if method == "least_squares":
        w, *_ = np.linalg.lstsq(x, y, rcond=None)
        return np.asarray(w, dtype=np.float32)
    if method == "ridge":
        # Dual ridge is much faster and stabler for Llama hidden features where D=4096
        # and the anchor count is at most a few hundred.
        if x.shape[1] > x.shape[0]:
            k = x @ x.T
            a = np.linalg.solve(k + float(alpha) * np.eye(x.shape[0], dtype=np.float64), y)
            w = x.T @ a
        else:
            xtx = x.T @ x
            reg = float(alpha) * np.eye(x.shape[1], dtype=np.float64)
            xty = x.T @ y
            w = np.linalg.solve(xtx + reg, xty)
        return np.asarray(w, dtype=np.float32)
    raise ValueError(f"unsupported mapping method: {method}")


def _evaluate_cross_retrieval(projected_text: np.ndarray, visual: np.ndarray, ids: Sequence[int], eval_ids: Sequence[int], candidate_ids: Sequence[int]) -> Dict[str, Any]:
    id_to_idx = {int(r): i for i, r in enumerate(ids)}
    cand_idx = [id_to_idx[int(r)] for r in candidate_ids if int(r) in id_to_idx and np.isfinite(visual[id_to_idx[int(r)]]).all()]
    eval_idx = [id_to_idx[int(r)] for r in eval_ids if int(r) in id_to_idx and np.isfinite(visual[id_to_idx[int(r)]]).all()]
    if not cand_idx or not eval_idx:
        return {"eval_count": 0}
    pt = _l2(projected_text)
    vv = _l2(np.nan_to_num(visual, nan=0.0))
    text_to_vision_ranks: List[int] = []
    vision_to_text_ranks: List[int] = []
    for qi in eval_idx:
        rid = int(ids[qi])
        # text query -> visual candidates
        sims = vv[cand_idx] @ pt[qi]
        order = np.argsort(-sims, kind="mergesort")
        target_pos = [j for j, ci in enumerate(cand_idx) if int(ids[ci]) == rid]
        if target_pos:
            target_j = target_pos[0]
            rank = int(np.where(order == target_j)[0][0]) + 1
            text_to_vision_ranks.append(rank)
        # visual query -> text candidates
        sims2 = pt[cand_idx] @ vv[qi]
        order2 = np.argsort(-sims2, kind="mergesort")
        if target_pos:
            target_j = target_pos[0]
            rank2 = int(np.where(order2 == target_j)[0][0]) + 1
            vision_to_text_ranks.append(rank2)
    out = {
        "eval_count": int(len(eval_idx)),
        "candidate_count": int(len(cand_idx)),
        **_summary_ranks(text_to_vision_ranks, prefix="t2v_"),
        **_summary_ranks(vision_to_text_ranks, prefix="v2t_"),
    }
    return out


class _LocalProjector(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__()
        input_dim = int(cfg.get("input_dim", 512))
        hidden_dim = int(cfg.get("hidden_dim", 1024))
        output_dim = int(cfg.get("output_dim", 768))
        dropout = float(cfg.get("dropout", 0.0))
        use_ln = bool(cfg.get("use_layernorm", True))
        ptype = str(cfg.get("projector_type", "mlp") or "mlp").strip().lower()
        layers: List[nn.Module] = []
        if ptype == "linear":
            layers.append(nn.Linear(input_dim, output_dim))
        elif ptype == "linear_ln":
            layers.append(nn.LayerNorm(input_dim))
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            if use_ln:
                layers.append(nn.LayerNorm(input_dim))
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.ptype = ptype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2.0, dim=-1)


def _infer_config_from_state_dict(sd: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    keys = list(sd.keys())
    linear_keys = [k for k in keys if k.endswith("weight") and getattr(sd[k], "ndim", 0) == 2]
    if len(linear_keys) == 1:
        w = sd[linear_keys[0]]
        return {"input_dim": int(w.shape[1]), "hidden_dim": 0, "output_dim": int(w.shape[0]), "dropout": 0.0, "use_layernorm": False, "projector_type": "linear"}
    # Historical MLP with optional layernorm.
    first_linear = None
    last_linear = None
    for k in linear_keys:
        if first_linear is None:
            first_linear = k
        last_linear = k
    if first_linear and last_linear:
        w0 = sd[first_linear]
        w1 = sd[last_linear]
        has_ln = any(k.endswith("bias") and getattr(sd[k], "ndim", 0) == 1 and int(sd[k].shape[0]) == int(w0.shape[1]) for k in keys)
        return {"input_dim": int(w0.shape[1]), "hidden_dim": int(w0.shape[0]), "output_dim": int(w1.shape[0]), "dropout": 0.0, "use_layernorm": bool(has_ln), "projector_type": "mlp"}
    raise RuntimeError(f"cannot infer projector config from state dict keys={keys[:10]}")


def _load_checkpoint_projector(checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, float, float, Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt.get("text_projector_state_dict")
    if not isinstance(sd, Mapping):
        raise RuntimeError(f"checkpoint missing text_projector_state_dict: {checkpoint_path}")
    cfg = dict(ckpt.get("text_projector_config", {}) or {})
    if not cfg:
        cfg = _infer_config_from_state_dict(sd)
    cfg.setdefault("projector_type", "mlp" if int(cfg.get("hidden_dim", 1024)) > 0 else "linear")
    # Old checkpoints do not record projector_type; infer linear when hidden_dim==0.
    if int(cfg.get("hidden_dim", 1024)) == 0 and str(cfg.get("projector_type", "")).lower() == "mlp":
        cfg["projector_type"] = "linear"
    model = _LocalProjector(cfg).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    theta = float(ckpt.get("theta_T", 0.0))
    # softplus(theta) + 1e-4, matching g8_bridge.
    temperature = float(F.softplus(torch.tensor(theta, dtype=torch.float32)).item() + 1e-4)
    unknown = float(ckpt.get("b_u", 0.0))
    return model, temperature, unknown, dict(ckpt), cfg


def _project_text_matrix(projector: nn.Module, text_matrix: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(text_matrix.shape[0]), int(batch_size)):
            arr = torch.from_numpy(np.asarray(text_matrix[start:start + batch_size], dtype=np.float32)).to(device=device, dtype=torch.float32)
            out = projector(arr)
            outs.append(np.asarray(out.detach().cpu().numpy(), dtype=np.float32))
    return _l2(np.concatenate(outs, axis=0))


def _load_text_bank_for_checkpoint(asset_root: Path, dataset_name: str, ckpt: Mapping[str, Any]) -> Tuple[List[int], np.ndarray, Dict[int, str], Dict[str, Any]]:
    tb = ckpt.get("text_bank", {}) if isinstance(ckpt.get("text_bank", {}), Mapping) else {}
    variant = str(tb.get("variant", "clip_current") or "clip_current")
    if variant == "clip_current":
        ids, mat, names, meta = _load_current_clip_text_bank(asset_root, dataset_name)
        meta.update({"loaded_by_checkpoint_text_bank_loader": False})
        return ids, mat, names, meta
    root = Path(str(tb.get("root", ""))).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"external text bank root not found: {root}")
    ids, mat, names, meta = _load_external_text_bank(root, variant)
    if tb.get("payload_sha256") and meta.get("payload_sha256") != tb.get("payload_sha256"):
        raise RuntimeError(f"payload sha mismatch for {variant}: checkpoint={tb.get('payload_sha256')} current={meta.get('payload_sha256')}")
    if tb.get("manifest_sha256") and meta.get("manifest_sha256") and meta.get("manifest_sha256") != tb.get("manifest_sha256"):
        raise RuntimeError(f"manifest sha mismatch for {variant}: checkpoint={tb.get('manifest_sha256')} current={meta.get('manifest_sha256')}")
    meta.update({
        "loaded_by_checkpoint_text_bank_loader": True,
        "payload_sha256_verified_against_checkpoint": bool(tb.get("payload_sha256")),
        "manifest_sha256_verified_against_checkpoint": bool(tb.get("manifest_sha256")),
    })
    return ids, mat, names, meta


def _discover_checkpoints(run_root: Path, explicit_specs: str = "") -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    if explicit_specs.strip():
        for part in explicit_specs.split(";"):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError("--checkpoint_specs must use NAME=PATH;NAME2=PATH2")
            name, path = part.split("=", 1)
            specs.append({"name": name.strip(), "checkpoint_path": str(Path(path).expanduser().resolve()), "source": "explicit"})
        return specs
    patterns = [
        "outputs/a8_textbank_compare/lvvis_train_base/*/final_summary.json",
        "outputs/a8_textbank_linear_projector_compare/lvvis_train_base/*/final_summary.json",
        "outputs/a8_textbank_linear_projector_smoke/lvvis_train_base/*/final_summary.json",
        "outputs/a8_textbank_compare_smoke/lvvis_train_base/*/final_summary.json",
    ]
    seen = set()
    for pat in patterns:
        for p in sorted(run_root.glob(pat)):
            try:
                obj = _read_json(p)
                ck = str(obj.get("checkpoint", "")).strip()
                if not ck:
                    continue
                ck_path = Path(ck).expanduser().resolve()
                if not ck_path.is_file():
                    continue
                name = str(obj.get("name", p.parent.name))
                key = (name, str(ck_path))
                if key in seen:
                    continue
                seen.add(key)
                tb = obj.get("setup", {}).get("text_bank", {}) if isinstance(obj.get("setup", {}), Mapping) else {}
                pc = obj.get("setup", {}).get("text_projector_config", {}) if isinstance(obj.get("setup", {}), Mapping) else {}
                specs.append({
                    "name": name,
                    "checkpoint_path": str(ck_path),
                    "variant": tb.get("variant"),
                    "feature_dim": tb.get("feature_dim"),
                    "projector_type": pc.get("projector_type", "mlp" if int(pc.get("hidden_dim", 1024) or 1024) > 0 else "linear"),
                    "source": str(p),
                })
            except Exception:
                continue
    return specs


def _load_all_text_banks(asset_root: Path, dataset_name: str, visual_root: Path, direct_root: Path, variants: Sequence[str]) -> Dict[str, Tuple[List[int], np.ndarray, Dict[int, str], Dict[str, Any]]]:
    out: Dict[str, Tuple[List[int], np.ndarray, Dict[int, str], Dict[str, Any]]] = {}
    for v in variants:
        if v == "clip_current":
            out[v] = _load_current_clip_text_bank(asset_root, dataset_name)
        elif v in {"clip_of_llm_mean", "llama_hidden_mean"}:
            out[v] = _load_external_text_bank(visual_root, v)
        elif v == "llama_direct_concept_mean":
            out[v] = _load_external_text_bank(direct_root, v)
        else:
            raise ValueError(f"unsupported variant: {v}")
    return out


def _submatrix_for_ids(ids_all: Sequence[int], mat: np.ndarray, ids: Sequence[int]) -> np.ndarray:
    idx = {int(r): i for i, r in enumerate(ids_all)}
    missing = [int(r) for r in ids if int(r) not in idx]
    if missing:
        raise RuntimeError(f"missing ids in matrix: count={len(missing)} first={missing[:10]}")
    return np.asarray(mat[[idx[int(r)] for r in ids]], dtype=np.float32)


def run_class_proto_alignment(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    ids: List[int] = list(ctx["ids"])
    visual_train = np.asarray(ctx["visual_train_mat"], dtype=np.float32)
    train_valid = np.asarray(ctx["train_valid"], dtype=bool)
    visual_val = np.asarray(ctx["visual_val_mat"], dtype=np.float32)
    val_valid = np.asarray(ctx["val_valid"], dtype=bool)
    text_banks = ctx["text_banks"]
    out_root: Path = ctx["output_root"] / "class_proto_holdout_alignment"
    out_root.mkdir(parents=True, exist_ok=True)
    args = ctx["args"]
    rng = np.random.default_rng(int(args.seed))
    eligible = [i for i, rid in enumerate(ids) if bool(train_valid[i])]
    rng.shuffle(eligible)
    holdout_n = max(1, int(round(len(eligible) * float(args.holdout_fraction))))
    holdout_idx = sorted(eligible[:holdout_n])
    anchor_idx = sorted(eligible[holdout_n:])
    if len(anchor_idx) < 2:
        raise RuntimeError("not enough anchor classes for holdout alignment")
    methods = [x.strip() for x in str(args.mapping_methods).split(",") if x.strip()]
    rows: List[Dict[str, Any]] = []
    for variant, (t_ids, t_mat, _names, meta) in text_banks.items():
        text = _submatrix_for_ids(t_ids, t_mat, ids)
        for method in methods:
            w = _fit_linear_map(text[anchor_idx], visual_train[anchor_idx], method=method, alpha=float(args.ridge_alpha))
            projected = _l2(text @ w)
            for target_name, vis, valid in [("train_proto", visual_train, train_valid), ("val_proto", visual_val, val_valid)]:
                eval_idx = [i for i in holdout_idx if bool(valid[i])]
                eval_ids = [ids[i] for i in eval_idx]
                cand_ids = [ids[i] for i in range(len(ids)) if bool(valid[i])]
                met = _evaluate_cross_retrieval(projected, vis, ids, eval_ids, cand_ids)
                rows.append({
                    "variant": variant,
                    "mapping": method,
                    "target_visual": target_name,
                    "anchor_count": len(anchor_idx),
                    "holdout_count": len(holdout_idx),
                    "eval_class_count": len(eval_ids),
                    "candidate_scope": "all_scoped_with_visual_proto",
                    "ridge_alpha": float(args.ridge_alpha),
                    "feature_dim": meta.get("feature_dim"),
                    **met,
                })
    _write_csv(out_root / "class_proto_alignment_summary.csv", rows)
    payload = {"status": "PASS", "rows": rows, "anchor_count": len(anchor_idx), "holdout_count": len(holdout_idx), "artifacts": {"summary_csv": str(out_root / "class_proto_alignment_summary.csv")}}
    _write_json(out_root / "class_proto_alignment_summary.json", payload)
    return payload


def run_anchor_curve(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    ids: List[int] = list(ctx["ids"])
    visual_train = np.asarray(ctx["visual_train_mat"], dtype=np.float32)
    train_valid = np.asarray(ctx["train_valid"], dtype=bool)
    visual_val = np.asarray(ctx["visual_val_mat"], dtype=np.float32)
    val_valid = np.asarray(ctx["val_valid"], dtype=bool)
    text_banks = ctx["text_banks"]
    out_root: Path = ctx["output_root"] / "anchor_count_curve"
    out_root.mkdir(parents=True, exist_ok=True)
    args = ctx["args"]
    anchor_counts = [int(x) for x in str(args.anchor_counts).split(",") if x.strip()]
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    eligible = [i for i, rid in enumerate(ids) if bool(train_valid[i])]
    rows: List[Dict[str, Any]] = []
    for variant, (t_ids, t_mat, _names, meta) in text_banks.items():
        text = _submatrix_for_ids(t_ids, t_mat, ids)
        for anchor_count in anchor_counts:
            if anchor_count >= len(eligible):
                continue
            for seed in seeds:
                rng = np.random.default_rng(seed)
                order = np.array(eligible, dtype=np.int64)
                rng.shuffle(order)
                anchor_idx = sorted([int(x) for x in order[:anchor_count]])
                holdout_idx = sorted([int(x) for x in order[anchor_count:]])
                try:
                    w = _fit_linear_map(text[anchor_idx], visual_train[anchor_idx], method="ridge", alpha=float(args.ridge_alpha))
                except Exception as exc:
                    rows.append({"variant": variant, "anchor_count": anchor_count, "seed": seed, "status": "FAIL", "error": str(exc)})
                    continue
                projected = _l2(text @ w)
                for target_name, vis, valid in [("train_proto", visual_train, train_valid), ("val_proto", visual_val, val_valid)]:
                    eval_ids = [ids[i] for i in holdout_idx if bool(valid[i])]
                    cand_ids = [ids[i] for i in range(len(ids)) if bool(valid[i])]
                    met = _evaluate_cross_retrieval(projected, vis, ids, eval_ids, cand_ids)
                    rows.append({
                        "status": "PASS",
                        "variant": variant,
                        "mapping": "ridge",
                        "target_visual": target_name,
                        "anchor_count": int(anchor_count),
                        "holdout_count": int(len(holdout_idx)),
                        "seed": int(seed),
                        "ridge_alpha": float(args.ridge_alpha),
                        "feature_dim": meta.get("feature_dim"),
                        **met,
                    })
    # Aggregate rows by variant/anchor/target.
    agg: List[Dict[str, Any]] = []
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("status") != "PASS":
            continue
        groups[(r.get("variant"), r.get("target_visual"), r.get("anchor_count"))].append(r)
    for (variant, target, anchor_count), rs in sorted(groups.items(), key=lambda x: (str(x[0][0]), str(x[0][1]), int(x[0][2]))):
        agg.append({
            "variant": variant,
            "target_visual": target,
            "anchor_count": anchor_count,
            "seed_count": len(rs),
            "t2v_rank@1_mean": _mean([_safe_float(r.get("t2v_rank@1")) for r in rs]),
            "t2v_rank@1_std": float(np.std([_safe_float(r.get("t2v_rank@1")) for r in rs if math.isfinite(_safe_float(r.get("t2v_rank@1")))]) if rs else float("nan")),
            "t2v_mean_rank_mean": _mean([_safe_float(r.get("t2v_mean_rank")) for r in rs]),
            "v2t_rank@1_mean": _mean([_safe_float(r.get("v2t_rank@1")) for r in rs]),
            "v2t_mean_rank_mean": _mean([_safe_float(r.get("v2t_mean_rank")) for r in rs]),
            "eval_count_mean": _mean([_safe_float(r.get("eval_count")) for r in rs]),
        })
    _write_csv(out_root / "anchor_count_curve_rows.csv", rows)
    _write_csv(out_root / "anchor_count_curve_summary.csv", agg)
    payload = {"status": "PASS", "row_count": len(rows), "summary_count": len(agg), "artifacts": {"rows_csv": str(out_root / "anchor_count_curve_rows.csv"), "summary_csv": str(out_root / "anchor_count_curve_summary.csv")}}
    _write_json(out_root / "anchor_count_curve_summary.json", payload)
    return payload


def run_projector_distortion(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    ids: List[int] = list(ctx["ids"])
    visual_train = np.asarray(ctx["visual_train_mat"], dtype=np.float32)
    visual_val = np.asarray(ctx["visual_val_mat"], dtype=np.float32)
    train_valid = np.asarray(ctx["train_valid"], dtype=bool)
    val_valid = np.asarray(ctx["val_valid"], dtype=bool)
    out_root: Path = ctx["output_root"] / "projector_distortion"
    out_root.mkdir(parents=True, exist_ok=True)
    args = ctx["args"]
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    ckpts = ctx["checkpoints"]
    rows: List[Dict[str, Any]] = []
    for spec in ckpts:
        ckpt_path = Path(str(spec["checkpoint_path"])).expanduser().resolve()
        try:
            projector, _temp, _unk, ckpt, cfg = _load_checkpoint_projector(ckpt_path, device)
            t_ids, t_mat, _names, tb_meta = _load_text_bank_for_checkpoint(ctx["asset_root"], args.train_dataset_name, ckpt)
            text = _submatrix_for_ids(t_ids, t_mat, ids)
            projected = _project_text_matrix(projector, text, device=device)
            raw_sim = _cos(text)
            proj_sim = _cos(projected)
            train_sim = _cos(np.nan_to_num(visual_train, nan=0.0))
            val_sim = _cos(np.nan_to_num(visual_val, nan=0.0))
            for target_name, vis_sim, valid in [("train_visual", train_sim, train_valid), ("val_visual", val_sim, val_valid)]:
                rows.append({
                    "status": "PASS",
                    "checkpoint_name": spec.get("name"),
                    "checkpoint_path": str(ckpt_path),
                    "variant": tb_meta.get("variant"),
                    "projector_type": cfg.get("projector_type"),
                    "feature_dim": tb_meta.get("feature_dim"),
                    "comparison": f"raw_text_vs_{target_name}",
                    **_graph_metrics(raw_sim, vis_sim, ids, valid, int(args.neighbor_k)),
                })
                rows.append({
                    "status": "PASS",
                    "checkpoint_name": spec.get("name"),
                    "checkpoint_path": str(ckpt_path),
                    "variant": tb_meta.get("variant"),
                    "projector_type": cfg.get("projector_type"),
                    "feature_dim": tb_meta.get("feature_dim"),
                    "comparison": f"projected_text_vs_{target_name}",
                    **_graph_metrics(proj_sim, vis_sim, ids, valid, int(args.neighbor_k)),
                })
            rows.append({
                "status": "PASS",
                "checkpoint_name": spec.get("name"),
                "checkpoint_path": str(ckpt_path),
                "variant": tb_meta.get("variant"),
                "projector_type": cfg.get("projector_type"),
                "feature_dim": tb_meta.get("feature_dim"),
                "comparison": "raw_text_vs_projected_text",
                **_graph_metrics(raw_sim, proj_sim, ids, np.ones((len(ids),), dtype=bool), int(args.neighbor_k)),
            })
        except Exception as exc:
            rows.append({"status": "FAIL", "checkpoint_name": spec.get("name"), "checkpoint_path": str(ckpt_path), "error": str(exc)})
    _write_csv(out_root / "projector_distortion_summary.csv", rows)
    payload = {"status": "PASS", "checkpoint_count": len(ckpts), "row_count": len(rows), "artifacts": {"summary_csv": str(out_root / "projector_distortion_summary.csv")}}
    _write_json(out_root / "projector_distortion_summary.json", payload)
    return payload


def run_row_level_margin(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    ids: List[int] = list(ctx["ids"])
    visible = set(ids)
    out_root: Path = ctx["output_root"] / "row_level_margin"
    out_root.mkdir(parents=True, exist_ok=True)
    args = ctx["args"]
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    rows_val = list(ctx["val_rows"])
    carrier_val = np.asarray(ctx["val_carrier"], dtype=np.float32)
    checkpoints = ctx["checkpoints"]
    summary_rows: List[Dict[str, Any]] = []
    all_per_row: List[Dict[str, Any]] = []
    all_class_rows: List[Dict[str, Any]] = []
    for spec in checkpoints:
        ckpt_path = Path(str(spec["checkpoint_path"])).expanduser().resolve()
        try:
            projector, temperature, _unk, ckpt, cfg = _load_checkpoint_projector(ckpt_path, device)
            t_ids, t_mat, t_names, tb_meta = _load_text_bank_for_checkpoint(ctx["asset_root"], args.train_dataset_name, ckpt)
            id_to_text_idx = {int(r): i for i, r in enumerate(t_ids)}
            candidate_ids = [rid for rid in ids if rid in id_to_text_idx]
            cand_idx = [id_to_text_idx[rid] for rid in candidate_ids]
            text_sub = np.asarray(t_mat[cand_idx], dtype=np.float32)
            projected = _project_text_matrix(projector, text_sub, device=device)
            z = _l2(carrier_val)
            logits = (z @ projected.T) / max(float(temperature), 1e-12)
            per: List[Dict[str, Any]] = []
            ranks: List[int] = []
            margin_top_wrong: List[float] = []
            margin_person: List[float] = []
            top1_ids: List[int] = []
            for i, row in enumerate(rows_val):
                gt = _as_int(row.get("raw_category_id"))
                if gt is None or int(gt) not in visible or int(gt) not in candidate_ids:
                    continue
                scores = np.asarray(logits[i], dtype=np.float64)
                order = np.argsort(-scores, kind="mergesort")
                gt_pos = candidate_ids.index(int(gt))
                rank = int(np.where(order == gt_pos)[0][0]) + 1
                top1_pos = int(order[0])
                top1 = int(candidate_ids[top1_pos])
                wrong_scores = np.delete(scores, gt_pos)
                nearest_wrong = float(np.max(wrong_scores)) if wrong_scores.size else float("nan")
                gt_logit = float(scores[gt_pos])
                person_logit = float(scores[candidate_ids.index(773)]) if 773 in candidate_ids else float("nan")
                m_wrong = float(gt_logit - nearest_wrong) if math.isfinite(nearest_wrong) else float("nan")
                m_person = float(gt_logit - person_logit) if math.isfinite(person_logit) else float("nan")
                ranks.append(rank)
                margin_top_wrong.append(m_wrong)
                margin_person.append(m_person)
                top1_ids.append(top1)
                per.append({
                    "checkpoint_name": spec.get("name"),
                    "variant": tb_meta.get("variant"),
                    "projector_type": cfg.get("projector_type"),
                    "trajectory_id": row.get("trajectory_id"),
                    "video_id": row.get("video_id"),
                    "clip_id": row.get("clip_id"),
                    "gt_raw_id": int(gt),
                    "gt_name": t_names.get(int(gt), f"raw_id_{gt}"),
                    "rank": rank,
                    "top1_raw_id": top1,
                    "top1_name": t_names.get(top1, f"raw_id_{top1}"),
                    "top1_is_gt": int(top1 == int(gt)),
                    "top1_is_person": int(top1 == 773),
                    "gt_logit": gt_logit,
                    "top1_logit": float(scores[top1_pos]),
                    "nearest_wrong_logit": nearest_wrong,
                    "person_logit": person_logit,
                    "margin_gt_vs_top_wrong": m_wrong,
                    "margin_gt_vs_person": m_person,
                })
            all_per_row.extend(per)
            top1_counter = Counter(top1_ids)
            summary_rows.append({
                "status": "PASS",
                "checkpoint_name": spec.get("name"),
                "checkpoint_path": str(ckpt_path),
                "variant": tb_meta.get("variant"),
                "projector_type": cfg.get("projector_type"),
                "feature_dim": tb_meta.get("feature_dim"),
                "candidate_count": len(candidate_ids),
                "class_count": len({int(r["gt_raw_id"]) for r in per}),
                **_summary_ranks(ranks),
                "mean_margin_gt_vs_top_wrong": _mean(margin_top_wrong),
                "median_margin_gt_vs_top_wrong": _median(margin_top_wrong),
                "positive_margin_gt_vs_top_wrong_rate": float(np.mean(np.asarray(margin_top_wrong) > 0)) if margin_top_wrong else 0.0,
                "mean_margin_gt_vs_person": _mean(margin_person),
                "positive_margin_gt_vs_person_rate": float(np.mean(np.asarray([m for m in margin_person if math.isfinite(m)]) > 0)) if any(math.isfinite(m) for m in margin_person) else 0.0,
                "top1_person_rate": float(top1_counter.get(773, 0) / max(len(per), 1)),
                "top1_max_hub_raw_id": int(top1_counter.most_common(1)[0][0]) if top1_counter else None,
                "top1_max_hub_count": int(top1_counter.most_common(1)[0][1]) if top1_counter else 0,
            })
            by_class: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
            for r in per:
                by_class[int(r["gt_raw_id"])].append(r)
            for rid, rs in by_class.items():
                rr = [int(r["rank"]) for r in rs]
                all_class_rows.append({
                    "checkpoint_name": spec.get("name"),
                    "variant": tb_meta.get("variant"),
                    "projector_type": cfg.get("projector_type"),
                    "gt_raw_id": rid,
                    "gt_name": t_names.get(rid, f"raw_id_{rid}"),
                    **_summary_ranks(rr),
                    "mean_margin_gt_vs_top_wrong": _mean([_safe_float(r.get("margin_gt_vs_top_wrong")) for r in rs]),
                    "top1_person_rate": float(np.mean([int(r.get("top1_is_person", 0)) for r in rs])),
                })
        except Exception as exc:
            summary_rows.append({"status": "FAIL", "checkpoint_name": spec.get("name"), "checkpoint_path": str(ckpt_path), "error": str(exc)})
    _write_csv(out_root / "row_level_margin_summary.csv", summary_rows)
    _write_csv(out_root / "row_level_margin_per_row.csv", all_per_row)
    _write_csv(out_root / "row_level_margin_per_class.csv", all_class_rows)
    payload = {"status": "PASS", "checkpoint_count": len(checkpoints), "summary_rows": summary_rows, "artifacts": {"summary_csv": str(out_root / "row_level_margin_summary.csv"), "per_row_csv": str(out_root / "row_level_margin_per_row.csv"), "per_class_csv": str(out_root / "row_level_margin_per_class.csv")}}
    _write_json(out_root / "row_level_margin_summary.json", payload)
    return payload


def _make_takeover(output_root: Path, result: Mapping[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# A8 Manifold Alignment Diagnosis TAKEOVER\n")
    lines.append("## Status\n")
    lines.append(f"- overall_status: `{result.get('status')}`")
    lines.append(f"- output_root: `{output_root}`")
    lines.append("\n## Experiments\n")
    for key in ["class_proto_alignment", "anchor_curve", "projector_distortion", "row_level_margin"]:
        val = result.get(key, {}) if isinstance(result.get(key, {}), Mapping) else {}
        lines.append(f"- {key}: `{val.get('status', 'SKIPPED')}`")
        arts = val.get("artifacts", {}) if isinstance(val.get("artifacts", {}), Mapping) else {}
        for ak, av in arts.items():
            lines.append(f"  - {ak}: `{av}`")
    lines.append("\n## Interpretation checklist\n")
    lines.append("- If class-prototype holdout and anchor-curve fail: manifold extrapolation premise is not supported.")
    lines.append("- If they pass but projector distortion fails: trained projector is likely destroying useful topology.")
    lines.append("- If projector graph is preserved but row-level margins fail: bottleneck is trajectory-to-class calibration / carrier noise / assignment dynamics.")
    lines.append("- Use rank@K and margin summaries as diagnostic evidence; this report does not mutate checkpoints or define a new primary metric.\n")
    (output_root / "manifold_diagnosis_takeover.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run A8 manifold alignment diagnosis audits")
    p.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--run_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--train_dataset_name", default="lvvis_train_base")
    p.add_argument("--val_dataset_name", default="lvvis_val")
    p.add_argument("--train_annotation_json", default="")
    p.add_argument("--val_annotation_json", default="")
    p.add_argument("--visible_csv", required=True)
    p.add_argument("--visual_only_root", default="/home/zyy/code/wsovvis_asserts/text_bank_llama3/lvvis/lvvis_visual_only_v1")
    p.add_argument("--direct_concept_root", default="/home/zyy/code/wsovvis_asserts/text_bank_llama3/lvvis/llama3_direct_concept_v1")
    p.add_argument("--variants", default="clip_current,clip_of_llm_mean,llama_hidden_mean,llama_direct_concept_mean")
    p.add_argument("--only", default="all", choices=["all", "class_proto", "anchor_curve", "projector_distortion", "row_margin"])
    p.add_argument("--mapping_methods", default="ridge,least_squares")
    p.add_argument("--ridge_alpha", type=float, default=1e-2)
    p.add_argument("--holdout_fraction", type=float, default=0.2)
    p.add_argument("--anchor_counts", default="32,64,128,256,384,450")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--neighbor_k", type=int, default=10)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--checkpoint_specs", default="", help="Optional semicolon-separated NAME=PATH list. If empty, discover known A8 text-bank runs under run_root.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _ensure_repo(repo_root)
    gtceil = _load_gtceil(repo_root)

    train_ann = Path(args.train_annotation_json).expanduser().resolve() if args.train_annotation_json else repo_root / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "train_instances.json"
    val_ann = Path(args.val_annotation_json).expanduser().resolve() if args.val_annotation_json else repo_root / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "val_instances.json"
    visible_ids = _load_visible_ids(Path(args.visible_csv).expanduser().resolve())
    variants = [x.strip() for x in str(args.variants).split(",") if x.strip()]
    text_banks = _load_all_text_banks(asset_root, args.train_dataset_name, Path(args.visual_only_root).expanduser().resolve(), Path(args.direct_concept_root).expanduser().resolve(), variants)

    # Scoped ids must be present in canonical CLIP bank and visible set.
    canonical_ids = set(text_banks["clip_current"][0]) if "clip_current" in text_banks else set(next(iter(text_banks.values()))[0])
    ids = sorted(canonical_ids & visible_ids)
    if not ids:
        raise RuntimeError("empty visible class scope")

    train_rows, train_carrier, train_meta = _rows_and_carriers(gtceil, asset_root=asset_root, dataset_name=args.train_dataset_name, ann=train_ann, max_rows=int(args.max_rows))
    val_rows, val_carrier, val_meta = _rows_and_carriers(gtceil, asset_root=asset_root, dataset_name=args.val_dataset_name, ann=val_ann, max_rows=int(args.max_rows))
    train_proto, train_counts = _visual_prototypes(train_rows, train_carrier)
    val_proto, val_counts = _visual_prototypes(val_rows, val_carrier)
    visual_train_mat, train_valid = _matrix_for_ids(train_proto, ids)
    visual_val_mat, val_valid = _matrix_for_ids(val_proto, ids)
    checkpoints = _discover_checkpoints(run_root, args.checkpoint_specs)

    ctx = {
        "args": args,
        "repo_root": repo_root,
        "asset_root": asset_root,
        "run_root": run_root,
        "output_root": output_root,
        "ids": ids,
        "text_banks": text_banks,
        "train_rows": train_rows,
        "train_carrier": train_carrier,
        "val_rows": val_rows,
        "val_carrier": val_carrier,
        "visual_train_mat": visual_train_mat,
        "visual_val_mat": visual_val_mat,
        "train_valid": train_valid,
        "val_valid": val_valid,
        "train_meta": train_meta,
        "val_meta": val_meta,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "checkpoints": checkpoints,
    }

    result: Dict[str, Any] = {
        "status": "PASS",
        "output_root": str(output_root),
        "class_scope": "visible525",
        "class_count": len(ids),
        "variants": variants,
        "checkpoint_count": len(checkpoints),
        "visual_source_meta": {"train": train_meta, "val": val_meta},
    }
    if args.only in {"all", "class_proto"}:
        result["class_proto_alignment"] = run_class_proto_alignment(ctx)
    if args.only in {"all", "anchor_curve"}:
        result["anchor_curve"] = run_anchor_curve(ctx)
    if args.only in {"all", "projector_distortion"}:
        result["projector_distortion"] = run_projector_distortion(ctx)
    if args.only in {"all", "row_margin"}:
        result["row_level_margin"] = run_row_level_margin(ctx)

    _write_json(output_root / "manifold_diagnosis_report.json", result)
    _make_takeover(output_root, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
