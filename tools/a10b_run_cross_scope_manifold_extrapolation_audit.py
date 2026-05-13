#!/usr/bin/env python3
"""A10B cross-scope manifold extrapolation audit.

Read-only diagnostic overlay.  It fits text->vision projectors only from a
sampled subset of visible525 train anchors, then evaluates transfer to classes
outside that anchor pool: train-base outside visible525, val-base, and novel.

This script writes only under --output_root and does not mutate training code,
checkpoints, text banks, carrier banks, or annotations.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_G8_ROOT = "codex/outputs/G8_inference_and_eval"
DEFAULT_RUN_NAME = "sinkhorn_safeneg_k32_b010_preaug_k10_repro_20260427"
DEFAULT_OUT_NAME = "A10B_CROSS_SCOPE_MANIFOLD_EXTRAPOLATION_AUDIT"
DEFAULT_VARIANTS = "clip_current,clip_of_llm_mean,llama_hidden_mean,llama_direct_concept_mean"
DEFAULT_TRANSFORMS = "S0_identity,S1_orthogonal,S2_anisotropic_linear,S3_noisy_linear,S4_lowrank_linear"
DEFAULT_PROJECTORS = "identity,orthogonal_procrustes,ridge,least_squares,lowrank_ridge,oracle_inverse"
DEFAULT_VISUAL_ONLY_ROOT = "/home/zyy/code/wsovvis_asserts/text_bank_llama3/lvvis/lvvis_visual_only_v1"
DEFAULT_DIRECT_CONCEPT_ROOT = "/home/zyy/code/wsovvis_asserts/text_bank_llama3/lvvis/llama3_direct_concept_v1"
PERSON_RAW_ID = 773


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
        raise FileNotFoundError(f"A10 helper is required but missing: {path}")
    return _load_module(path, "_a10_helper_for_a10b")


def _repo_default() -> Path:
    return Path.cwd().resolve()


def _run_root_default(repo_root: Path) -> Path:
    p = repo_root / DEFAULT_G8_ROOT / DEFAULT_RUN_NAME
    return p if p.exists() else repo_root / DEFAULT_G8_ROOT


def _output_root_default(repo_root: Path) -> Path:
    return repo_root / DEFAULT_G8_ROOT / DEFAULT_OUT_NAME


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return default


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def _rank_summary(ranks: Sequence[int], prefix: str) -> Dict[str, Any]:
    n = len(ranks)
    if n <= 0:
        return {prefix + "count": 0, prefix + "rank@1": 0.0, prefix + "rank@5": 0.0, prefix + "rank@10": 0.0, prefix + "mean_rank": None, prefix + "median_rank": None}
    arr = np.asarray(ranks, dtype=np.float64)
    return {
        prefix + "count": int(n),
        prefix + "rank@1": float(np.mean(arr <= 1)),
        prefix + "rank@5": float(np.mean(arr <= 5)),
        prefix + "rank@10": float(np.mean(arr <= 10)),
        prefix + "mean_rank": float(np.mean(arr)),
        prefix + "median_rank": float(np.median(arr)),
    }


def _mean(vals: Iterable[float]) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return float(np.mean(xs)) if xs else float("nan")


def _load_official_split(repo_root: Path, split_json: Optional[str] = None) -> Tuple[set[int], set[int], Dict[int, str]]:
    p = Path(split_json).expanduser().resolve() if split_json else repo_root / "package" / "reference" / "lvvis_official_base_novel_split.json"
    if not p.is_file():
        raise FileNotFoundError(f"official split json not found: {p}")
    obj = _read_json(p)
    base = set(int(x) for x in obj.get("base_raw_ids", []))
    novel = set(int(x) for x in obj.get("novel_raw_ids", []))
    names: Dict[int, str] = {}
    for c in obj.get("categories", []) or []:
        rid = _as_int(c.get("raw_id")) if isinstance(c, Mapping) else None
        if rid is not None:
            names[int(rid)] = str(c.get("class_name", c.get("name", f"raw_id_{rid}")))
    if not base or not novel:
        raise RuntimeError(f"bad official split: base={len(base)} novel={len(novel)} path={p}")
    return base, novel, names


def _matrix_from_proto(proto: Mapping[int, np.ndarray], ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    finite_vecs = [np.asarray(v, dtype=np.float32) for v in proto.values() if np.asarray(v).ndim == 1]
    if not finite_vecs:
        raise RuntimeError("empty prototype dictionary")
    dim = int(finite_vecs[0].shape[0])
    mat = np.full((len(ids), dim), np.nan, dtype=np.float32)
    valid = np.zeros(len(ids), dtype=bool)
    for i, rid in enumerate(ids):
        v = proto.get(int(rid))
        if v is None:
            continue
        vv = np.asarray(v, dtype=np.float32)
        if vv.shape[0] != dim or not np.isfinite(vv).all():
            continue
        mat[i] = vv
        valid[i] = True
    mat[valid] = _l2(mat[valid])
    return mat, valid


def _source_visual_matrix(ids: Sequence[int], train_proto: Mapping[int, np.ndarray], val_proto: Mapping[int, np.ndarray], source_policy: str, anchor_pool: set[int]) -> Tuple[np.ndarray, np.ndarray]:
    # source_policy controls synthetic text construction.  Anchor ids always use
    # train prototypes because projector fitting is train-anchor based.
    dim_src = None
    for d in (train_proto, val_proto):
        for v in d.values():
            vv = np.asarray(v)
            if vv.ndim == 1:
                dim_src = int(vv.shape[0]); break
        if dim_src is not None:
            break
    if dim_src is None:
        raise RuntimeError("cannot infer visual dimension")
    mat = np.full((len(ids), dim_src), np.nan, dtype=np.float32)
    valid = np.zeros(len(ids), dtype=bool)
    for i, rid0 in enumerate(ids):
        rid = int(rid0)
        v = None
        if rid in anchor_pool and rid in train_proto:
            v = train_proto.get(rid)
        elif source_policy == "val_prefer" and rid in val_proto:
            v = val_proto.get(rid)
        elif source_policy == "train_prefer" and rid in train_proto:
            v = train_proto.get(rid)
        elif rid in val_proto:
            v = val_proto.get(rid)
        elif rid in train_proto:
            v = train_proto.get(rid)
        if v is None:
            continue
        vv = np.asarray(v, dtype=np.float32)
        if vv.shape[0] != dim_src or not np.isfinite(vv).all():
            continue
        mat[i] = vv
        valid[i] = True
    mat[valid] = _l2(mat[valid])
    return mat, valid


def _sample_anchors(a10: Any, anchor_pool: Sequence[int], train_counts: Mapping[int, int], per_class_meta: Mapping[int, Mapping[str, Any]], ratio: float, calib_fraction: float, seed: int) -> Tuple[List[int], List[int], Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    by_bucket: Dict[str, List[int]] = defaultdict(list)
    for rid in anchor_pool:
        by_bucket[a10._support_bucket_for(int(rid), train_counts, per_class_meta)].append(int(rid))
    train: List[int] = []
    calib: List[int] = []
    bucket_rows: List[Dict[str, Any]] = []
    for bucket, vals0 in sorted(by_bucket.items()):
        vals = list(vals0)
        rng.shuffle(vals)
        n = len(vals)
        n_train = int(round(n * float(ratio)))
        n_train = min(max(1, n_train), n)
        rem = vals[n_train:]
        n_calib = int(round(n * float(calib_fraction)))
        if rem:
            n_calib = min(max(1, n_calib), len(rem))
            calib_part = rem[:n_calib]
        else:
            # ratio=1.0: use a small deterministic subset only for selecting
            # hyperparameters.  Target scopes remain unseen and are never used.
            n_calib = min(max(1, int(round(n * float(calib_fraction)))), n_train)
            calib_part = vals[:n_calib]
        train.extend(vals[:n_train])
        calib.extend(calib_part)
        bucket_rows.append({"support_bucket": bucket, "pool_count": n, "anchor_train_count": n_train, "anchor_calib_count": n_calib})
    return sorted(set(train)), sorted(set(calib)), {"bucket_rows": bucket_rows}


def _candidate_ids(scope: str, target_ids: Sequence[int], anchor_pool: Sequence[int], base_ids: set[int], novel_ids: set[int], train_proto: Mapping[int, np.ndarray], val_proto: Mapping[int, np.ndarray], eval_visual: str, text_id_set: set[int]) -> List[int]:
    target = set(int(x) for x in target_ids)
    anchor = set(int(x) for x in anchor_pool)
    if eval_visual == "train_proto":
        available = set(int(x) for x in train_proto.keys()) & text_id_set
    else:
        available = set(int(x) for x in val_proto.keys()) & text_id_set
    if scope == "target_only":
        ids = target
    elif scope == "visible525_plus_target":
        ids = anchor | target
    elif scope == "official_base_available":
        ids = set(base_ids) & available
    elif scope == "novel_available":
        ids = set(novel_ids) & available
    elif scope == "base_plus_novel_available":
        ids = (set(base_ids) | set(novel_ids)) & available
    elif scope == "full_available":
        ids = available
    else:
        raise ValueError(f"unsupported candidate_scope={scope}")
    return sorted(int(x) for x in ids if int(x) in text_id_set)


def _evaluate_rows(projected: np.ndarray, ids: Sequence[int], rows: Sequence[Mapping[str, Any]], carrier: np.ndarray, target_ids: Sequence[int], candidate_ids: Sequence[int], names: Mapping[int, str], row_limit: int, meta: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    id_to_idx = {int(r): i for i, r in enumerate(ids)}
    cand_ids = [int(r) for r in candidate_ids if int(r) in id_to_idx]
    cand_idx = [id_to_idx[r] for r in cand_ids]
    if not cand_idx:
        return {**meta, "row_count": 0, "candidate_count": 0}, []
    cand_mat = _l2(np.nan_to_num(projected[cand_idx], nan=0.0))
    target_set = set(int(x) for x in target_ids)
    ranks: List[int] = []
    top1_ids: List[int] = []
    margins: List[float] = []
    per_class: Dict[int, List[int]] = defaultdict(list)
    scanned = 0
    person_pos = cand_ids.index(PERSON_RAW_ID) if PERSON_RAW_ID in cand_ids else None
    person_margins: List[float] = []
    for i, row in enumerate(rows):
        rid = _as_int(_row_get(row, "raw_category_id"))
        if rid is None or int(rid) not in target_set or int(rid) not in id_to_idx:
            continue
        if i >= int(carrier.shape[0]):
            continue
        if int(rid) not in cand_ids:
            continue
        if row_limit > 0 and scanned >= row_limit:
            break
        z = np.asarray(carrier[i], dtype=np.float32)
        if not np.isfinite(z).all():
            continue
        scanned += 1
        scores = cand_mat @ _l2(z.reshape(1, -1))[0]
        order = np.argsort(-scores, kind="mergesort")
        target_pos = cand_ids.index(int(rid))
        rank = int(np.where(order == target_pos)[0][0]) + 1
        top1_pos = int(order[0])
        top1 = int(cand_ids[top1_pos])
        top1_ids.append(top1)
        ranks.append(rank)
        per_class[int(rid)].append(rank)
        gt_score = float(scores[target_pos])
        if len(order) > 1:
            wrong_pos = int(order[0]) if int(order[0]) != target_pos else int(order[1])
            margins.append(gt_score - float(scores[wrong_pos]))
        if person_pos is not None:
            person_margins.append(gt_score - float(scores[person_pos]))
    top_counter = Counter(top1_ids)
    summary = {
        **meta,
        "row_count": int(len(ranks)),
        "candidate_count": int(len(cand_ids)),
        "class_count_with_rows": int(len(per_class)),
        **_rank_summary(ranks, "row_"),
        "mean_margin_gt_vs_top_wrong": _mean(margins),
        "positive_margin_gt_vs_top_wrong_rate": float(np.mean(np.asarray(margins) > 0)) if margins else 0.0,
        "mean_margin_gt_vs_person": _mean(person_margins),
        "positive_margin_gt_vs_person_rate": float(np.mean(np.asarray(person_margins) > 0)) if person_margins else 0.0,
        "top1_person_rate": float(top_counter.get(PERSON_RAW_ID, 0) / max(len(ranks), 1)),
        "top1_max_raw_id": int(top_counter.most_common(1)[0][0]) if top_counter else None,
        "top1_max_count": int(top_counter.most_common(1)[0][1]) if top_counter else 0,
    }
    per_class_rows: List[Dict[str, Any]] = []
    for rid, rr in sorted(per_class.items()):
        per_class_rows.append({**meta, "gt_raw_id": int(rid), "gt_name": names.get(int(rid), f"raw_id_{rid}"), "row_count": len(rr), **_rank_summary(rr, "row_")})
    return summary, per_class_rows


def _class_eval(a10: Any, projected: np.ndarray, visual: np.ndarray, ids: Sequence[int], target_ids: Sequence[int], candidate_ids: Sequence[int]) -> Dict[str, Any]:
    return a10._evaluate_class_retrieval(projected, visual, ids, target_ids, candidate_ids)



def _load_real_banks_available(a10: Any, a8: Any, asset_root: Path, dataset_name: str, visual_root: Path, direct_root: Path, variants: Sequence[str]) -> Tuple[Dict[str, Tuple[List[int], np.ndarray, Dict[int, str], Dict[str, Any]]], List[Dict[str, Any]]]:
    """Load requested real text banks, skipping optional banks whose files are absent.

    A10B should not fail the entire cross-scope synthetic audit just because an
    optional external text-bank root is unavailable on a particular snapshot.
    clip_current remains required; external LLaMA/visual-only banks are included
    when their roots are present.
    """
    out: Dict[str, Tuple[List[int], np.ndarray, Dict[int, str], Dict[str, Any]]] = {}
    skipped: List[Dict[str, Any]] = []
    for v in variants:
        try:
            if v == "clip_current":
                out[v] = a8._load_current_clip_text_bank(asset_root, dataset_name)
            elif v in {"clip_of_llm_mean", "llama_hidden_mean"}:
                root = visual_root
                if not (root / "lvvis_class_names.json").is_file():
                    skipped.append({"variant": v, "root": str(root), "reason": "missing lvvis_class_names.json"})
                    continue
                out[v] = a10._load_external_text_bank_robust(a8, root, v)
            elif v == "llama_direct_concept_mean":
                root = direct_root
                if not (root / "lvvis_class_names.json").is_file():
                    skipped.append({"variant": v, "root": str(root), "reason": "missing lvvis_class_names.json"})
                    continue
                out[v] = a10._load_external_text_bank_robust(a8, root, v)
            else:
                skipped.append({"variant": v, "root": "", "reason": "unsupported variant"})
        except Exception as exc:
            skipped.append({"variant": v, "root": str(visual_root if v in {"clip_of_llm_mean", "llama_hidden_mean"} else direct_root), "reason": repr(exc)})
    if not out:
        raise RuntimeError("no real text bank could be loaded; check --asset_root/--visual_only_root/--direct_concept_root or pass --variants clip_current")
    return out, skipped

def _build_takeover(output_root: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# A10B Cross-Scope Manifold Extrapolation TAKEOVER", "",
        "## Status",
        f"- overall_status: `{result.get('status')}`",
        f"- output_root: `{result.get('output_root')}`",
        f"- analysis_root: `{result.get('analysis_root')}`", "",
        "## Scope",
        "- Fit source: sampled visible525 train anchors only.",
        "- Test scopes: train_base_outside_525, val_base_all, val_base_outside_525, novel_val.",
        "- Synthetic features are positive controls; real features use existing text banks.",
        "- Read-only audit; no model/training/data assets are mutated.", "",
        "## Required artifacts",
    ]
    for rel in [
        "analysis/cross_scope_availability_inventory.csv",
        "analysis/cross_scope_class_proto_summary.csv",
        "analysis/cross_scope_row_level_summary.csv",
        "analysis/cross_scope_anchor_ratio_curve.csv",
        "analysis/cross_scope_projector_selection_by_calib.csv",
        "analysis/A10B_cross_scope_summary.json",
    ]:
        p = output_root / rel
        lines.append(f"- `{p}`: {'FOUND' if p.is_file() else 'MISSING'}")
    lines += ["", "## Interpretation checklist",
              "- synthetic high but real low => real text/vision manifold mismatch across target scope.",
              "- synthetic visible525-heldout high but synthetic novel low => visible525 anchors do not cover novel visual manifold.",
              "- class high but row low => row-level boundary/topK-to-top1 failure.",
              "- target_only high but full_available low => global distractor competition."]
    (output_root / "A10B_CROSS_SCOPE_MANIFOLD_EXTRAPOLATION_TAKEOVER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    p.add_argument("--direct_concept_root", default=DEFAULT_DIRECT_CONCEPT_ROOT)
    p.add_argument("--variants", default=DEFAULT_VARIANTS)
    p.add_argument("--synthetic_transforms", default=DEFAULT_TRANSFORMS)
    p.add_argument("--projectors", default=DEFAULT_PROJECTORS)
    p.add_argument("--anchor_ratios", default="0.1,0.2,0.4,0.6,0.8,1.0")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--anchor_calib_fraction", type=float, default=0.1)
    p.add_argument("--candidate_scopes", default="target_only,visible525_plus_target,official_base_available,full_available")
    p.add_argument("--test_scopes", default="train_base_outside_525,val_base_all,val_base_outside_525,novel_val")
    p.add_argument("--ridge_alphas", default="0.0001,0.001,0.01,0.1,1.0")
    p.add_argument("--lowrank_dims", default="32,64,128,256")
    p.add_argument("--synthetic_noise_sigma", type=float, default=0.03)
    p.add_argument("--synthetic_lowrank_dim", type=int, default=128)
    p.add_argument("--max_rows", type=int, default=0, help="Max GT-bound rows/carriers loaded per dataset; 0 means full.")
    p.add_argument("--row_max_rows", type=int, default=0, help="Max evaluated rows per case; 0 means all loaded rows.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_max_rows", type=int, default=5000)
    p.add_argument("--smoke_row_max_rows", type=int, default=2000)
    p.add_argument("--skip_row_level", action="store_true")
    p.add_argument("--write_per_class", action="store_true", help="Write row-level per-class details.")
    p.add_argument("--continue_on_error", action="store_true")
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
    if args.smoke:
        args.anchor_ratios = "0.2,0.8"
        args.seeds = "0,1"
        args.ridge_alphas = "0.01"
        args.lowrank_dims = "64,128"
        if int(args.max_rows) <= 0:
            args.max_rows = int(args.smoke_max_rows)
        if int(args.row_max_rows) <= 0:
            args.row_max_rows = int(args.smoke_row_max_rows)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    result: Dict[str, Any] = {"status": "PASS", "repo_root": str(repo_root), "asset_root": str(asset_root), "run_root": str(run_root), "output_root": str(output_root), "analysis_root": str(analysis_root), "start_time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        a10 = _load_a10(repo_root)
        a8 = a10._load_a8_helper(repo_root)
        base_ids, novel_ids, official_names = _load_official_split(repo_root, args.official_split_json or None)
        visible_csv = Path(args.visible_csv).expanduser().resolve() if args.visible_csv else a10._find_visible_csv(repo_root, run_root)
        if not visible_csv or not Path(visible_csv).is_file():
            raise RuntimeError("visible525 csv not found; pass --visible_csv")
        visible_ids = set(int(x) for x in a10._load_visible_ids(a8, Path(visible_csv)))
        per_class_join = Path(args.per_class_join).expanduser().resolve() if args.per_class_join else a10._find_per_class_join(repo_root, run_root)
        per_class_meta = a10._load_per_class_meta(per_class_join)
        train_ann = Path(args.train_annotation_json).expanduser().resolve() if args.train_annotation_json else repo_root / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "train_instances.json"
        val_ann = Path(args.val_annotation_json).expanduser().resolve() if args.val_annotation_json else repo_root / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "val_instances.json"
        gtceil = a8._load_gtceil(repo_root)
        train_rows, train_carrier, train_meta = a8._rows_and_carriers(gtceil, asset_root=asset_root, dataset_name=args.train_dataset_name, ann=train_ann, max_rows=int(args.max_rows))
        val_rows, val_carrier, val_meta = a8._rows_and_carriers(gtceil, asset_root=asset_root, dataset_name=args.val_dataset_name, ann=val_ann, max_rows=int(args.max_rows))
        train_proto, train_counts = a8._visual_prototypes(train_rows, train_carrier)
        val_proto, val_counts = a8._visual_prototypes(val_rows, val_carrier)
        clip_ids, _clip_mat, clip_names, _clip_meta = a8._load_current_clip_text_bank(asset_root, args.train_dataset_name)
        text_id_set = set(int(x) for x in clip_ids)
        names = {**official_names, **{int(k): str(v) for k, v in clip_names.items()}}
        anchor_pool = sorted(visible_ids & base_ids & set(train_proto.keys()) & text_id_set)
        if len(anchor_pool) < 10:
            raise RuntimeError(f"too few anchor_pool classes: {len(anchor_pool)}")
        # Candidate/evaluation universe.  Include all classes that can appear in requested candidates.
        universe = sorted((text_id_set & (base_ids | novel_ids)) & (set(train_proto.keys()) | set(val_proto.keys()) | set(anchor_pool)))
        id_to_idx = {int(r): i for i, r in enumerate(universe)}
        visual_train_mat, train_valid = _matrix_from_proto(train_proto, universe)
        visual_val_mat, val_valid = _matrix_from_proto(val_proto, universe)
        # Text banks over full universe.
        variants = [x.strip() for x in str(args.variants).split(",") if x.strip()]
        real_banks, skipped_real_banks = _load_real_banks_available(a10, a8, asset_root, args.train_dataset_name, Path(args.visual_only_root).expanduser().resolve(), Path(args.direct_concept_root).expanduser().resolve(), variants)
        result["loaded_real_variants"] = sorted(real_banks.keys())
        result["skipped_real_variants"] = skipped_real_banks
        real_cases: List[Dict[str, Any]] = []
        for variant, (t_ids, t_mat, _names, meta) in real_banks.items():
            mat = a8._submatrix_for_ids(t_ids, t_mat, universe)
            real_cases.append({"feature_kind": "real", "feature_name": variant, "transform": "real_text", "text_matrix": _l2(np.asarray(mat, dtype=np.float32)), "meta": meta})
        target_defs: Dict[str, Dict[str, Any]] = {
            "train_base_outside_525": {"target_ids": sorted((base_ids - visible_ids) & set(train_proto.keys()) & text_id_set), "eval_visual": "train_proto", "rows": train_rows, "carrier": train_carrier, "source_policy": "train_prefer"},
            "val_base_all": {"target_ids": sorted(base_ids & set(val_proto.keys()) & text_id_set), "eval_visual": "val_proto", "rows": val_rows, "carrier": val_carrier, "source_policy": "val_prefer"},
            "val_base_outside_525": {"target_ids": sorted((base_ids - visible_ids) & set(val_proto.keys()) & text_id_set), "eval_visual": "val_proto", "rows": val_rows, "carrier": val_carrier, "source_policy": "val_prefer"},
            "novel_val": {"target_ids": sorted(novel_ids & set(val_proto.keys()) & text_id_set), "eval_visual": "val_proto", "rows": val_rows, "carrier": val_carrier, "source_policy": "val_prefer"},
        }
        test_scopes = [x.strip() for x in str(args.test_scopes).split(",") if x.strip()]
        candidate_scopes = [x.strip() for x in str(args.candidate_scopes).split(",") if x.strip()]
        anchor_ratios = [float(x) for x in str(args.anchor_ratios).split(",") if x.strip()]
        seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
        projectors = [x.strip() for x in str(args.projectors).split(",") if x.strip()]
        ridge_alphas = [float(x) for x in str(args.ridge_alphas).split(",") if x.strip()]
        lowrank_dims = [int(x) for x in str(args.lowrank_dims).split(",") if x.strip()]
        transforms = [x.strip() for x in str(args.synthetic_transforms).split(",") if x.strip()]
        class_rows: List[Dict[str, Any]] = []
        row_rows: List[Dict[str, Any]] = []
        selection_rows: List[Dict[str, Any]] = []
        availability_rows: List[Dict[str, Any]] = []
        per_class_rows: List[Dict[str, Any]] = []
        # Availability inventory.
        for ts in test_scopes:
            td = target_defs.get(ts)
            if not td:
                continue
            tset = set(td["target_ids"])
            row_count = 0
            for i, row in enumerate(td["rows"]):
                rid = _as_int(_row_get(row, "raw_category_id"))
                if rid is not None and int(rid) in tset and i < int(td["carrier"].shape[0]):
                    row_count += 1
            availability_rows.append({"test_scope": ts, "eval_visual": td["eval_visual"], "evaluable_class_count": len(tset), "row_count": row_count, "anchor_pool_count": len(anchor_pool)})
        _write_csv(analysis_root / "cross_scope_availability_inventory.csv", availability_rows)
        # Main loops.
        transform_seed = 20260512
        dim = int(visual_train_mat.shape[1])
        for seed in seeds:
            for ratio in anchor_ratios:
                fit_ids, calib_ids, split_meta = _sample_anchors(a10, anchor_pool, train_counts, per_class_meta, ratio, float(args.anchor_calib_fraction), seed)
                fit_idx = [id_to_idx[int(r)] for r in fit_ids if int(r) in id_to_idx and np.isfinite(visual_train_mat[id_to_idx[int(r)]]).all()]
                if not fit_idx:
                    continue
                # real cases do not depend on target scope; synthetic cases do because source visual can be train/val.
                for ts in test_scopes:
                    td = target_defs.get(ts)
                    if not td:
                        continue
                    target_ids = [int(x) for x in td["target_ids"] if int(x) in id_to_idx]
                    if not target_ids:
                        continue
                    eval_visual = str(td["eval_visual"])
                    eval_mat = visual_train_mat if eval_visual == "train_proto" else visual_val_mat
                    source_mat, source_valid = _source_visual_matrix(universe, train_proto, val_proto, str(td["source_policy"]), set(anchor_pool))
                    synth_cases: List[Dict[str, Any]] = []
                    for sname in transforms:
                        tobj = a10._make_transform(sname, dim, transform_seed, float(args.synthetic_noise_sigma), int(args.synthetic_lowrank_dim))
                        tmat = a10._apply_transform(np.nan_to_num(source_mat, nan=0.0), tobj, transform_seed)
                        # retain NaN for classes without source visual so retrieval ignores them via candidate filtering where possible
                        tmat[~source_valid] = np.nan
                        synth_cases.append({"feature_kind": "synthetic", "feature_name": sname, "transform": sname, "text_matrix": tmat, "transform_obj": tobj})
                    for case in synth_cases + real_cases:
                        # Choose projector using only visible525 anchor calib, never target classes.
                        selection_candidates = sorted(set(fit_ids) | set(calib_ids))
                        try:
                            best, sel_rows = a10._select_best_configs(case, fit_ids, calib_ids, universe, visual_train_mat, projectors, ridge_alphas, lowrank_dims, selection_candidates, case.get("transform_obj"))
                        except Exception as exc:
                            selection_rows.append({"seed": seed, "anchor_ratio": ratio, "test_scope": ts, "feature_kind": case.get("feature_kind"), "feature_name": case.get("feature_name"), "transform": case.get("transform"), "status": "FAIL", "error": str(exc)})
                            continue
                        for sr in sel_rows:
                            selection_rows.append({"seed": seed, "anchor_ratio": ratio, "test_scope": ts, "anchor_train_count": len(fit_ids), "anchor_calib_count": len(calib_ids), **sr})
                        projected = np.asarray(best["projected"], dtype=np.float32)
                        base_meta = {"seed": seed, "anchor_ratio": ratio, "anchor_train_count": len(fit_ids), "anchor_calib_count": len(calib_ids), "test_scope": ts, "eval_visual": eval_visual, "feature_kind": case.get("feature_kind"), "feature_name": case.get("feature_name"), "transform": case.get("transform"), "selected_projector": best.get("projector"), "selected_ridge_alpha": best.get("ridge_alpha") if math.isfinite(float(best.get("ridge_alpha", float('nan')))) else "", "selected_lowrank_dim": int(best.get("lowrank_dim")) if best.get("lowrank_dim") else ""}
                        for cs in candidate_scopes:
                            cand = _candidate_ids(cs, target_ids, anchor_pool, base_ids, novel_ids, train_proto, val_proto, eval_visual, text_id_set)
                            cand = [int(x) for x in cand if int(x) in id_to_idx]
                            met = _class_eval(a10, projected, eval_mat, universe, target_ids, cand)
                            class_rows.append({**base_meta, "candidate_scope": cs, "target_class_count": len(target_ids), **met})
                            if not args.skip_row_level:
                                summ, pcs = _evaluate_rows(projected, universe, td["rows"], td["carrier"], target_ids, cand, names, int(args.row_max_rows), {**base_meta, "candidate_scope": cs})
                                row_rows.append(summ)
                                if args.write_per_class:
                                    per_class_rows.extend(pcs)
        _write_csv(analysis_root / "cross_scope_projector_selection_by_calib.csv", selection_rows)
        _write_csv(analysis_root / "cross_scope_class_proto_summary.csv", class_rows)
        _write_csv(analysis_root / "cross_scope_row_level_summary.csv", row_rows)
        if args.write_per_class:
            _write_csv(analysis_root / "cross_scope_row_level_per_class.csv", per_class_rows)
        # A compact anchor-ratio table is the class summary aggregated by exact rows; downstream can aggregate further.
        _write_csv(analysis_root / "cross_scope_anchor_ratio_curve.csv", class_rows)
        summary = {
            "status": "PASS",
            "repo_root": str(repo_root),
            "asset_root": str(asset_root),
            "run_root": str(run_root),
            "output_root": str(output_root),
            "analysis_root": str(analysis_root),
            "visible_csv": str(visible_csv),
            "per_class_join": str(per_class_join) if per_class_join else "",
            "official_base_count": len(base_ids),
            "official_novel_count": len(novel_ids),
            "anchor_pool_count": len(anchor_pool),
            "universe_count": len(universe),
            "test_scopes": test_scopes,
            "candidate_scopes": candidate_scopes,
            "anchor_ratios": anchor_ratios,
            "seeds": seeds,
            "class_summary_rows": len(class_rows),
            "row_summary_rows": len(row_rows),
            "selection_rows": len(selection_rows),
            "availability_rows": len(availability_rows),
            "artifacts": {
                "availability_csv": str(analysis_root / "cross_scope_availability_inventory.csv"),
                "class_summary_csv": str(analysis_root / "cross_scope_class_proto_summary.csv"),
                "row_summary_csv": str(analysis_root / "cross_scope_row_level_summary.csv"),
                "anchor_ratio_curve_csv": str(analysis_root / "cross_scope_anchor_ratio_curve.csv"),
                "selection_csv": str(analysis_root / "cross_scope_projector_selection_by_calib.csv"),
            },
        }
        result.update(summary)
        _write_json(analysis_root / "A10B_cross_scope_summary.json", summary)
        _build_takeover(output_root, result)
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        _write_json(analysis_root / "A10B_cross_scope_summary.json", result)
        _build_takeover(output_root, result)
        if not args.continue_on_error:
            raise
    result["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(analysis_root / "A10B_run_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
