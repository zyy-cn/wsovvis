#!/usr/bin/env python3
"""A10G visible525 text-side correction endpoint probe for WS-OVVIS.

This read-only diagnostic creates corrected LLaMA4096 text banks from an existing
baseline/prompt/ensemble bank, then reuses the existing A10C GPU endpoint
evaluator. Correction fitting uses visible525 train anchors only. It does not
modify G7/G8 training/inference code, checkpoints, carrier banks, or source text
banks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

BASELINE_RIDGE_T2V_AT5 = 0.533156


def _repo_default() -> Path:
    return Path.cwd().resolve()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for k in row.keys():
            if str(k) not in fields:
                fields.append(str(k))
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fields:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_name(s: str) -> str:
    out = []
    for ch in str(s):
        out.append(ch if ch.isalnum() or ch in {"_", "-", "."} else "_")
    return "".join(out).strip("_") or "item"


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _run(cmd: Sequence[str], *, cwd: Path, log_path: Path, progress: bool) -> int:
    if progress:
        print("[A10G][cmd] " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write("\n[A10G][cmd] " + " ".join(shlex.quote(str(x)) for x in cmd) + "\n")
        log_f.flush()
        proc = subprocess.Popen(list(map(str, cmd)), cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            if progress:
                print(line, end="", flush=True)
            log_f.write(line)
            log_f.flush()
        rc = int(proc.wait())
        log_f.write(f"[A10G][exit] {rc}\n")
    return rc


def _load_bank(root: Path) -> Tuple[List[int], Dict[int, str], np.ndarray, Dict[str, Any]]:
    root = root.expanduser().resolve()
    class_path = root / "lvvis_class_names.json"
    payload_path = root / "payload" / "llama_hidden_mean.fp16.npz"
    manifest_path = root / "manifest.json"
    if not class_path.is_file():
        raise FileNotFoundError(class_path)
    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)
    obj = _read_json(class_path)
    rows = obj.get("classes", obj) if isinstance(obj, Mapping) else obj
    ids: List[int] = []
    names: Dict[int, str] = {}
    for row in rows:
        rid = int(row["raw_id"])
        ids.append(rid)
        names[rid] = str(row.get("name", row.get("class_name", f"raw_id_{rid}")))
    if ids != sorted(ids):
        raise RuntimeError(f"class raw_id order is not ascending: {class_path}")
    with np.load(payload_path, allow_pickle=False) as npz:
        if "protos" not in npz:
            raise RuntimeError(f"{payload_path} missing protos")
        arr = _l2(np.asarray(npz["protos"], dtype=np.float32))
    if int(arr.shape[0]) != len(ids):
        raise RuntimeError(f"class/payload mismatch root={root} ids={len(ids)} arr={arr.shape}")
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    return ids, names, arr, manifest if isinstance(manifest, dict) else {}


def _load_visual_anchor_target(repo_root: Path, asset_root: Path, run_root: Path, text_ids: Sequence[int], text_mat: np.ndarray, args: argparse.Namespace) -> Dict[str, Any]:
    a10c = _load_module(repo_root / "tools" / "a10c_run_llama4096_linear_isometric_distortion_calibration.py", "_a10c_for_a10g")
    a10 = a10c._load_a10(repo_root)
    a10b = a10c._load_a10b(repo_root)
    a8 = a10._load_a8_helper(repo_root)
    base_ids, novel_ids, official_names = a10b._load_official_split(repo_root, None)
    visible_csv = Path(args.visible_csv).expanduser().resolve() if args.visible_csv else a10._find_visible_csv(repo_root, run_root)
    visible_ids = set(int(x) for x in a10._load_visible_ids(a8, Path(visible_csv)))
    train_ann = Path(args.train_annotation_json).expanduser().resolve() if args.train_annotation_json else repo_root / "videocutler" / "datasets" / "LV-VIS" / "annotations" / "train_instances.json"
    gtceil = a8._load_gtceil(repo_root)
    train_rows, train_carrier, _meta = a8._rows_and_carriers(gtceil, asset_root=asset_root, dataset_name=args.train_dataset_name, ann=train_ann, max_rows=int(args.max_rows))
    train_proto, train_counts = a8._visual_prototypes(train_rows, train_carrier)
    text_id_set = set(int(x) for x in text_ids)
    anchor_pool = sorted(visible_ids & base_ids & set(train_proto.keys()) & text_id_set)
    if len(anchor_pool) < 10:
        raise RuntimeError(f"too few anchor_pool classes: {len(anchor_pool)}")
    id_to_idx = {int(r): i for i, r in enumerate(text_ids)}
    anchor_idx = [id_to_idx[int(r)] for r in anchor_pool]
    r_anchor = _l2(np.asarray(text_mat[anchor_idx], dtype=np.float32))
    v_anchor = _l2(np.asarray(a8._submatrix_for_ids(list(train_proto.keys()), np.stack([train_proto[k] for k in train_proto.keys()], axis=0), anchor_pool), dtype=np.float32))
    # The helper above depends on dict key order; rebuild deterministically to be safe.
    train_proto_ids = list(map(int, train_proto.keys()))
    train_proto_mat = np.stack([np.asarray(train_proto[int(k)], dtype=np.float32) for k in train_proto_ids], axis=0)
    v_anchor = _l2(np.asarray(a8._submatrix_for_ids(train_proto_ids, train_proto_mat, anchor_pool), dtype=np.float32))
    # Row-orthogonal ideal DINO-compatible basis in LLaMA4096 coordinates, same as A10C ideal-basis idea.
    try:
        basis = a10c._fit_row_orthogonal_embedding_basis(v_anchor, r_anchor)
    except Exception:
        # Fallback rectangular Procrustes if an older A10C helper lacks the named function.
        basis = a10c._fit_rectangular_procrustes(v_anchor, r_anchor)
    ideal_anchor = _l2(v_anchor @ np.asarray(basis, dtype=np.float32))
    fit = {
        "visible_csv": str(visible_csv),
        "anchor_pool_count": int(len(anchor_pool)),
        "anchor_recovery_cosine_mean": float(np.mean(np.sum(r_anchor * ideal_anchor, axis=1))),
        "anchor_recovery_cosine_median": float(np.median(np.sum(r_anchor * ideal_anchor, axis=1))),
    }
    return {"anchor_ids": anchor_pool, "anchor_idx": anchor_idx, "r_anchor": r_anchor, "ideal_anchor": ideal_anchor, "fit_base": fit}


def _diag_correction(all_r: np.ndarray, r_anchor: np.ndarray, ideal_anchor: np.ndarray, ridge: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    num = np.sum(r_anchor * ideal_anchor, axis=0)
    den = np.sum(r_anchor * r_anchor, axis=0) + float(ridge)
    scale = np.clip(num / np.maximum(den, 1e-12), -10.0, 10.0).astype(np.float32)
    out = _l2(all_r * scale[None, :])
    return out, {"scale_mean": float(np.mean(scale)), "scale_std": float(np.std(scale)), "scale_min": float(np.min(scale)), "scale_max": float(np.max(scale))}


def _lowrank_residual_correction(all_r: np.ndarray, r_anchor: np.ndarray, ideal_anchor: np.ndarray, rank: int, ridge: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    residual = np.asarray(ideal_anchor - r_anchor, dtype=np.float32)
    # PCA of residual targets. This avoids a 4096x4096 full model and caps expressivity.
    _u, _s, vt = np.linalg.svd(residual, full_matrices=False)
    r = int(max(1, min(rank, vt.shape[0])))
    comp = vt[:r].T.astype(np.float32)  # [D,r]
    y = residual @ comp                 # [N,r]
    k = r_anchor @ r_anchor.T + float(ridge) * np.eye(r_anchor.shape[0], dtype=np.float32)
    coef = np.linalg.solve(k, y).astype(np.float32)  # [N,r]
    pred_coef = (all_r @ r_anchor.T) @ coef
    pred_res = pred_coef @ comp.T
    out = _l2(all_r + pred_res.astype(np.float32))
    energy = float(np.sum(_s[:r] ** 2) / max(float(np.sum(_s ** 2)), 1e-12))
    return out, {"rank": r, "residual_energy_kept": energy, "ridge": float(ridge)}


def _whiten_recolor_correction(all_r: np.ndarray, r_anchor: np.ndarray, ideal_anchor: np.ndarray, rank: int, eps: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    mu_r = np.mean(r_anchor, axis=0, keepdims=True).astype(np.float32)
    mu_t = np.mean(ideal_anchor, axis=0, keepdims=True).astype(np.float32)
    rc = (r_anchor - mu_r).astype(np.float32)
    tc = (ideal_anchor - mu_t).astype(np.float32)
    ur, sr, vtr = np.linalg.svd(rc, full_matrices=False)
    ut, st, vtt = np.linalg.svd(tc, full_matrices=False)
    k = int(max(1, min(rank, vtr.shape[0], vtt.shape[0], int(np.sum(sr > eps)), int(np.sum(st > eps)))))
    vr = vtr[:k].T.astype(np.float32)
    vt = vtt[:k].T.astype(np.float32)
    # Whiten by source singular values, recolor by target singular values.
    z = ((all_r - mu_r) @ vr) / np.maximum(sr[:k][None, :], float(eps))
    rec = z * st[:k][None, :]
    out = _l2(mu_t + rec @ vt.T)
    return out.astype(np.float32), {"rank": k, "source_singular_min": float(sr[:k].min()), "target_singular_min": float(st[:k].min()), "eps": float(eps)}


def _anchor_metrics(corrected: np.ndarray, anchor_idx: Sequence[int], ideal_anchor: np.ndarray) -> Dict[str, Any]:
    ca = _l2(corrected[list(anchor_idx)])
    ia = _l2(ideal_anchor)
    cos = np.sum(ca * ia, axis=1)
    return {
        "anchor_to_ideal_cosine_mean": float(np.mean(cos)),
        "anchor_to_ideal_cosine_median": float(np.median(cos)),
        "anchor_to_ideal_cosine_min": float(np.min(cos)),
    }


def _write_bank(output_root: Path, correction_id: str, source_root: Path, ids: List[int], names: Dict[int, str], protos: np.ndarray, source_manifest: Mapping[str, Any], corr_meta: Mapping[str, Any]) -> Dict[str, Any]:
    sig = _sha256_text(json.dumps({"correction_id": correction_id, "source_root": str(source_root), "corr_meta": corr_meta}, ensure_ascii=False, sort_keys=True))[:12]
    bank_root = output_root / "text_banks" / f"a10g_{_safe_name(correction_id)}_{sig}"
    payload_dir = bank_root / "payload"
    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_path = payload_dir / "llama_hidden_mean.fp16.npz"
    np.savez_compressed(payload_path, protos=_l2(protos).astype(np.float16))
    class_rows = [{"raw_id": int(rid), "name": str(names.get(int(rid), f"raw_id_{rid}"))} for rid in ids]
    (bank_root / "lvvis_class_names.json").write_text(json.dumps({"classes": class_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "profile_id": f"a10g_{correction_id}",
        "profile_type": "visible525_text_side_correction",
        "source_profile_id": source_manifest.get("profile_id"),
        "source_root": str(source_root),
        "class_count": len(ids),
        "feature_dim": int(protos.shape[1]),
        "uses_old_corr_feats": False,
        "does_not_use_coco_class_list": True,
        "token_feature_alignment": "text_side_corrected_llama_hidden_mean",
        "correction": dict(corr_meta),
        "artifacts": {"llama_hidden_mean_path": str(payload_path), "llama_hidden_shape": list(protos.shape)},
    }
    _write_json(bank_root / "manifest.json", manifest)
    rec = {"correction_id": correction_id, "correction_hash": sig, "bank_root": str(bank_root), "payload_sha256": _sha256_file(payload_path)}
    rec.update({f"correction_{k}": v for k, v in corr_meta.items() if isinstance(v, (str, int, float, bool))})
    return rec


def _run_a10c_endpoint(*, repo_root: Path, asset_root: Path, run_root: Path, bank_root: Path, eval_root: Path, item_id: str, args: argparse.Namespace, log_path: Path) -> Dict[str, Any]:
    out = eval_root / _safe_name(item_id)
    cmd = [
        sys.executable, str(repo_root / "tools" / "a10c_run_llama4096_linear_isometric_distortion_calibration.py"),
        "--repo_root", str(repo_root),
        "--asset_root", str(asset_root),
        "--run_root", str(run_root),
        "--output_root", str(out),
        "--visual_only_root", str(bank_root),
        "--text_variant", "llama_hidden_mean",
        "--text_dim", "4096",
        "--alphas", str(args.endpoint_alphas),
        "--projectors", str(args.projectors),
        "--ridge_alpha", str(args.projector_ridge_alpha),
        "--test_scopes", str(args.test_scopes),
        "--candidate_scope", str(args.candidate_scope),
        "--seeds", str(args.eval_seeds),
        "--anchor_ratios", str(args.anchor_ratios),
        "--device", str(args.device),
        "--no_plots",
    ]
    if args.progress:
        cmd.append("--progress")
    rc = _run(cmd, cwd=repo_root, log_path=log_path, progress=bool(args.progress))
    rec = {"correction_id": item_id, "eval_output_root": str(out), "eval_status": "PASS" if rc == 0 else "FAIL", "eval_exit_code": rc}
    agg = out / "analysis" / "a10c_llama4096_alpha_aggregate.csv"
    if rc == 0 and agg.is_file():
        rec["aggregate_csv"] = str(agg)
    return rec


def _collect_rows(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        p = Path(str(rec.get("aggregate_csv", "")))
        if not p.is_file():
            continue
        for row in _read_csv(p):
            out = dict(row)
            out.update({k: rec[k] for k in ["correction_id", "correction_hash", "bank_root", "eval_output_root"] if k in rec})
            rows.append(out)
    return rows


def _ranking(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        cid = str(r.get("correction_id", ""))
        if not cid:
            continue
        prof = by.setdefault(cid, {"correction_id": cid, "correction_hash": r.get("correction_hash"), "bank_root": r.get("bank_root")})
        if abs(_safe_float(r.get("alpha")) - 1.0) > 1e-9:
            continue
        prefix = f"{r.get('test_scope')}_{r.get('projector')}"
        for src, dst in [("t2v_rank@1_mean", "t2v_at1"), ("t2v_rank@5_mean", "t2v_at5"), ("mean_normalized_rank_mean", "mean_normalized_rank"), ("spearman_Xalpha_vs_V_mean", "spearman"), ("knn_overlap@10_mean", "knn10"), ("triplet_agreement_mean", "triplet"), ("hubness_top1_concentration_mean", "hubness")]:
            if src in r:
                prof[f"{prefix}_{dst}"] = _safe_float(r.get(src))
    out: List[Dict[str, Any]] = []
    for cid, r in by.items():
        novel = _safe_float(r.get("novel_val_ridge_linear_t2v_at5"), 0.0)
        val = _safe_float(r.get("val_base_all_ridge_linear_t2v_at5"), 0.0)
        knn = _safe_float(r.get("novel_val_ridge_linear_knn10"), 0.0)
        trip = _safe_float(r.get("novel_val_ridge_linear_triplet"), 0.0)
        hub = _safe_float(r.get("novel_val_ridge_linear_hubness"), 0.0)
        rr = dict(r)
        rr["rank_score"] = 0.35 * novel + 0.20 * val + 0.20 * knn + 0.15 * trip - 0.10 * hub
        rr["primary_metric"] = "novel_val_ridge_linear_t2v_at5"
        rr["baseline_current_A10C_ridge_t2v_at5"] = BASELINE_RIDGE_T2V_AT5
        rr["delta_vs_current_baseline"] = novel - BASELINE_RIDGE_T2V_AT5
        rr["candidate_level"] = "EFFECTIVE" if novel >= 0.65 else ("WEAK_IMPROVEMENT" if novel > BASELINE_RIDGE_T2V_AT5 else "NO_ENDPOINT_GAIN")
        out.append(rr)
    return sorted(out, key=lambda x: _safe_float(x.get("rank_score"), -999.0), reverse=True)


def _parse_methods(spec: str) -> List[str]:
    return [x.strip() for x in str(spec).split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    repo = _repo_default()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo_root", default=str(repo))
    p.add_argument("--asset_root", default="/mnt/sda/zyy/code/wsovvis_asserts")
    p.add_argument("--run_root", default=str(repo / "codex/outputs/G8_inference_and_eval/sinkhorn_safeneg_k32_b010_preaug_k10_repro_20260427"))
    p.add_argument("--output_root", default=str(repo / "codex/outputs/G8_inference_and_eval/A10G_TEXT_SIDE_CORRECTION_ENDPOINT"))
    p.add_argument("--source_textbank_root", default="")
    p.add_argument("--methods", default="identity,diag,lowrank16,lowrank32,lowrank64,whiten")
    p.add_argument("--correction_ridge", type=float, default=1e-3)
    p.add_argument("--whiten_rank", type=int, default=256)
    p.add_argument("--whiten_eps", type=float, default=1e-4)
    p.add_argument("--visible_csv", default="")
    p.add_argument("--train_dataset_name", default="lvvis_train_base")
    p.add_argument("--train_annotation_json", default="")
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--endpoint_alphas", default="1.0")
    p.add_argument("--projectors", default="orthogonal_linear,ridge_linear")
    p.add_argument("--projector_ridge_alpha", type=float, default=0.01)
    p.add_argument("--test_scopes", default="novel_val,val_base_all")
    p.add_argument("--candidate_scope", default="full_available")
    p.add_argument("--eval_seeds", default="0")
    p.add_argument("--anchor_ratios", default="1.0")
    p.add_argument("--device", default="auto")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--continue_on_error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    analysis_root = output_root / "analysis"
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "run.log"
    source_root = Path(args.source_textbank_root).expanduser().resolve() if args.source_textbank_root else asset_root / "text_bank_llama3" / "lvvis" / "lvvis_visual_only_v1"
    result: Dict[str, Any] = {"status": "PASS", "start_time": time.strftime("%Y-%m-%d %H:%M:%S"), "repo_root": str(repo_root), "asset_root": str(asset_root), "run_root": str(run_root), "output_root": str(output_root), "analysis_root": str(analysis_root), "source_textbank_root": str(source_root)}
    manifest_rows: List[Dict[str, Any]] = []
    fit_rows: List[Dict[str, Any]] = []
    eval_records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    try:
        ids, names, real_mat, source_manifest = _load_bank(source_root)
        target = _load_visual_anchor_target(repo_root, asset_root, run_root, ids, real_mat, args)
        anchor_idx = target["anchor_idx"]
        r_anchor = target["r_anchor"]
        ideal_anchor = target["ideal_anchor"]
        methods = _parse_methods(args.methods)
        for method in methods:
            try:
                if args.progress:
                    print(f"[A10G] correction={method}", flush=True)
                method_l = method.lower()
                if method_l == "identity":
                    corr = real_mat.copy()
                    corr_meta = {"method": "identity"}
                elif method_l == "diag":
                    corr, corr_meta = _diag_correction(real_mat, r_anchor, ideal_anchor, ridge=float(args.correction_ridge))
                    corr_meta.update({"method": "diag", "ridge": float(args.correction_ridge)})
                elif method_l.startswith("lowrank"):
                    rank_txt = method_l.replace("lowrank", "")
                    rank = int(rank_txt) if rank_txt else 32
                    corr, corr_meta = _lowrank_residual_correction(real_mat, r_anchor, ideal_anchor, rank=rank, ridge=float(args.correction_ridge))
                    corr_meta.update({"method": "lowrank_residual"})
                elif method_l == "whiten":
                    corr, corr_meta = _whiten_recolor_correction(real_mat, r_anchor, ideal_anchor, rank=int(args.whiten_rank), eps=float(args.whiten_eps))
                    corr_meta.update({"method": "whiten_recolor"})
                else:
                    raise ValueError(f"unsupported correction method: {method}")
                fit = {"correction_id": method, **target["fit_base"], **_anchor_metrics(corr, anchor_idx, ideal_anchor)}
                fit.update({f"correction_{k}": v for k, v in corr_meta.items() if isinstance(v, (str, int, float, bool))})
                fit_rows.append(fit)
                if method_l == "identity":
                    # Reuse source bank for identity to avoid duplicate assets.
                    rec = {"correction_id": method, "correction_hash": "source", "bank_root": str(source_root), "payload_sha256": ""}
                else:
                    rec = _write_bank(output_root, method, source_root, ids, names, corr, source_manifest, corr_meta)
                manifest_rows.append(rec)
                if not args.skip_eval:
                    ev = _run_a10c_endpoint(repo_root=repo_root, asset_root=asset_root, run_root=run_root, bank_root=Path(rec["bank_root"]), eval_root=output_root / "eval", item_id=method, args=args, log_path=log_path)
                    ev.update(rec)
                    eval_records.append(ev)
                    if ev.get("eval_status") != "PASS":
                        failures.append({"correction_id": method, "stage": "eval", "status": ev.get("eval_status")})
                        if not args.continue_on_error:
                            raise RuntimeError(f"A10G eval failed for {method}: {ev}")
            except Exception as exc:
                failures.append({"correction_id": method, "stage": "build_or_eval", "status": "FAIL", "reason": repr(exc)})
                if not args.continue_on_error:
                    raise
        rows = _collect_rows(eval_records)
        ranking = _ranking(rows)
        _write_csv(analysis_root / "correction_manifest.csv", manifest_rows)
        _write_csv(analysis_root / "correction_fit_metrics.csv", fit_rows)
        _write_csv(analysis_root / "correction_endpoint_eval_runs.csv", eval_records)
        _write_csv(analysis_root / "correction_endpoint_eval.csv", rows)
        _write_csv(analysis_root / "correction_ranking_summary.csv", ranking)
        result.update({"method_count": len(methods), "manifest_rows": len(manifest_rows), "fit_rows": len(fit_rows), "endpoint_eval_run_rows": len(eval_records), "endpoint_eval_rows": len(rows), "ranking_rows": len(ranking), "failures": failures, "artifacts": {"correction_manifest_csv": str(analysis_root / "correction_manifest.csv"), "correction_fit_metrics_csv": str(analysis_root / "correction_fit_metrics.csv"), "correction_endpoint_eval_csv": str(analysis_root / "correction_endpoint_eval.csv"), "correction_ranking_summary_csv": str(analysis_root / "correction_ranking_summary.csv")}})
        if failures:
            result["status"] = "PARTIAL_FAIL" if args.continue_on_error else "FAIL"
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = repr(exc)
        result["failures"] = failures
        _write_json(analysis_root / "A10G_run_result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 1
    _write_json(analysis_root / "A10G_run_result.json", result)
    (output_root / "TAKEOVER_A10G_TEXT_SIDE_CORRECTION_ENDPOINT.md").write_text("\n".join([
        "# A10G Text-Side Correction Endpoint TAKEOVER", "", f"- status: `{result.get('status')}`", f"- method_count: `{result.get('method_count')}`", f"- endpoint_eval_rows: `{result.get('endpoint_eval_rows')}`", "", "## Key artifacts", "- analysis/correction_fit_metrics.csv", "- analysis/correction_ranking_summary.csv", "- analysis/correction_endpoint_eval.csv", "", "## Scope", "- Correction fitting uses visible525 train anchors only.", "- Corrected features are written as side-output text banks under output_root/text_banks.", "- Final projector is still trained inside A10C with visible525 anchors.", "- No formal training/inference path is mutated.", ""]), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
