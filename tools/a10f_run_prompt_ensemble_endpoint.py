#!/usr/bin/env python3
"""A10F prompt-feature ensemble endpoint probe for WS-OVVIS.

Read-only/side-output diagnostic. It reuses A10E-generated LLaMA4096 text banks,
constructs normalized prompt-feature ensembles, then calls the existing A10C GPU
endpoint evaluator. No LLaMA generation, no row-level iteration, no formal G7/G8
pipeline mutation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import shutil
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


def _run(cmd: Sequence[str], *, cwd: Path, log_path: Path, progress: bool) -> int:
    if progress:
        print("[A10F][cmd] " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write("\n[A10F][cmd] " + " ".join(shlex.quote(str(x)) for x in cmd) + "\n")
        log_f.flush()
        proc = subprocess.Popen(list(map(str, cmd)), cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            if progress:
                print(line, end="", flush=True)
            log_f.write(line)
            log_f.flush()
        rc = int(proc.wait())
        log_f.write(f"[A10F][exit] {rc}\n")
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
    classes_obj = _read_json(class_path)
    rows = classes_obj.get("classes", classes_obj) if isinstance(classes_obj, Mapping) else classes_obj
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


def _discover_from_a10e(a10e_root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    p = a10e_root / "analysis" / "prompt_profile_manifest.csv"
    if not p.is_file():
        return out
    for row in _read_csv(p):
        pid = str(row.get("profile_id", ""))
        bank = str(row.get("bank_root", ""))
        if not pid or not bank:
            continue
        root = Path(bank).expanduser().resolve()
        if root.is_dir():
            out[pid] = root
    return out


def _resolve_roots(args: argparse.Namespace, asset_root: Path, output_root: Path) -> Dict[str, Path]:
    roots: Dict[str, Path] = {}
    if args.a10e_root:
        roots.update(_discover_from_a10e(Path(args.a10e_root).expanduser().resolve()))
    roots.setdefault("a10e_P0_baseline_visual_only_v1", Path(args.p0_root).expanduser().resolve() if args.p0_root else asset_root / "text_bank_llama3" / "lvvis" / "lvvis_visual_only_v1")

    # Add short aliases used in ensemble specs.
    aliases: Dict[str, str] = {
        "P0": "a10e_P0_baseline_visual_only_v1",
        "P1": "a10e_P1_crop_morphology",
        "P2": "a10e_P2_local_global",
        "P3": "a10e_P3_discriminative",
        "P4": "a10e_P4_shape_first",
        "P5": "a10e_P5_texture_material",
        "P6": "a10e_P6_balanced_metric",
    }
    for short, full in aliases.items():
        if full in roots:
            roots[short] = roots[full]

    # Fallback glob discovery for P3/P6 if A10E manifest is absent.
    base = asset_root / "text_bank_llama3" / "lvvis"
    patterns = {
        "P1": "a10e_a10e_P1_crop_morphology_*",
        "P2": "a10e_a10e_P2_local_global_*",
        "P3": "a10e_a10e_P3_discriminative_*",
        "P4": "a10e_a10e_P4_shape_first_*",
        "P5": "a10e_a10e_P5_texture_material_*",
        "P6": "a10e_a10e_P6_balanced_metric_*",
    }
    for key, pat in patterns.items():
        if key not in roots:
            hits = sorted([p for p in base.glob(pat) if p.is_dir()])
            if hits:
                roots[key] = hits[-1]
    return roots


def _parse_ensembles(spec: str) -> List[Tuple[str, List[Tuple[str, float]]]]:
    out: List[Tuple[str, List[Tuple[str, float]]]] = []
    for item in str(spec).split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"bad ensemble item, expected NAME=KEY:W,KEY:W: {item}")
        name, rhs = item.split("=", 1)
        comps: List[Tuple[str, float]] = []
        for part in rhs.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"bad component {part} in {item}")
            key, w = part.split(":", 1)
            comps.append((key.strip(), float(w)))
        if not comps:
            raise ValueError(f"empty ensemble: {item}")
        out.append((_safe_name(name), comps))
    return out


def _write_ensemble_bank(out_root: Path, ens_name: str, comps: List[Tuple[str, float]], roots: Dict[str, Path]) -> Dict[str, Any]:
    loaded = []
    ref_ids: Optional[List[int]] = None
    ref_names: Optional[Dict[int, str]] = None
    accum: Optional[np.ndarray] = None
    total_abs = 0.0
    comp_meta: List[Dict[str, Any]] = []
    for key, weight in comps:
        if key not in roots:
            raise KeyError(f"ensemble {ens_name}: missing bank root for component {key}; available={sorted(roots)}")
        ids, names, arr, manifest = _load_bank(roots[key])
        if ref_ids is None:
            ref_ids = ids
            ref_names = names
            accum = np.zeros_like(arr, dtype=np.float32)
        elif ids != ref_ids:
            raise RuntimeError(f"ensemble {ens_name}: class order mismatch for {key}")
        assert accum is not None
        accum += float(weight) * arr
        total_abs += abs(float(weight))
        comp_meta.append({"key": key, "weight": float(weight), "root": str(roots[key]), "profile_id": manifest.get("profile_id")})
        loaded.append(key)
    assert ref_ids is not None and ref_names is not None and accum is not None
    protos = _l2(accum).astype(np.float16)
    sig = _sha256_text(json.dumps(comp_meta, ensure_ascii=False, sort_keys=True))[:12]
    bank_root = out_root / "text_banks" / f"a10f_{ens_name}_{sig}"
    payload_dir = bank_root / "payload"
    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_path = payload_dir / "llama_hidden_mean.fp16.npz"
    np.savez_compressed(payload_path, protos=protos)
    class_rows = [{"raw_id": int(rid), "name": str(ref_names.get(int(rid), f"raw_id_{rid}"))} for rid in ref_ids]
    (bank_root / "lvvis_class_names.json").write_text(json.dumps({"classes": class_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "profile_id": f"a10f_{ens_name}",
        "profile_type": "prompt_feature_ensemble",
        "class_count": len(ref_ids),
        "feature_dim": int(protos.shape[1]),
        "uses_old_corr_feats": False,
        "does_not_use_coco_class_list": True,
        "token_feature_alignment": "ensemble_of_exact_full_forward_generated_token_or_class_span_slice",
        "ensemble_components": comp_meta,
        "artifacts": {"llama_hidden_mean_path": str(payload_path), "llama_hidden_shape": list(protos.shape)},
    }
    _write_json(bank_root / "manifest.json", manifest)
    return {
        "ensemble_id": ens_name,
        "ensemble_hash": sig,
        "bank_root": str(bank_root),
        "component_keys": ",".join(loaded),
        "component_spec": json.dumps(comp_meta, ensure_ascii=False),
        "payload_sha256": _sha256_file(payload_path),
    }


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
        "--ridge_alpha", str(args.ridge_alpha),
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
    rec = {"ensemble_id": item_id, "eval_output_root": str(out), "eval_status": "PASS" if rc == 0 else "FAIL", "eval_exit_code": rc}
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
            out.update({k: rec[k] for k in ["ensemble_id", "ensemble_hash", "bank_root", "component_keys", "component_spec", "eval_output_root"] if k in rec})
            rows.append(out)
    return rows


def _ranking(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        eid = str(r.get("ensemble_id", ""))
        if not eid:
            continue
        prof = by.setdefault(eid, {"ensemble_id": eid, "ensemble_hash": r.get("ensemble_hash"), "bank_root": r.get("bank_root"), "component_keys": r.get("component_keys")})
        if abs(_safe_float(r.get("alpha")) - 1.0) > 1e-9:
            continue
        prefix = f"{r.get('test_scope')}_{r.get('projector')}"
        for src, dst in [("t2v_rank@1_mean", "t2v_at1"), ("t2v_rank@5_mean", "t2v_at5"), ("mean_normalized_rank_mean", "mean_normalized_rank"), ("spearman_Xalpha_vs_V_mean", "spearman"), ("knn_overlap@10_mean", "knn10"), ("triplet_agreement_mean", "triplet"), ("hubness_top1_concentration_mean", "hubness")]:
            if src in r:
                prof[f"{prefix}_{dst}"] = _safe_float(r.get(src))
    out: List[Dict[str, Any]] = []
    for eid, r in by.items():
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


def parse_args() -> argparse.Namespace:
    repo = _repo_default()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo_root", default=str(repo))
    p.add_argument("--asset_root", default="/mnt/sda/zyy/code/wsovvis_asserts")
    p.add_argument("--run_root", default=str(repo / "codex/outputs/G8_inference_and_eval/sinkhorn_safeneg_k32_b010_preaug_k10_repro_20260427"))
    p.add_argument("--output_root", default=str(repo / "codex/outputs/G8_inference_and_eval/A10F_PROMPT_ENSEMBLE_ENDPOINT"))
    p.add_argument("--a10e_root", default=str(repo / "codex/outputs/G8_inference_and_eval/A10E_LLAMA_PROMPT_MANIFOLD_SEARCH"))
    p.add_argument("--p0_root", default="")
    p.add_argument("--ensembles", default="E0=P0:1.0;E1=P0:0.5,P3:0.5;E2=P0:0.5,P6:0.5;E3=P0:0.333333,P3:0.333333,P6:0.333334;E4=P0:0.7,P3:0.3;E5=P0:0.7,P6:0.3;E6=P0:0.6,P3:0.2,P6:0.2;E7=P0:0.5,P3:0.3,P6:0.2")
    p.add_argument("--endpoint_alphas", default="1.0")
    p.add_argument("--projectors", default="orthogonal_linear,ridge_linear")
    p.add_argument("--ridge_alpha", type=float, default=0.01)
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
    result: Dict[str, Any] = {"status": "PASS", "start_time": time.strftime("%Y-%m-%d %H:%M:%S"), "repo_root": str(repo_root), "asset_root": str(asset_root), "run_root": str(run_root), "output_root": str(output_root), "analysis_root": str(analysis_root)}
    manifest_rows: List[Dict[str, Any]] = []
    eval_records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    try:
        roots = _resolve_roots(args, asset_root, output_root)
        ensembles = _parse_ensembles(args.ensembles)
        for ens_name, comps in ensembles:
            try:
                if args.progress:
                    print(f"[A10F] ensemble={ens_name} components={comps}", flush=True)
                rec = _write_ensemble_bank(output_root, ens_name, comps, roots)
                manifest_rows.append(rec)
                if not args.skip_eval:
                    ev = _run_a10c_endpoint(repo_root=repo_root, asset_root=asset_root, run_root=run_root, bank_root=Path(rec["bank_root"]), eval_root=output_root / "eval", item_id=ens_name, args=args, log_path=log_path)
                    ev.update(rec)
                    eval_records.append(ev)
                    if ev.get("eval_status") != "PASS":
                        failures.append({"ensemble_id": ens_name, "stage": "eval", "status": ev.get("eval_status")})
                        if not args.continue_on_error:
                            raise RuntimeError(f"A10F eval failed for {ens_name}: {ev}")
            except Exception as exc:
                failures.append({"ensemble_id": ens_name, "stage": "build_or_eval", "status": "FAIL", "reason": repr(exc)})
                if not args.continue_on_error:
                    raise
        rows = _collect_rows(eval_records)
        ranking = _ranking(rows)
        _write_csv(analysis_root / "ensemble_manifest.csv", manifest_rows)
        _write_csv(analysis_root / "ensemble_endpoint_eval_runs.csv", eval_records)
        _write_csv(analysis_root / "ensemble_endpoint_eval.csv", rows)
        _write_csv(analysis_root / "ensemble_ranking_summary.csv", ranking)
        result.update({"ensemble_count": len(ensembles), "manifest_rows": len(manifest_rows), "endpoint_eval_run_rows": len(eval_records), "endpoint_eval_rows": len(rows), "ranking_rows": len(ranking), "failures": failures, "artifacts": {"ensemble_manifest_csv": str(analysis_root / "ensemble_manifest.csv"), "ensemble_endpoint_eval_csv": str(analysis_root / "ensemble_endpoint_eval.csv"), "ensemble_ranking_summary_csv": str(analysis_root / "ensemble_ranking_summary.csv")}})
        if failures:
            result["status"] = "PARTIAL_FAIL" if args.continue_on_error else "FAIL"
    except Exception as exc:
        result["status"] = "FAIL"
        result["error"] = repr(exc)
        result["failures"] = failures
        _write_json(analysis_root / "A10F_run_result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 1
    _write_json(analysis_root / "A10F_run_result.json", result)
    (output_root / "TAKEOVER_A10F_PROMPT_ENSEMBLE_ENDPOINT.md").write_text("\n".join([
        "# A10F Prompt Ensemble Endpoint TAKEOVER", "", f"- status: `{result.get('status')}`", f"- ensemble_count: `{result.get('ensemble_count')}`", f"- endpoint_eval_rows: `{result.get('endpoint_eval_rows')}`", "", "## Key artifacts", f"- analysis/ensemble_ranking_summary.csv", f"- analysis/ensemble_endpoint_eval.csv", "", "## Scope", "- Reuses existing A10E text banks; no LLaMA generation.", "- Writes only generated ensemble text banks under output_root/text_banks and eval tables under output_root/analysis.", "- Final projector is still trained inside A10C with visible525 anchors.", ""]), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
