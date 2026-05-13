#!/usr/bin/env python3
"""A10E LLaMA prompt manifold search orchestrator.

Configuration-driven, read-mostly prompt search for WS-OVVIS text-side manifold
alignment.  It reuses the existing LLaMA text-bank builder and the A10C GPU
linear-isometric endpoint evaluator instead of reimplementing feature extraction
or class-prototype evaluation.

Default mode:
  1. read prompt profiles from configs/a10e_llama_prompt_profiles.json;
  2. build/cache one LLaMA4096 text bank per enabled/profile-selected prompt;
  3. validate each bank with tools/check_lvvis_text_bank_alignment.py;
  4. run A10C endpoint-only eval with alpha=1.0 for each bank;
  5. collect prompt ranking tables under --output_root/analysis.

It does not mutate formal G7/G8 training/inference paths and does not overwrite
existing lvvis_visual_only_v1 unless explicitly asked to regenerate the baseline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _repo_default() -> Path:
    return Path.cwd().resolve()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


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
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if v == v and abs(v) != float("inf") else default
    except Exception:
        return default


def _safe_name(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "profile"


def _profile_hash(profile: Mapping[str, Any]) -> str:
    payload = {
        "profile_id": profile.get("profile_id"),
        "profile_type": profile.get("profile_type"),
        "system_prompt": profile.get("system_prompt"),
        "system_prompt_template": profile.get("system_prompt_template"),
        "user_prompts": profile.get("user_prompts"),
        "generation_defaults": profile.get("generation_defaults"),
        "class_placeholder_style": profile.get("class_placeholder_style"),
    }
    return _sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))[:12]


def _run(cmd: Sequence[str], *, cwd: Path, log_path: Optional[Path], progress: bool) -> Tuple[int, str]:
    if progress:
        print("[A10E][cmd] " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    log_f = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = log_path.open("a", encoding="utf-8")
        log_f.write("\n[A10E][cmd] " + " ".join(shlex.quote(str(x)) for x in cmd) + "\n")
        log_f.flush()
    proc = subprocess.Popen(
        list(map(str, cmd)),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        captured.append(line)
        if progress:
            print(line, end="", flush=True)
        if log_f is not None:
            log_f.write(line)
            log_f.flush()
    rc = proc.wait()
    if log_f is not None:
        log_f.write(f"[A10E][exit] {rc}\n")
        log_f.close()
    return int(rc), "".join(captured)


def _bank_ready(bank_root: Path, *, allow_smoke: bool, expected_dim: int) -> Tuple[bool, str]:
    manifest = bank_root / "manifest.json"
    payload = bank_root / "payload" / "llama_hidden_mean.fp16.npz"
    if not manifest.is_file():
        return False, "missing manifest.json"
    if not payload.is_file():
        return False, "missing payload/llama_hidden_mean.fp16.npz"
    try:
        m = _read_json(manifest)
        cls_count = int(m.get("class_count", 0))
        shape = m.get("artifacts", {}).get("llama_hidden_shape")
        if shape and int(shape[1]) != int(expected_dim):
            return False, f"dim mismatch in manifest: {shape}"
        if not allow_smoke and cls_count != 1196:
            return False, f"class_count={cls_count} != 1196"
        return True, "ready"
    except Exception as e:
        return False, f"manifest parse/check failed: {e}"


def _select_profiles(config: Mapping[str, Any], profile_ids: str) -> List[Dict[str, Any]]:
    profiles = dict(config.get("profiles", {}))
    if profile_ids.strip():
        wanted = [x.strip() for x in profile_ids.split(",") if x.strip()]
    else:
        wanted = [pid for pid, p in profiles.items() if bool(p.get("enabled", True))]
    missing = [pid for pid in wanted if pid not in profiles]
    if missing:
        raise KeyError(f"missing profiles in config: {missing}; available={sorted(profiles)}")
    return [dict(profiles[pid]) for pid in wanted]


def _build_or_reuse_bank(
    *,
    repo_root: Path,
    asset_root: Path,
    output_root: Path,
    textbank_root: Path,
    prompt_profiles_path: Path,
    profile: Mapping[str, Any],
    args: argparse.Namespace,
    log_path: Path,
) -> Dict[str, Any]:
    pid = str(profile["profile_id"])
    ph = _profile_hash(profile)
    output_name = f"a10e_{_safe_name(pid)}_{ph}"
    bank_root = textbank_root / output_name
    baseline_root = Path(args.baseline_textbank_root).expanduser().resolve() if args.baseline_textbank_root else (asset_root / "text_bank_llama3" / "lvvis" / "lvvis_visual_only_v1")
    is_p0 = pid.startswith("a10e_P0") or pid == "lvvis_visual_only_v1"
    if is_p0 and bool(args.use_existing_baseline) and not bool(args.regenerate_baseline) and baseline_root.is_dir():
        ready, reason = _bank_ready(baseline_root, allow_smoke=bool(args.smoke), expected_dim=int(args.text_dim))
        if ready:
            return {
                "profile_id": pid,
                "profile_hash": ph,
                "bank_root": str(baseline_root),
                "bank_output_name": baseline_root.name,
                "build_status": "REUSED_BASELINE",
                "build_reason": reason,
            }
    ready, reason = _bank_ready(bank_root, allow_smoke=bool(args.smoke), expected_dim=int(args.text_dim))
    if ready and not bool(args.rebuild_text_banks):
        return {"profile_id": pid, "profile_hash": ph, "bank_root": str(bank_root), "bank_output_name": output_name, "build_status": "CACHE_HIT", "build_reason": reason}
    if bool(args.skip_generation):
        return {"profile_id": pid, "profile_hash": ph, "bank_root": str(bank_root), "bank_output_name": output_name, "build_status": "MISSING_SKIPPED", "build_reason": reason}
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "build_lvvis_llama3_text_bank.py"),
        "--repo_root", str(repo_root),
        "--assert_root", str(asset_root),
        "--output_root", str(textbank_root),
        "--output_name", output_name,
        "--profile", pid,
        "--prompt_profiles_path", str(prompt_profiles_path),
        "--ckpt_dir", str(args.ckpt_dir),
        "--tokenizer_path", str(args.tokenizer_path),
        "--max_seq_len", str(args.max_seq_len),
        "--max_batch_size", str(args.max_batch_size),
        "--master_port", str(args.master_port),
        "--local_rank", str(args.local_rank),
        "--seed", str(args.llama_seed),
        "--no-build_clip_of_llm",
        "--build_llama_hidden",
        "--print_progress",
        "--log_every_classes", str(args.log_every_classes),
    ]
    if int(args.max_classes) > 0:
        cmd += ["--max_classes", str(args.max_classes)]
    if bool(args.rebuild_text_banks) or bank_root.exists():
        cmd += ["--overwrite"]
    rc, _ = _run(cmd, cwd=repo_root, log_path=log_path, progress=bool(args.progress))
    return {"profile_id": pid, "profile_hash": ph, "bank_root": str(bank_root), "bank_output_name": output_name, "build_status": "PASS" if rc == 0 else "FAIL", "build_exit_code": rc}


def _selfcheck_bank(*, repo_root: Path, bank_root: Path, args: argparse.Namespace, log_path: Path) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "check_lvvis_text_bank_alignment.py"),
        "--bank_root", str(bank_root),
        "--expect_class_count", "1196",
    ]
    if bool(args.smoke) or int(args.max_classes) > 0:
        cmd += ["--allow_smoke"]
    rc, out = _run(cmd, cwd=repo_root, log_path=log_path, progress=bool(args.progress))
    rec: Dict[str, Any] = {"bank_root": str(bank_root), "selfcheck_status": "PASS" if rc == 0 else "FAIL", "selfcheck_exit_code": rc}
    try:
        # Pull the last JSON object printed by the checker.
        start = out.rfind("{")
        if start >= 0:
            parsed = json.loads(out[start:])
            rec.update({"selfcheck_profile_id": parsed.get("profile_id"), "selfcheck_class_count": parsed.get("class_count")})
    except Exception:
        pass
    return rec


def _run_a10c_endpoint(*, repo_root: Path, asset_root: Path, run_root: Path, bank_root: Path, eval_root: Path, profile_id: str, args: argparse.Namespace, log_path: Path) -> Dict[str, Any]:
    eval_out = eval_root / _safe_name(profile_id)
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "a10c_run_llama4096_linear_isometric_distortion_calibration.py"),
        "--repo_root", str(repo_root),
        "--asset_root", str(asset_root),
        "--run_root", str(run_root),
        "--output_root", str(eval_out),
        "--visual_only_root", str(bank_root),
        "--text_variant", "llama_hidden_mean",
        "--text_dim", str(args.text_dim),
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
    if bool(args.progress):
        cmd += ["--progress"]
    rc, _ = _run(cmd, cwd=repo_root, log_path=log_path, progress=bool(args.progress))
    rec: Dict[str, Any] = {"profile_id": profile_id, "eval_output_root": str(eval_out), "eval_status": "PASS" if rc == 0 else "FAIL", "eval_exit_code": rc}
    agg = eval_out / "analysis" / "a10c_llama4096_alpha_aggregate.csv"
    if rc == 0 and agg.is_file():
        rec["aggregate_csv"] = str(agg)
    return rec


def _collect_endpoint_rows(profile_records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pr in profile_records:
        agg = pr.get("aggregate_csv")
        if not agg:
            continue
        p = Path(str(agg))
        if not p.is_file():
            continue
        for row in _read_csv(p):
            out = dict(row)
            for k in ["profile_id", "profile_hash", "bank_root", "bank_output_name", "build_status", "selfcheck_status", "eval_output_root"]:
                if k in pr:
                    out[k] = pr[k]
            rows.append(out)
    return rows


def _ranking(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    # One compact profile-level row. Prefer alpha=1.0 endpoint and ridge novel score.
    by_profile: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        pid = str(r.get("profile_id", ""))
        if not pid:
            continue
        prof = by_profile.setdefault(pid, {"profile_id": pid, "profile_hash": r.get("profile_hash"), "bank_root": r.get("bank_root")})
        alpha = _safe_float(r.get("alpha"))
        if abs(alpha - 1.0) > 1e-9:
            continue
        projector = str(r.get("projector"))
        scope = str(r.get("test_scope"))
        prefix = f"{scope}_{projector}"
        for src, dst in [
            ("t2v_rank@1_mean", "t2v_at1"),
            ("t2v_rank@5_mean", "t2v_at5"),
            ("mean_normalized_rank_mean", "mean_normalized_rank"),
            ("spearman_Xalpha_vs_V_mean", "spearman"),
            ("knn_overlap@10_mean", "knn10"),
            ("triplet_agreement_mean", "triplet"),
            ("hubness_top1_concentration_mean", "hubness"),
        ]:
            if src in r:
                prof[f"{prefix}_{dst}"] = _safe_float(r.get(src))
    ranked: List[Dict[str, Any]] = []
    for pid, r in by_profile.items():
        novel_ridge_t5 = _safe_float(r.get("novel_val_ridge_linear_t2v_at5"), 0.0)
        val_ridge_t5 = _safe_float(r.get("val_base_all_ridge_linear_t2v_at5"), 0.0)
        knn = _safe_float(r.get("novel_val_ridge_linear_knn10"), _safe_float(r.get("novel_val_orthogonal_linear_knn10"), 0.0))
        trip = _safe_float(r.get("novel_val_ridge_linear_triplet"), _safe_float(r.get("novel_val_orthogonal_linear_triplet"), 0.0))
        hub = _safe_float(r.get("novel_val_ridge_linear_hubness"), _safe_float(r.get("novel_val_orthogonal_linear_hubness"), 0.0))
        score = 0.35 * novel_ridge_t5 + 0.20 * val_ridge_t5 + 0.20 * knn + 0.15 * trip - 0.10 * hub
        rr = dict(r)
        rr["rank_score"] = score
        rr["primary_metric"] = "novel_val_ridge_linear_t2v_at5"
        rr["baseline_current_A10C_ridge_t2v_at5"] = 0.533156
        rr["delta_vs_current_baseline"] = novel_ridge_t5 - 0.533156
        if novel_ridge_t5 >= 0.80:
            rr["candidate_level"] = "USABLE_LEVEL"
        elif novel_ridge_t5 >= 0.75:
            rr["candidate_level"] = "STRONG"
        elif novel_ridge_t5 >= 0.65:
            rr["candidate_level"] = "EFFECTIVE"
        elif novel_ridge_t5 > 0.533156:
            rr["candidate_level"] = "WEAK_IMPROVEMENT"
        else:
            rr["candidate_level"] = "NO_ENDPOINT_GAIN"
        ranked.append(rr)
    return sorted(ranked, key=lambda x: _safe_float(x.get("rank_score"), -999.0), reverse=True)


def _write_takeover(output_root: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# A10E LLaMA Prompt Manifold Search TAKEOVER",
        "",
        f"- status: `{result.get('status')}`",
        f"- output_root: `{result.get('output_root')}`",
        f"- analysis_root: `{result.get('analysis_root')}`",
        f"- prompt_profiles_path: `{result.get('prompt_profiles_path')}`",
        "",
        "## Scope",
        "- Configuration-driven prompt search; add new profiles in `configs/a10e_llama_prompt_profiles.json` without changing code.",
        "- Reuses `tools/build_lvvis_llama3_text_bank.py` for LLaMA4096 hidden feature extraction.",
        "- Reuses `tools/a10c_run_llama4096_linear_isometric_distortion_calibration.py` for endpoint class-prototype evaluation.",
        "- Does not run row-level or G8 segmentation eval.",
        "- Does not overwrite `lvvis_visual_only_v1`; P0 can reuse it as baseline.",
        "",
        "## Key artifacts",
        "- `analysis/prompt_profile_manifest.csv`",
        "- `analysis/prompt_textbank_selfcheck.csv`",
        "- `analysis/prompt_endpoint_eval.csv`",
        "- `analysis/prompt_ranking_summary.csv`",
        "- `analysis/A10E_run_result.json`",
    ]
    (output_root / "A10E_takeover.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    repo = _repo_default()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo_root", default=str(repo))
    p.add_argument("--asset_root", default="/mnt/sda/zyy/code/wsovvis_asserts")
    p.add_argument("--run_root", default="/mnt/sda/zyy/code/wsovvis/codex/outputs/G8_inference_and_eval/sinkhorn_safeneg_k32_b010_preaug_k10_repro_20260427")
    p.add_argument("--output_root", default="/mnt/sda/zyy/code/wsovvis/codex/outputs/G8_inference_and_eval/A10E_LLAMA_PROMPT_MANIFOLD_SEARCH")
    p.add_argument("--prompt_profiles_path", default="configs/a10e_llama_prompt_profiles.json")
    p.add_argument("--profile_ids", default="", help="comma-separated profile ids; default: all enabled profiles in config")
    p.add_argument("--textbank_root", default="", help="default: <asset_root>/text_bank_llama3/lvvis")
    p.add_argument("--baseline_textbank_root", default="", help="default: <asset_root>/text_bank_llama3/lvvis/lvvis_visual_only_v1")
    p.add_argument("--use_existing_baseline", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--regenerate_baseline", action="store_true")
    p.add_argument("--skip_generation", action="store_true", help="do not run LLaMA builder; only evaluate existing cached banks")
    p.add_argument("--rebuild_text_banks", action="store_true", help="overwrite/rebuild A10E generated banks")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--smoke", action="store_true", help="build a small class subset and skip endpoint eval unless --force_smoke_eval is set")
    p.add_argument("--force_smoke_eval", action="store_true")
    p.add_argument("--max_classes", type=int, default=0)
    # LLaMA builder args.
    p.add_argument("--ckpt_dir", default="Meta-Llama-3-8B-Instruct")
    p.add_argument("--tokenizer_path", default="Meta-Llama-3-8B-Instruct/tokenizer.model")
    p.add_argument("--max_seq_len", type=int, default=384)
    p.add_argument("--max_batch_size", type=int, default=64)
    p.add_argument("--master_port", default="56789")
    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("--llama_seed", type=int, default=2024)
    p.add_argument("--log_every_classes", type=int, default=20)
    # A10C endpoint eval args.
    p.add_argument("--text_dim", type=int, default=4096)
    p.add_argument("--endpoint_alphas", default="1.0")
    p.add_argument("--projectors", default="orthogonal_linear,ridge_linear")
    p.add_argument("--ridge_alpha", type=float, default=0.01)
    p.add_argument("--test_scopes", default="novel_val,val_base_all")
    p.add_argument("--candidate_scope", default="full_available")
    p.add_argument("--eval_seeds", default="0")
    p.add_argument("--anchor_ratios", default="1.0")
    p.add_argument("--device", default="auto")
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
    logs_root = output_root / "logs"
    eval_root = output_root / "eval"
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    if bool(args.smoke) and int(args.max_classes) <= 0:
        args.max_classes = 20
    prompt_profiles_path = Path(args.prompt_profiles_path)
    if not prompt_profiles_path.is_absolute():
        prompt_profiles_path = repo_root / prompt_profiles_path
    textbank_root = Path(args.textbank_root).expanduser().resolve() if args.textbank_root else (asset_root / "text_bank_llama3" / "lvvis")
    config = _read_json(prompt_profiles_path)
    profiles = _select_profiles(config, str(args.profile_ids))
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    result: Dict[str, Any] = {
        "status": "PASS",
        "start_time": started,
        "repo_root": str(repo_root),
        "asset_root": str(asset_root),
        "run_root": str(run_root),
        "output_root": str(output_root),
        "analysis_root": str(analysis_root),
        "prompt_profiles_path": str(prompt_profiles_path),
        "profile_count": len(profiles),
        "profiles": [p.get("profile_id") for p in profiles],
    }
    manifest_rows: List[Dict[str, Any]] = []
    selfcheck_rows: List[Dict[str, Any]] = []
    eval_records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    try:
        for profile in profiles:
            pid = str(profile["profile_id"])
            if args.progress:
                print(f"[A10E] profile={pid} start", flush=True)
            log_path = logs_root / f"{_safe_name(pid)}.log"
            build = _build_or_reuse_bank(
                repo_root=repo_root,
                asset_root=asset_root,
                output_root=output_root,
                textbank_root=textbank_root,
                prompt_profiles_path=prompt_profiles_path,
                profile=profile,
                args=args,
                log_path=log_path,
            )
            build.update({
                "description": profile.get("description", ""),
                "target_hypothesis": profile.get("a10e_metadata", {}).get("target_hypothesis", ""),
                "prompt_profile_config_hash": _profile_hash(profile),
            })
            manifest_rows.append(build)
            if str(build.get("build_status")) in {"FAIL", "MISSING_SKIPPED"}:
                failures.append({"profile_id": pid, "stage": "build", "status": build.get("build_status"), "reason": build.get("build_reason")})
                if not args.continue_on_error:
                    raise RuntimeError(f"profile {pid} build/cache failed: {build}")
                continue
            bank_root = Path(str(build["bank_root"])).expanduser().resolve()
            check = _selfcheck_bank(repo_root=repo_root, bank_root=bank_root, args=args, log_path=log_path)
            check.update({"profile_id": pid, "profile_hash": build.get("profile_hash"), "bank_root": str(bank_root)})
            selfcheck_rows.append(check)
            if check.get("selfcheck_status") != "PASS":
                failures.append({"profile_id": pid, "stage": "selfcheck", "status": check.get("selfcheck_status")})
                if not args.continue_on_error:
                    raise RuntimeError(f"profile {pid} selfcheck failed: {check}")
                continue
            if args.skip_eval or (args.smoke and not args.force_smoke_eval):
                continue
            ev = _run_a10c_endpoint(repo_root=repo_root, asset_root=asset_root, run_root=run_root, bank_root=bank_root, eval_root=eval_root, profile_id=pid, args=args, log_path=log_path)
            ev.update({"profile_hash": build.get("profile_hash"), "bank_root": str(bank_root), "bank_output_name": build.get("bank_output_name"), "build_status": build.get("build_status"), "selfcheck_status": check.get("selfcheck_status")})
            eval_records.append(ev)
            if ev.get("eval_status") != "PASS":
                failures.append({"profile_id": pid, "stage": "endpoint_eval", "status": ev.get("eval_status")})
                if not args.continue_on_error:
                    raise RuntimeError(f"profile {pid} endpoint eval failed: {ev}")
        endpoint_rows = _collect_endpoint_rows(eval_records)
        ranking_rows = _ranking(endpoint_rows)
        _write_csv(analysis_root / "prompt_profile_manifest.csv", manifest_rows)
        _write_csv(analysis_root / "prompt_textbank_selfcheck.csv", selfcheck_rows)
        _write_csv(analysis_root / "prompt_endpoint_eval_runs.csv", eval_records)
        _write_csv(analysis_root / "prompt_endpoint_eval.csv", endpoint_rows)
        _write_csv(analysis_root / "prompt_ranking_summary.csv", ranking_rows)
        result.update({
            "manifest_rows": len(manifest_rows),
            "selfcheck_rows": len(selfcheck_rows),
            "endpoint_eval_run_rows": len(eval_records),
            "endpoint_eval_rows": len(endpoint_rows),
            "ranking_rows": len(ranking_rows),
            "failures": failures,
            "artifacts": {
                "prompt_profile_manifest_csv": str(analysis_root / "prompt_profile_manifest.csv"),
                "prompt_textbank_selfcheck_csv": str(analysis_root / "prompt_textbank_selfcheck.csv"),
                "prompt_endpoint_eval_csv": str(analysis_root / "prompt_endpoint_eval.csv"),
                "prompt_ranking_summary_csv": str(analysis_root / "prompt_ranking_summary.csv"),
            },
        })
        if failures and not args.continue_on_error:
            result["status"] = "FAIL"
        _write_json(analysis_root / "A10E_run_result.json", result)
        _write_takeover(output_root, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0 if result.get("status") == "PASS" else 1
    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = repr(e)
        result["failures"] = failures
        _write_json(analysis_root / "A10E_run_result.json", result)
        _write_takeover(output_root, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
