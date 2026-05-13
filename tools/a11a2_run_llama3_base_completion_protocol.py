#!/usr/bin/env python3
"""Run A11A-2 Llama3-base completion protocol endpoint probes.

This orchestrates:
  1. build one or more base-compatible completion text banks;
  2. validate each text bank with check_lvvis_text_bank_alignment.py;
  3. evaluate each text bank with A10C at alpha=1.0;
  4. write compact endpoint summaries under <output_root>/analysis.

It assumes tools/a10c_run_llama4096_linear_isometric_distortion_calibration.py
has already been deployed in the repo, as in prior A10C/A10F/A10G runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

_PROTOCOLS = [
    "completion_visual_generated",
    "completion_visual_class_span",
    "natural_visual_generated",
]


def _repo_default() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(str(k))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _run(cmd: Sequence[str], *, log_path: Path, progress: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        print("[A11A2][cmd] " + " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            log.write(line)
            log.flush()
            if progress:
                print(line, end="", flush=True)
        return int(p.wait())


def _float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    repo = _repo_default()
    p.add_argument("--repo_root", default=str(repo))
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--run_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--textbank_root", default="", help="default: <asset_root>/text_bank_llama3/lvvis")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--protocols", default=",".join(_PROTOCOLS))
    p.add_argument("--output_prefix", default="a11a2_llama3_8b_base")
    p.add_argument("--max_classes", type=int, default=0, help="0 means full 1196; use 20 for smoke")
    p.add_argument("--max_seq_len", type=int, default=384)
    p.add_argument("--max_batch_size", type=int, default=16)
    p.add_argument("--max_gen_len", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--master_port_base", type=int, default=56850)
    p.add_argument("--local_rank", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--endpoint_alphas", default="1.0")
    p.add_argument("--projectors", default="orthogonal_linear,ridge_linear")
    p.add_argument("--test_scopes", default="novel_val,val_base_all")
    p.add_argument("--candidate_scope", default="full_available")
    p.add_argument("--eval_seeds", default="0")
    p.add_argument("--anchor_ratios", default="1.0")
    p.add_argument("--ridge_alpha", type=float, default=0.01)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--continue_on_error", action="store_true")
    p.add_argument("--progress", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    analysis_root = output_root / "analysis"
    textbank_root = Path(args.textbank_root).expanduser().resolve() if args.textbank_root else asset_root / "text_bank_llama3" / "lvvis"
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)

    builder = repo_root / "tools" / "build_lvvis_llama3_base_completion_text_bank.py"
    checker = repo_root / "tools" / "check_lvvis_text_bank_alignment.py"
    a10c = repo_root / "tools" / "a10c_run_llama4096_linear_isometric_distortion_calibration.py"
    missing = [str(p) for p in (builder, checker, a10c) if not p.is_file()]
    if missing:
        raise FileNotFoundError("required tool(s) missing: " + ", ".join(missing))

    protocols = [p.strip() for p in str(args.protocols).split(",") if p.strip()]
    result: Dict[str, Any] = {
        "status": "PASS",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_root": str(repo_root),
        "asset_root": str(asset_root),
        "run_root": str(run_root),
        "output_root": str(output_root),
        "analysis_root": str(analysis_root),
        "textbank_root": str(textbank_root),
        "protocols": protocols,
        "failures": [],
    }
    build_rows: List[Dict[str, Any]] = []
    endpoint_rows: List[Dict[str, Any]] = []

    for idx, protocol in enumerate(protocols):
        suffix = f"_smoke{int(args.max_classes)}" if int(args.max_classes) > 0 else ""
        bank_name = f"{args.output_prefix}_{protocol}{suffix}"
        bank_root = textbank_root / bank_name
        protocol_out = output_root / protocol
        protocol_out.mkdir(parents=True, exist_ok=True)
        if args.progress:
            print(f"[A11A2] protocol={protocol} bank={bank_root}", flush=True)
        build_cmd = [
            sys.executable, str(builder),
            "--repo_root", str(repo_root),
            "--assert_root", str(asset_root),
            "--output_root", str(textbank_root),
            "--output_name", bank_name,
            "--protocol", protocol,
            "--ckpt_dir", str(Path(args.ckpt_dir).expanduser()),
            "--tokenizer_path", str(Path(args.tokenizer_path).expanduser()),
            "--max_seq_len", str(int(args.max_seq_len)),
            "--max_batch_size", str(int(args.max_batch_size)),
            "--max_gen_len", str(int(args.max_gen_len)),
            "--temperature", str(float(args.temperature)),
            "--top_p", str(float(args.top_p)),
            "--seed", str(int(args.seed)),
            "--master_port", str(int(args.master_port_base) + idx),
            "--local_rank", str(int(args.local_rank)),
            "--log_every_classes", "20",
            "--print_progress",
        ]
        if int(args.max_classes) > 0:
            build_cmd += ["--max_classes", str(int(args.max_classes))]
        if bool(args.overwrite):
            build_cmd += ["--overwrite"]
        try:
            rc = _run(build_cmd, log_path=protocol_out / "build_textbank.log", progress=bool(args.progress))
            if rc != 0:
                raise RuntimeError(f"builder failed rc={rc}")
            check_cmd = [
                sys.executable, str(checker),
                "--bank_root", str(bank_root),
                "--expect_class_count", str(int(args.max_classes) if int(args.max_classes) > 0 else 1196),
            ]
            if int(args.max_classes) > 0:
                check_cmd += ["--allow_smoke"]
            rc = _run(check_cmd, log_path=protocol_out / "check_textbank.log", progress=bool(args.progress))
            if rc != 0:
                raise RuntimeError(f"textbank check failed rc={rc}")
            manifest = _read_json(bank_root / "manifest.json")
            build_rows.append({
                "protocol": protocol,
                "bank_root": str(bank_root),
                "class_count": manifest.get("class_count"),
                "profile_id": manifest.get("profile_id"),
                "pooling_mode": manifest.get("pooling_mode"),
                "token_feature_alignment": manifest.get("token_feature_alignment"),
                "llama_hidden_mean_path": manifest.get("artifacts", {}).get("llama_hidden_mean_path"),
                "llama_hidden_mean_sha256": manifest.get("artifacts", {}).get("llama_hidden_mean_sha256"),
            })
            eval_root = protocol_out / "endpoint"
            a10c_cmd = [
                sys.executable, str(a10c),
                "--repo_root", str(repo_root),
                "--asset_root", str(asset_root),
                "--run_root", str(run_root),
                "--output_root", str(eval_root),
                "--visual_only_root", str(bank_root),
                "--text_variant", "llama_hidden_mean",
                "--text_dim", "4096",
                "--alphas", str(args.endpoint_alphas),
                "--projectors", str(args.projectors),
                "--ridge_alpha", str(float(args.ridge_alpha)),
                "--test_scopes", str(args.test_scopes),
                "--candidate_scope", str(args.candidate_scope),
                "--seeds", str(args.eval_seeds),
                "--anchor_ratios", str(args.anchor_ratios),
                "--device", str(args.device),
                "--no_plots",
                "--progress",
            ]
            rc = _run(a10c_cmd, log_path=protocol_out / "a10c_endpoint.log", progress=bool(args.progress))
            if rc != 0:
                raise RuntimeError(f"A10C endpoint failed rc={rc}")
            agg = eval_root / "analysis" / "a10c_llama4096_alpha_aggregate.csv"
            for row in _read_csv(agg):
                out = dict(row)
                out["protocol"] = protocol
                out["bank_root"] = str(bank_root)
                out["endpoint_root"] = str(eval_root)
                endpoint_rows.append(out)
        except Exception as exc:
            failure = {"protocol": protocol, "error": str(exc)}
            result["failures"].append(failure)
            result["status"] = "PARTIAL" if bool(args.continue_on_error) else "FAIL"
            if not bool(args.continue_on_error):
                break

    _write_csv(analysis_root / "textbank_manifest_summary.csv", build_rows)
    _write_csv(analysis_root / "endpoint_summary.csv", endpoint_rows)

    ranking_rows: List[Dict[str, Any]] = []
    for protocol in protocols:
        rows = [r for r in endpoint_rows if r.get("protocol") == protocol]
        novel_ridge = next((r for r in rows if r.get("test_scope") == "novel_val" and r.get("projector") == "ridge_linear"), {})
        novel_orth = next((r for r in rows if r.get("test_scope") == "novel_val" and r.get("projector") == "orthogonal_linear"), {})
        val_ridge = next((r for r in rows if r.get("test_scope") == "val_base_all" and r.get("projector") == "ridge_linear"), {})
        ranking_rows.append({
            "protocol": protocol,
            "bank_root": next((r.get("bank_root") for r in rows if r.get("bank_root")), ""),
            "novel_val_ridge_t2v_at1": novel_ridge.get("t2v_rank@1_mean", ""),
            "novel_val_ridge_t2v_at5": novel_ridge.get("t2v_rank@5_mean", ""),
            "novel_val_ridge_mean_normalized_rank": novel_ridge.get("mean_normalized_rank_mean", ""),
            "novel_val_orthogonal_t2v_at1": novel_orth.get("t2v_rank@1_mean", ""),
            "novel_val_orthogonal_t2v_at5": novel_orth.get("t2v_rank@5_mean", ""),
            "novel_val_orthogonal_mean_normalized_rank": novel_orth.get("mean_normalized_rank_mean", ""),
            "val_base_all_ridge_t2v_at5": val_ridge.get("t2v_rank@5_mean", ""),
            "spearman_Xalpha_vs_V": novel_ridge.get("spearman_Xalpha_vs_V_mean", ""),
            "knn_overlap@10": novel_ridge.get("knn_overlap@10_mean", ""),
            "triplet_agreement": novel_ridge.get("triplet_agreement_mean", ""),
            "primary_metric": "novel_val_ridge_t2v_at5",
            "primary_value": _float(novel_ridge, "t2v_rank@5_mean"),
        })
    ranking_rows.sort(key=lambda r: float(r.get("primary_value", float("nan"))), reverse=True)
    _write_csv(analysis_root / "protocol_ranking_summary.csv", ranking_rows)

    result.update({
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol_count": len(protocols),
        "textbank_rows": len(build_rows),
        "endpoint_rows": len(endpoint_rows),
        "ranking_rows": len(ranking_rows),
        "artifacts": {
            "textbank_manifest_summary_csv": str(analysis_root / "textbank_manifest_summary.csv"),
            "endpoint_summary_csv": str(analysis_root / "endpoint_summary.csv"),
            "protocol_ranking_summary_csv": str(analysis_root / "protocol_ranking_summary.csv"),
        },
    })
    _write_json(analysis_root / "A11A2_run_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
