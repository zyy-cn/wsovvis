#!/usr/bin/env python3
"""A9 read-only manifold compatibility audit launcher.

This wrapper does not mutate training code, checkpoints, or assets. It only
calls existing A8 audit scripts under a single A9 output root, then invokes the
A9 collector to build a compact local summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


DEFAULT_G8_ROOT = "codex/outputs/G8_inference_and_eval"
DEFAULT_RUN_NAME = "sinkhorn_safeneg_k32_b010_preaug_k10_repro_20260427"
DEFAULT_A9_NAME = "A9_MANIFOLD_COMPATIBILITY_AUDIT"
DEFAULT_VARIANTS = "clip_current,clip_of_llm_mean,llama_hidden_mean,llama_direct_concept_mean"


@dataclass
class StepResult:
    name: str
    status: str
    command: List[str]
    output_dir: str
    log_path: str
    returncode: Optional[int] = None
    error: str = ""


def _repo_default() -> Path:
    return Path.cwd().resolve()


def _run_root_default(repo_root: Path) -> Path:
    preferred = repo_root / DEFAULT_G8_ROOT / DEFAULT_RUN_NAME
    if preferred.exists():
        return preferred
    return repo_root / DEFAULT_G8_ROOT


def _output_root_default(repo_root: Path) -> Path:
    return repo_root / DEFAULT_G8_ROOT / DEFAULT_A9_NAME


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
    roots = [run_root / "analysis", repo_root / DEFAULT_G8_ROOT]
    found: list[Path] = []
    for root in roots:
        if root.exists():
            found.extend(root.rglob("per_class_train_val_525_join.csv"))
    if found:
        found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return found[0]
    return None


def _find_val_visible_per_row(repo_root: Path, run_root: Path) -> Optional[Path]:
    roots = [run_root / "analysis", repo_root / DEFAULT_G8_ROOT]
    names = ["visible525_candidate_rankk_per_row.csv", "row_level_margin_per_row.csv"]
    found: list[Path] = []
    for root in roots:
        if root.exists():
            for name in names:
                found.extend(root.rglob(name))
    found = [p for p in found if "lvvis_val" in str(p)] or found
    if found:
        found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return found[0]
    return None


def _parse_checkpoint_specs(specs: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for chunk in str(specs or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            name, path = chunk.split("=", 1)
            out.append((name.strip(), path.strip()))
        else:
            p = Path(chunk)
            out.append((p.stem, chunk))
    return out


def _first_checkpoint_path(specs: str, run_root: Path) -> str:
    parsed = _parse_checkpoint_specs(specs)
    if parsed:
        return parsed[0][1]
    default = run_root / "outputs/a8_joint_train_time_dynamic_hungarian/lvvis_train_base/D-J3_pre1_dyn1_ep10/train/joint_train_time_dynamic_hungarian/a8_joint_train_time_dynamic_last.pth"
    return str(default)


def _cmd(repo_root: Path, script: str, args: Sequence[str]) -> list[str]:
    return [sys.executable, str(repo_root / "tools" / script), *args]


def _run_step(name: str, command: Sequence[str], output_dir: Path, logs_dir: Path, env: dict[str, str], continue_on_error: bool) -> StepResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.log"
    cmd_line = " ".join(shlex.quote(str(x)) for x in command)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {name}\n")
        log.write(f"# cwd={Path.cwd()}\n")
        log.write(f"# command={cmd_line}\n\n")
        log.flush()
        proc = subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(Path.cwd()))
    status = "PASS" if proc.returncode == 0 else "FAIL"
    res = StepResult(name=name, status=status, command=list(map(str, command)), output_dir=str(output_dir), log_path=str(log_path), returncode=int(proc.returncode))
    if proc.returncode != 0:
        res.error = f"returncode={proc.returncode}; see {log_path}"
        if not continue_on_error:
            raise RuntimeError(f"{name} failed: {res.error}")
    return res


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run A9 read-only manifold compatibility audit by orchestrating existing A8 audits.")
    p.add_argument("--repo_root", default=str(_repo_default()))
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--run_root", default="")
    p.add_argument("--output_root", default="")
    p.add_argument("--train_dataset_name", default="lvvis_train_base")
    p.add_argument("--val_dataset_name", default="lvvis_val")
    p.add_argument("--train_annotation_json", default="")
    p.add_argument("--val_annotation_json", default="")
    p.add_argument("--visible_csv", default="", help="Defaults to latest base_641_visibility_by_class.csv with 525 in_row_gap classes.")
    p.add_argument("--per_class_join", default="", help="Optional per_class_train_val_525_join.csv for graph grouping.")
    p.add_argument("--val_visible_per_row", default="", help="Optional val visible525 per-row CSV for topology scripts.")
    p.add_argument("--variants", default=DEFAULT_VARIANTS)
    p.add_argument("--visual_only_root", default="/home/zyy/code/wsovvis_asserts/text_bank_llama3/lvvis/lvvis_visual_only_v1")
    p.add_argument("--direct_concept_root", default="/home/zyy/code/wsovvis_asserts/text_bank_llama3/lvvis/llama3_direct_concept_v1")
    p.add_argument("--checkpoint_specs", default="", help="Semicolon separated NAME=PATH list; passed to manifold diagnosis when applicable.")
    p.add_argument("--checkpoint_path", default="", help="Optional single checkpoint for graph-isomorphism/topology/rescue scripts.")
    p.add_argument("--mapping_methods", default="ridge,least_squares")
    p.add_argument("--ridge_alpha", type=float, default=1e-2)
    p.add_argument("--holdout_fraction", type=float, default=0.2)
    p.add_argument("--anchor_counts", default="32,64,128,256,384,450")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--neighbor_k", type=int, default=10)
    p.add_argument("--spectral_m", type=int, default=32)
    p.add_argument("--bootstrap_rounds", type=int, default=20)
    p.add_argument("--random_perm_rounds", type=int, default=50)
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="Set max_rows to smoke_max_rows unless max_rows is already positive.")
    p.add_argument("--smoke_max_rows", type=int, default=2000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip_e1", action="store_true")
    p.add_argument("--skip_e2", action="store_true")
    p.add_argument("--skip_e3", action="store_true")
    p.add_argument("--skip_e4", action="store_true")
    p.add_argument("--continue_on_error", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve() if args.run_root else _run_root_default(repo_root)
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else _output_root_default(repo_root)
    analysis_root = output_root / "analysis"
    logs_dir = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    max_rows = int(args.max_rows)
    if args.smoke and max_rows <= 0:
        max_rows = int(args.smoke_max_rows)

    visible_csv = Path(args.visible_csv).expanduser().resolve() if args.visible_csv else _find_visible_csv(repo_root, run_root)
    if not visible_csv or not visible_csv.is_file():
        raise RuntimeError("Could not locate visible525 CSV. Pass --visible_csv explicitly.")
    per_class_join = Path(args.per_class_join).expanduser().resolve() if args.per_class_join else _find_per_class_join(repo_root, run_root)
    val_visible_per_row = Path(args.val_visible_per_row).expanduser().resolve() if args.val_visible_per_row else _find_val_visible_per_row(repo_root, run_root)
    checkpoint_path = args.checkpoint_path or _first_checkpoint_path(args.checkpoint_specs, run_root)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(repo_root)

    # Common arguments accepted by most A8 audit scripts.  Do not include
    # --device here: a8_textbank_graph_audit.py is CPU-only and does not
    # accept that flag in the current code snapshot.  GPU-aware scripts get
    # --device through common_with_device below.
    common = [
        "--repo_root", str(repo_root),
        "--asset_root", str(Path(args.asset_root).expanduser()),
        "--run_root", str(run_root),
        "--train_dataset_name", args.train_dataset_name,
        "--val_dataset_name", args.val_dataset_name,
        "--max_rows", str(max_rows),
    ]
    if args.train_annotation_json:
        common.extend(["--train_annotation_json", args.train_annotation_json])
    if args.val_annotation_json:
        common.extend(["--val_annotation_json", args.val_annotation_json])

    common_with_device = [*common, "--device", args.device]

    results: list[StepResult] = []

    if not args.skip_e1:
        e1 = analysis_root / "E1_graph_compatibility"
        results.append(_run_step(
            "E1_textbank_graph",
            _cmd(repo_root, "a8_textbank_graph_audit.py", [
                *common,
                "--output_root", str(e1 / "textbank_graph"),
                "--visible_csv", str(visible_csv),
                "--variants", args.variants,
                "--visual_only_root", args.visual_only_root,
                "--direct_concept_root", args.direct_concept_root,
                "--neighbor_k", str(args.neighbor_k),
            ]),
            e1 / "textbank_graph", logs_dir, env, args.continue_on_error,
        ))
        results.append(_run_step(
            "E1_text_vision_graph_isomorphism",
            _cmd(repo_root, "a8_text_vision_prototype_graph_isomorphism_audit.py", [
                *common_with_device,
                "--output_root", str(e1 / "text_vision_graph"),
                "--checkpoint_path", str(checkpoint_path),
                "--per_class_join", str(per_class_join or ""),
                "--val_visible_per_row", str(val_visible_per_row or ""),
                "--neighbor_k", str(args.neighbor_k),
                "--spectral_m", str(args.spectral_m),
                "--bootstrap_rounds", str(args.bootstrap_rounds),
                "--random_perm_rounds", str(args.random_perm_rounds),
            ]),
            e1 / "text_vision_graph", logs_dir, env, args.continue_on_error,
        ))
        results.append(_run_step(
            "E1_topology_mismatch",
            _cmd(repo_root, "a8_topology_mismatch_audit.py", [
                *common_with_device,
                "--output_root", str(e1 / "topology_mismatch"),
                "--checkpoint_path", str(checkpoint_path),
                "--per_class_join", str(per_class_join or ""),
                "--val_visible_per_row", str(val_visible_per_row or ""),
                "--neighbor_k", str(args.neighbor_k),
            ]),
            e1 / "topology_mismatch", logs_dir, env, args.continue_on_error,
        ))

    manifold_common = [
        *common_with_device,
        "--visible_csv", str(visible_csv),
        "--variants", args.variants,
        "--visual_only_root", args.visual_only_root,
        "--direct_concept_root", args.direct_concept_root,
        "--mapping_methods", args.mapping_methods,
        "--ridge_alpha", str(args.ridge_alpha),
        "--holdout_fraction", str(args.holdout_fraction),
        "--anchor_counts", args.anchor_counts,
        "--seeds", args.seeds,
        "--neighbor_k", str(args.neighbor_k),
        "--checkpoint_specs", args.checkpoint_specs,
    ]
    if not args.skip_e2:
        e2 = analysis_root / "E2_anchor_holdout"
        results.append(_run_step(
            "E2_class_proto_holdout",
            _cmd(repo_root, "a8_manifold_alignment_diagnosis.py", [*manifold_common, "--output_root", str(e2 / "class_proto"), "--only", "class_proto"]),
            e2 / "class_proto", logs_dir, env, args.continue_on_error,
        ))
        results.append(_run_step(
            "E2_anchor_curve",
            _cmd(repo_root, "a8_manifold_alignment_diagnosis.py", [*manifold_common, "--output_root", str(e2 / "anchor_curve"), "--only", "anchor_curve"]),
            e2 / "anchor_curve", logs_dir, env, args.continue_on_error,
        ))

    if not args.skip_e3:
        e3 = analysis_root / "E3_projector_capacity"
        results.append(_run_step(
            "E3_projector_distortion",
            _cmd(repo_root, "a8_manifold_alignment_diagnosis.py", [*manifold_common, "--output_root", str(e3 / "projector_distortion"), "--only", "projector_distortion"]),
            e3 / "projector_distortion", logs_dir, env, args.continue_on_error,
        ))

    if not args.skip_e4:
        e4 = analysis_root / "E4_proto_to_row_release"
        results.append(_run_step(
            "E4_row_level_margin",
            _cmd(repo_root, "a8_manifold_alignment_diagnosis.py", [*manifold_common, "--output_root", str(e4 / "row_margin"), "--only", "row_margin"]),
            e4 / "row_margin", logs_dir, env, args.continue_on_error,
        ))
        results.append(_run_step(
            "E4_visual_proto_rescue",
            _cmd(repo_root, "a8_visual_prototype_rescue_probe.py", [
                *common_with_device,
                "--output_root", str(e4 / "visual_proto_rescue"),
                "--checkpoint_path", str(checkpoint_path),
                "--per_class_join", str(per_class_join or ""),
                "--alphas", args.alphas,
            ]),
            e4 / "visual_proto_rescue", logs_dir, env, args.continue_on_error,
        ))

    # Collect compact summary.
    collector = repo_root / "tools" / "a9_collect_manifold_compatibility_summary.py"
    if collector.is_file():
        results.append(_run_step(
            "A9_collect_summary",
            [sys.executable, str(collector), "--output_root", str(output_root), "--analysis_root", str(analysis_root)],
            output_root / "analysis", logs_dir, env, True,
        ))

    payload = {
        "status": "PASS" if all(r.status == "PASS" for r in results) else "PARTIAL" if args.continue_on_error else "FAIL",
        "repo_root": str(repo_root),
        "run_root": str(run_root),
        "output_root": str(output_root),
        "analysis_root": str(analysis_root),
        "visible_csv": str(visible_csv),
        "per_class_join": str(per_class_join or ""),
        "val_visible_per_row": str(val_visible_per_row or ""),
        "checkpoint_path": str(checkpoint_path),
        "max_rows": max_rows,
        "steps": [asdict(r) for r in results],
    }
    (output_root / "A9_MANIFOLD_COMPATIBILITY_RUN_SUMMARY.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] in {"PASS", "PARTIAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
