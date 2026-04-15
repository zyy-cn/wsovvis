from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from videocutler.ext_stageb_ovvis.algorithms.soft_em import SoftEMConfig, run_soft_em
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import (
    Phase1MaterializationConfig,
    materialize_phase1_training_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G7 soft-EM training entrypoint (bounded/provisional implementation scope).")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mode", default="base_then_aug", choices=("base_only", "aug_only", "base_then_aug"))
    parser.add_argument("--dataset_name", default="lvvis_train_base", choices=("lvvis_train_base", "lvvis_val"))
    parser.add_argument("--trajectory_source_branch", default="mainline", choices=("mainline", "gt_upper_bound"))
    parser.add_argument("--smoke_max_trajectories", type=int, default=128)
    parser.add_argument("--base_epochs", type=int, default=None)
    parser.add_argument("--aug_epochs", type=int, default=None)
    parser.add_argument("--base_learning_rate", type=float, default=None)
    parser.add_argument("--aug_learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--em_subiterations", type=int, default=1)
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _summary_path(repo_root: Path) -> Path:
    return repo_root / "codex" / "outputs" / "G7_training" / "g7_softem_smoke_summary.json"


def _samples_path(repo_root: Path) -> Path:
    return repo_root / "codex" / "outputs" / "G7_training" / "g7_softem_smoke_samples.jsonl"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    if str(args.dataset_name) != "lvvis_train_base":
        print("ERROR: soft-EM implementation currently supports dataset_name=lvvis_train_base only")
        return 2

    repo_root = _repo_root()
    requested_output_root = Path(args.output_root).expanduser()
    if requested_output_root.is_absolute():
        output_root = requested_output_root
    else:
        cwd_text = os.environ.get("PWD", "").strip()
        cwd = Path(cwd_text) if cwd_text else Path.cwd()
        output_root = cwd / requested_output_root

    materialized = materialize_phase1_training_samples(
        output_root,
        Phase1MaterializationConfig(
            dataset_name=str(args.dataset_name),
            trajectory_source_branch=str(args.trajectory_source_branch),
            smoke=bool(args.smoke),
            smoke_max_trajectories=int(args.smoke_max_trajectories),
        ),
    )
    _write_jsonl(_samples_path(repo_root), list(materialized["samples"]))

    if bool(args.smoke):
        base_epochs = int(args.base_epochs) if args.base_epochs is not None else 1
        aug_epochs = int(args.aug_epochs) if args.aug_epochs is not None else 1
        base_lr = float(args.base_learning_rate) if args.base_learning_rate is not None else 5e-5
        aug_lr = float(args.aug_learning_rate) if args.aug_learning_rate is not None else 5e-5
    else:
        base_epochs = int(args.base_epochs) if args.base_epochs is not None else 5
        aug_epochs = int(args.aug_epochs) if args.aug_epochs is not None else 5
        base_lr = float(args.base_learning_rate) if args.base_learning_rate is not None else 1e-4
        aug_lr = float(args.aug_learning_rate) if args.aug_learning_rate is not None else 1e-4

    softem_result = run_soft_em(
        output_root=output_root,
        materialized_samples=materialized["samples"],
        config=SoftEMConfig(
            dataset_name=str(args.dataset_name),
            trajectory_source_branch=str(args.trajectory_source_branch),
            mode=str(args.mode),
            device=str(args.device),
            seed=int(args.seed),
            smoke=bool(args.smoke),
            temperature=float(args.temperature),
            weight_decay=float(args.weight_decay),
            em_subiterations=int(args.em_subiterations),
            base_epochs=int(base_epochs),
            aug_epochs=int(aug_epochs),
            base_learning_rate=float(base_lr),
            aug_learning_rate=float(aug_lr),
        ),
    )

    artifacts = {
        "materialized_sample_artifact": "codex/outputs/G7_training/g7_softem_smoke_samples.jsonl",
        "summary_artifact": "codex/outputs/G7_training/g7_softem_smoke_summary.json",
    }
    for stage in softem_result["stage_reports"]:
        artifacts[f"{stage['stage_id']}_train_state"] = stage["train_state_path"]
        artifacts[f"{stage['stage_id']}_responsibility_records"] = stage["responsibility_records_path"]
        artifacts[f"{stage['stage_id']}_checkpoint_last"] = stage["checkpoint_last_path"]

    summary = {
        "status": "PASS",
        "gate_id": "G7_training",
        "phase_scope": "phase3_softem_bounded_training",
        "run_scope": "smoke" if bool(args.smoke) else "full",
        "exp_name": str(args.exp_name),
        "dataset_name": str(args.dataset_name),
        "trajectory_source_branch": str(args.trajectory_source_branch),
        "mode": str(args.mode),
        "formal_training_ready": False,
        "training_executed": True,
        "resolution": materialized["resolution"],
        "materialization_stats": materialized["stats"],
        "training_stats": {
            "record_count_input": softem_result["record_count_input"],
            "record_count_trainable": softem_result["record_count_trainable"],
            "record_count_output": softem_result["record_count_output"],
            "coverage_ratio_trainable": softem_result["coverage_ratio_trainable"],
            "skipped_reason_histogram": softem_result["skipped_reason_histogram"],
            "stage_reports": softem_result["stage_reports"],
            "base_epochs": int(base_epochs),
            "aug_epochs": int(aug_epochs),
            "base_learning_rate": float(base_lr),
            "aug_learning_rate": float(aug_lr),
        },
        "artifacts": artifacts,
        "selected_checkpoint_path": softem_result["selected_checkpoint_path"],
    }
    _write_json(_summary_path(repo_root), summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
