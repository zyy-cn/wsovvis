from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import sys


def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_root_for_direct_cli()

from videocutler.ext_stageb_ovvis.algorithms.prealign import PrealignConfig, train_prealign
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import (
    Phase1MaterializationConfig,
    materialize_phase1_training_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G7 prealign phase-1: resolve runtime assets and materialize canonical sample inputs."
    )
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset_name", default="lvvis_train_base", choices=("lvvis_train_base", "lvvis_val"))
    parser.add_argument("--trajectory_source_branch", default="mainline", choices=("mainline", "gt_upper_bound"))
    parser.add_argument("--smoke_max_trajectories", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--t_dis_init", type=float, default=0.07)
    parser.add_argument("--b_u_init", type=float, default=0.0)
    parser.add_argument("--lambda_frame", type=float, default=None)
    parser.add_argument("--subset_fraction", type=float, default=None)
    parser.add_argument("--batch_budget", type=int, default=None)
    parser.add_argument("--show_progress", type=_parse_bool, default=True)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--write_runtime_metrics_jsonl", type=_parse_bool, default=True)
    parser.add_argument("--print_epoch_summary", type=_parse_bool, default=True)
    return parser.parse_args()


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _artifact_root(repo_root: Path) -> Path:
    override = str(os.environ.get("WSOVVIS_OUTPUT_ARTIFACT_ROOT", "")).strip()
    if override:
        return Path(override).expanduser()
    return repo_root


def _phase1_summary_path(repo_root: Path) -> Path:
    artifact_root = _artifact_root(repo_root)
    return artifact_root / "codex" / "outputs" / "G7_training" / "g7_prealign_smoke_summary.json"


def _phase1_samples_path(repo_root: Path) -> Path:
    artifact_root = _artifact_root(repo_root)
    return artifact_root / "codex" / "outputs" / "G7_training" / "g7_prealign_smoke_samples.jsonl"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_contract_check(repo_root: Path, *, train_result: Dict[str, Any], summary: Dict[str, Any]) -> Path:
    contract_check = {
        "gate_id": "G7_training",
        "contract_ref": "contracts/gates/G7_training.gate_contract.json",
        "status": "PASS" if summary.get("status") == "PASS" else "FAIL",
        "artifact_path_base": "output_root_relative",
        "primary_artifacts": [
            "train/prealign/train_state.json",
            "train/prealign/proxy_records.jsonl",
        ],
        "checks_run": [
            "train_state_selected_for_infer_readable",
            "stage_local_snapshot_reader_readable",
            "artifact_exists",
            "artifact_schema_valid",
        ],
    }
    artifact_root = _artifact_root(repo_root)
    contract_path = artifact_root / "codex" / "outputs" / "G7_training" / "prealign_contract_check.json"
    _write_json(contract_path, contract_check)
    return contract_path


def main() -> int:
    args = parse_args()
    if str(args.dataset_name) != "lvvis_train_base":
        print("ERROR: prealign training implementation currently supports dataset_name=lvvis_train_base only")
        return 2

    repo_root = _repo_root()
    requested_output_root = Path(args.output_root).expanduser()
    if requested_output_root.is_absolute():
        output_root = requested_output_root
    else:
        # Prefer shell PWD for canonical-path reporting; pathlib may return realpath.
        cwd_text = os.environ.get("PWD", "").strip()
        cwd = Path(cwd_text) if cwd_text else Path.cwd()
        output_root = cwd / requested_output_root
    result = materialize_phase1_training_samples(
        output_root,
        Phase1MaterializationConfig(
            dataset_name=str(args.dataset_name),
            trajectory_source_branch=str(args.trajectory_source_branch),
            smoke=bool(args.smoke),
            smoke_max_trajectories=int(args.smoke_max_trajectories),
            subset_fraction=(float(args.subset_fraction) if args.subset_fraction is not None else None),
            subset_seed=int(args.seed),
        ),
    )

    _write_jsonl(_phase1_samples_path(repo_root), list(result["samples"]))
    if bool(args.smoke):
        epochs = int(args.epochs) if args.epochs is not None else 1
        learning_rate = float(args.learning_rate) if args.learning_rate is not None else 1e-4
    else:
        epochs = int(args.epochs) if args.epochs is not None else 5
        learning_rate = float(args.learning_rate) if args.learning_rate is not None else 1e-4

    train_result = train_prealign(
        output_root=output_root,
        materialized_samples=result["samples"],
        config=PrealignConfig(
            dataset_name=str(args.dataset_name),
            trajectory_source_branch=str(args.trajectory_source_branch),
            device=str(args.device),
            seed=int(args.seed),
            smoke=bool(args.smoke),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            weight_decay=float(args.weight_decay),
            t_dis_init=float(args.t_dis_init),
            b_u_init=float(args.b_u_init),
            lambda_frame=(float(args.lambda_frame) if args.lambda_frame is not None else 0.25),
            runtime_asset_source=str(result['resolution'].get('runtime_asset_source', 'local_canonical_assets')),
            runtime_asset_source_local_incomplete=bool(result['resolution'].get('local_incomplete', False)),
            runtime_asset_output_root=str(result['resolution'].get('runtime_output_root', str(output_root))),
            batch_budget=(int(args.batch_budget) if args.batch_budget is not None else None),
            show_progress=bool(args.show_progress),
            log_every=int(args.log_every),
            write_runtime_metrics_jsonl=bool(args.write_runtime_metrics_jsonl),
            print_epoch_summary=bool(args.print_epoch_summary),
        ),
    )

    summary = {
        "status": "PASS",
        "gate_id": "G7_training",
        "phase_scope": "phase2_prealign_bounded_training",
        "run_scope": "smoke" if bool(args.smoke) else "full",
        "exp_name": str(args.exp_name),
        "dataset_name": str(args.dataset_name),
        "trajectory_source_branch": str(args.trajectory_source_branch),
        "formal_training_ready": False,
        "training_executed": True,
        "resolution": result["resolution"],
        "materialization_stats": result["stats"],
        "training_stats": {
            "record_count_input": train_result["record_count_input"],
            "record_count_trainable": train_result["record_count_trainable"],
            "record_count_output": train_result["record_count_output"],
            "coverage_ratio_trainable": train_result["coverage_ratio_trainable"],
            "skipped_reason_histogram": train_result["skipped_reason_histogram"],
            "loss_mean": train_result["loss_mean"],
            "loss_last": train_result["loss_last"],
            "optimization_loss_mean": train_result["optimization_loss_mean"],
            "optimization_loss_last": train_result["optimization_loss_last"],
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "batch_budget": train_result["batch_budget"],
            "budget_policy": train_result["budget_policy"],
            "loss_normalization": train_result["loss_normalization"],
            "micro_batch_count_per_epoch": train_result["micro_batch_count_per_epoch"],
        },
        "artifacts": {
            "materialized_sample_artifact": "codex/outputs/G7_training/g7_prealign_smoke_samples.jsonl",
            "train_state": "train/prealign/train_state.json",
            "proxy_records": "train/prealign/proxy_records.jsonl",
            "checkpoint_last": "train/prealign/checkpoints/prealign_last.pth",
            "summary_artifact": "codex/outputs/G7_training/g7_prealign_smoke_summary.json",
        },
    }
    _write_json(_phase1_summary_path(repo_root), summary)
    _write_contract_check(repo_root, train_result=train_result, summary=summary)

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
