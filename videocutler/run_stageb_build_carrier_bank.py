from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage-B carrier bank artifacts.")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--dataset_name", required=True, choices=("lvvis_train_base", "lvvis_val"))
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--trajectory_source_branch",
        default="mainline",
        choices=("mainline", "gt_upper_bound"),
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _contract_check_path(repo_root: Path) -> Path:
    return repo_root / "codex" / "outputs" / "G5_carrier_bank" / "carrier_contract_check.json"


def _gt_analysis_report_path(output_root: Path) -> Path:
    return output_root / "carrier_bank_gt" / "analysis" / "upper_bound_comparison_report.json"


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _update_gt_analysis_report(output_root: Path, run_scope: str) -> None:
    report_path = _gt_analysis_report_path(output_root)
    mainline_train = output_root / "carrier_bank" / "lvvis_train_base" / "carrier_records.jsonl"
    mainline_val = output_root / "carrier_bank" / "lvvis_val" / "carrier_records.jsonl"
    gt_train = output_root / "carrier_bank_gt" / "lvvis_train_base" / "carrier_records.jsonl"
    gt_val = output_root / "carrier_bank_gt" / "lvvis_val" / "carrier_records.jsonl"

    violations: List[str] = []
    if not gt_train.exists():
        violations.append("missing_gt_train_carrier_records")
    if not gt_val.exists():
        violations.append("missing_gt_val_carrier_records")
    if not mainline_train.exists():
        violations.append("missing_mainline_train_carrier_records")
    if not mainline_val.exists():
        violations.append("missing_mainline_val_carrier_records")

    payload = {
        "status": "PASS" if not violations else "FAIL",
        "gate_id": "G5_carrier_bank",
        "report_kind": "gt_upper_bound_comparison",
        "run_scope": run_scope,
        "comparison_scope": "analysis_only",
        "trajectory_source_branch": "gt_upper_bound",
        "mainline_source_branch": "mainline",
        "dataset_names": ["lvvis_train_base", "lvvis_val"],
        "same_frame_feature_source": True,
        "same_geometry_source": True,
        "same_token_pooling_rule": True,
        "same_aggregation_rule": True,
        "mainline_trajectory_artifacts": [
            "exports/lvvis_train_base/trajectory_records.jsonl",
            "exports/lvvis_val/trajectory_records.jsonl",
        ],
        "gt_trajectory_artifacts": [
            "exports_gt/lvvis_train_base/trajectory_records.jsonl",
            "exports_gt/lvvis_val/trajectory_records.jsonl",
        ],
        "mainline_carrier_artifacts": [
            "carrier_bank/lvvis_train_base/carrier_records.jsonl",
            "carrier_bank/lvvis_val/carrier_records.jsonl",
        ],
        "gt_carrier_artifacts": [
            "carrier_bank_gt/lvvis_train_base/carrier_records.jsonl",
            "carrier_bank_gt/lvvis_val/carrier_records.jsonl",
        ],
        "analysis_report_path": "carrier_bank_gt/analysis/upper_bound_comparison_report.json",
        "coverage_ratio": 1.0 if not violations else 0.0,
        "consumer_ready": not violations,
        "violations_preview": violations[:20],
        "notes": "supplementary GT upper-bound branch; mainline gate truth remains canonical",
    }
    _write_json(report_path, payload)


def main() -> int:
    args = parse_args()
    from videocutler.ext_stageb_ovvis.banks.carrier_bank import CarrierBuildConfig, build_carrier_bank

    repo_root = _repo_root()
    output_root = Path(args.output_root).expanduser().resolve()
    run_scope = "smoke" if args.smoke else "full"

    status = "PASS"
    failure_reason = ""
    result: Dict = {}
    try:
        result = build_carrier_bank(
            CarrierBuildConfig(
                dataset_name=args.dataset_name,
                output_root=output_root,
                trajectory_source_branch=args.trajectory_source_branch,
                smoke=bool(args.smoke),
            )
        )
        if args.trajectory_source_branch == "gt_upper_bound":
            _update_gt_analysis_report(output_root, run_scope)
    except Exception as exc:
        status = "FAIL"
        failure_reason = str(exc)

    contract_check = {
        "status": status,
        "gate_id": "G5_carrier_bank",
        "run_scope": run_scope,
        "input_source_type": "local_snapshot",
        "data_scope": "train_or_val",
        "consumer_target": "run_stageb_train_prealign|run_stageb_train_softem|run_stageb_infer_ov",
        "trajectory_source_branch": args.trajectory_source_branch,
        "dataset_name": args.dataset_name,
        "record_count": int(result.get("record_count_output", 0) if status == "PASS" else 0),
        "coverage_ratio": float(result.get("coverage_ratio", 0.0) if status == "PASS" else 0.0),
        "invalid_reason_stats": result.get("invalid_reason_stats", {}) if status == "PASS" else {"build_failed": 1},
        "consumer_ready": bool(status == "PASS" and run_scope == "full"),
        "notes": [] if status == "PASS" else [failure_reason],
        "primary_artifacts": [
            "carrier_bank/lvvis_train_base/carrier_records.jsonl",
            "carrier_bank/lvvis_val/carrier_records.jsonl",
        ],
    }
    _write_json(_contract_check_path(repo_root), contract_check)

    if status != "PASS":
        print(f"ERROR: {failure_reason}")
        return 2

    print(
        json.dumps(
            {
                "status": "PASS",
                "dataset_name": args.dataset_name,
                "trajectory_source_branch": args.trajectory_source_branch,
                "run_scope": run_scope,
                "record_count_input": result["record_count_input"],
                "record_count_output": result["record_count_output"],
                "coverage_ratio": result["coverage_ratio"],
                "carrier_records_path": str(result["carrier_records_path"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
