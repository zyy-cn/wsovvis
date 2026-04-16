from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from videocutler.ext_stageb_ovvis.algorithms.prealign import PrealignConfig, train_prealign
from videocutler.ext_stageb_ovvis.algorithms.soft_em import SoftEMConfig, run_soft_em
from videocutler.run_stageb_train_softem import resolve_em_subiterations
from videocutler.ext_stageb_ovvis.audit.attribution_ledger import AttributionLedgerBuffer
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import (
    Phase1MaterializationConfig,
    materialize_phase1_training_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G7 Audit-v1: attribution trend audit across prealign, softem_base, and softem_aug."
    )
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset_name", default="lvvis_train_base", choices=("lvvis_train_base",))
    parser.add_argument("--trajectory_source_branch", default="mainline", choices=("mainline", "gt_upper_bound"))
    parser.add_argument("--smoke_max_trajectories", type=int, default=128)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--gt_sidecar_dir", default="audit")
    parser.add_argument("--prealign_epochs", type=int, default=None)
    parser.add_argument("--base_epochs", type=int, default=None)
    parser.add_argument("--aug_epochs", type=int, default=None)
    parser.add_argument("--prealign_learning_rate", type=float, default=None)
    parser.add_argument("--base_learning_rate", type=float, default=None)
    parser.add_argument("--aug_learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--em_subiterations", type=int, default=None)
    parser.add_argument("--mode", default="base_then_aug", choices=("base_only", "aug_only", "base_then_aug"))
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _summary_path(repo_root: Path) -> Path:
    return repo_root / "codex" / "outputs" / "G7_training" / "g7_audit_v1_impl_latest.json"


def _summary_md_path(repo_root: Path) -> Path:
    return repo_root / "codex" / "outputs" / "G7_training" / "g7_audit_v1_impl_latest.md"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = _repo_root()
    requested_output_root = Path(args.output_root).expanduser()
    if requested_output_root.is_absolute():
        output_root = requested_output_root
    else:
        cwd_text = os.environ.get("PWD", "").strip()
        cwd = Path(cwd_text) if cwd_text else Path.cwd()
        output_root = cwd / requested_output_root

    smoke = bool(args.smoke)
    materialized = materialize_phase1_training_samples(
        output_root,
        Phase1MaterializationConfig(
            dataset_name=str(args.dataset_name),
            trajectory_source_branch=str(args.trajectory_source_branch),
            smoke=smoke,
            smoke_max_trajectories=int(args.smoke_max_trajectories),
        ),
    )
    audit = AttributionLedgerBuffer(
        output_root=output_root,
        dataset_name=str(args.dataset_name),
        trajectory_source_branch=str(args.trajectory_source_branch),
        topk=int(args.topk),
        gt_sidecar_dir=str(args.gt_sidecar_dir),
    )

    if smoke:
        prealign_epochs = int(args.prealign_epochs) if args.prealign_epochs is not None else 1
        base_epochs = int(args.base_epochs) if args.base_epochs is not None else 1
        aug_epochs = int(args.aug_epochs) if args.aug_epochs is not None else 1
        prealign_lr = float(args.prealign_learning_rate) if args.prealign_learning_rate is not None else 1e-4
        base_lr = float(args.base_learning_rate) if args.base_learning_rate is not None else 5e-5
        aug_lr = float(args.aug_learning_rate) if args.aug_learning_rate is not None else 5e-5
        em_subiterations = resolve_em_subiterations(smoke=True, explicit=args.em_subiterations)
    else:
        prealign_epochs = int(args.prealign_epochs) if args.prealign_epochs is not None else 5
        base_epochs = int(args.base_epochs) if args.base_epochs is not None else 5
        aug_epochs = int(args.aug_epochs) if args.aug_epochs is not None else 5
        prealign_lr = float(args.prealign_learning_rate) if args.prealign_learning_rate is not None else 1e-4
        base_lr = float(args.base_learning_rate) if args.base_learning_rate is not None else 1e-4
        aug_lr = float(args.aug_learning_rate) if args.aug_learning_rate is not None else 1e-4
        em_subiterations = resolve_em_subiterations(smoke=False, explicit=args.em_subiterations)

    prealign_result = train_prealign(
        output_root=output_root,
        materialized_samples=materialized["samples"],
        config=PrealignConfig(
            dataset_name=str(args.dataset_name),
            trajectory_source_branch=str(args.trajectory_source_branch),
            device=str(args.device),
            seed=int(args.seed),
            smoke=smoke,
            epochs=int(prealign_epochs),
            learning_rate=float(prealign_lr),
            weight_decay=float(args.weight_decay),
            temperature=float(args.temperature),
        ),
        audit_callback=audit,
    )

    softem_result = run_soft_em(
        output_root=output_root,
        materialized_samples=materialized["samples"],
        config=SoftEMConfig(
            dataset_name=str(args.dataset_name),
            trajectory_source_branch=str(args.trajectory_source_branch),
            mode=str(args.mode),
            device=str(args.device),
            seed=int(args.seed),
            smoke=smoke,
            temperature=float(args.temperature),
            weight_decay=float(args.weight_decay),
            em_subiterations=int(em_subiterations),
            base_epochs=int(base_epochs),
            aug_epochs=int(aug_epochs),
            base_learning_rate=float(base_lr),
            aug_learning_rate=float(aug_lr),
        ),
        audit_callback=audit,
    )

    summary = audit.finalize()
    payload = {
        "status": "PASS",
        "gate_id": "G7_training",
        "phase_scope": "audit_v1_attribution_trend",
        "exp_name": str(args.exp_name),
        "dataset_name": str(args.dataset_name),
        "trajectory_source_branch": str(args.trajectory_source_branch),
        "formal_training_ready": False,
        "audit_only": True,
        "training_semantics_changed": False,
        "materialization_stats": materialized["stats"],
        "prealign_result": {
            "record_count_input": prealign_result["record_count_input"],
            "record_count_trainable": prealign_result["record_count_trainable"],
            "record_count_output": prealign_result["record_count_output"],
            "coverage_ratio_trainable": prealign_result["coverage_ratio_trainable"],
            "skipped_reason_histogram": prealign_result["skipped_reason_histogram"],
            "loss_mean": prealign_result["loss_mean"],
            "loss_last": prealign_result["loss_last"],
        },
        "softem_result": {
            "record_count_input": softem_result["record_count_input"],
            "record_count_trainable": softem_result["record_count_trainable"],
            "record_count_output": softem_result["record_count_output"],
            "coverage_ratio_trainable": softem_result["coverage_ratio_trainable"],
            "skipped_reason_histogram": softem_result["skipped_reason_histogram"],
            "stage_reports": softem_result["stage_reports"],
            "selected_checkpoint_path": softem_result["selected_checkpoint_path"],
        },
        "ledger_summary": summary,
        "artifacts": {
            "prealign_ledger": "train/prealign/attribution_ledger.jsonl",
            "softem_base_ledger": "train/softem_base/attribution_ledger.jsonl",
            "softem_aug_ledger": "train/softem_aug/attribution_ledger.jsonl",
            "summary": "train/audit/attribution_summary.json",
            "hand_off_md": "codex/outputs/G7_training/g7_audit_v1_impl_latest.md",
            "hand_off_json": "codex/outputs/G7_training/g7_audit_v1_impl_latest.json",
        },
        "current_asset_mode_behavior": "bounded_provisional_audit_with_optional_gt_sidecar",
        "blocked_follow_up_items": [
            "G7 core repair remains a separate task",
            "extra recovery audit remains blocked until aug semantics are real",
        ],
    }
    _write_json(_summary_path(repo_root), payload)
    _write_md(
        _summary_md_path(repo_root),
        [
            "# G7 Audit-v1 Handoff",
            "",
            "- status: PASS",
            "- audit_only: true",
            "- training_semantics_changed: false",
            "- formal_training_ready: false",
            "- emitted_artifacts: train/prealign/attribution_ledger.jsonl, train/softem_base/attribution_ledger.jsonl, train/softem_aug/attribution_ledger.jsonl, train/audit/attribution_summary.json",
            "- blocked_follow_up_items:",
            "  - G7 core repair remains a separate task",
            "  - extra recovery audit remains blocked until aug semantics are real",
        ],
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
