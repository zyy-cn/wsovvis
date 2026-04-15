from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from videocutler.ext_stageb_ovvis.audit.extra_recovery_audit import run_extra_recovery_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G7 extra recovery audit: audit-only analysis of whether softem_aug extra classes correspond to real missing classes."
    )
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset_name", default="lvvis_train_base", choices=("lvvis_train_base",))
    parser.add_argument("--trajectory_source_branch", default="mainline", choices=("mainline",))
    parser.add_argument("--smoke_max_trajectories", type=int, default=128)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--gt_sidecar_dir", default="audit")
    parser.add_argument(
        "--generate_val_sidecars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate val-sidecar audit artifacts alongside the requested train-sidecar audit artifacts.",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _summary_path(repo_root: Path) -> Path:
    return repo_root / "codex" / "outputs" / "G7_training" / "g7_extra_recovery_audit_latest.json"


def _summary_md_path(repo_root: Path) -> Path:
    return repo_root / "codex" / "outputs" / "G7_training" / "g7_extra_recovery_audit_latest.md"


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

    payload = run_extra_recovery_audit(
        output_root=output_root,
        dataset_name=str(args.dataset_name),
        trajectory_source_branch=str(args.trajectory_source_branch),
        smoke=bool(args.smoke),
        smoke_max_trajectories=int(args.smoke_max_trajectories),
        topk=int(args.topk),
        generate_val_sidecars=bool(args.generate_val_sidecars),
        gt_sidecar_dir=str(args.gt_sidecar_dir),
    )
    payload.update(
        {
            "status": payload.get("status", "PASS"),
            "gate_id": "G7_training",
            "phase_scope": "extra_recovery_audit",
            "exp_name": str(args.exp_name),
            "device": str(args.device),
            "seed": int(args.seed),
            "formal_training_ready": False,
            "audit_only": True,
            "training_semantics_changed": False,
            "generate_val_sidecars": bool(args.generate_val_sidecars),
            "artifacts": {
                "sidecars": [
                    "audit/trajectory_gt_match_train_mainline.jsonl",
                    "audit/trajectory_gt_identity_train_gt.jsonl",
                    "audit/trajectory_gt_match_val_mainline.jsonl",
                    "audit/trajectory_gt_identity_val_gt.jsonl",
                ],
                "ledger": "train/softem_aug/extra_recovery_ledger.jsonl",
                "summary": "train/audit/extra_recovery_summary.json",
                "hand_off_md": "codex/outputs/G7_training/g7_extra_recovery_audit_latest.md",
                "hand_off_json": "codex/outputs/G7_training/g7_extra_recovery_audit_latest.json",
            },
            "current_asset_mode_behavior": "bounded_provisional_extra_recovery_audit_with_offline_gt_sidecar",
            "blocked_follow_up_items": [
                "G7 formal closure remains a separate task",
                "extra recovery audit is diagnostic-only evidence",
            ],
        }
    )
    _write_json(_summary_path(repo_root), payload)
    _write_md(
        _summary_md_path(repo_root),
        [
            "# G7 Extra Recovery Audit",
            "",
            "- status: PASS" if payload.get("status") == "PASS" else f"- status: {payload.get('status', 'EMPTY')}",
            "- audit_only: true",
            "- training_semantics_changed: false",
            "- formal_training_ready: false",
            "- emitted_artifacts: audit/trajectory_gt_match_train_mainline.jsonl, audit/trajectory_gt_identity_train_gt.jsonl, audit/trajectory_gt_match_val_mainline.jsonl, audit/trajectory_gt_identity_val_gt.jsonl, train/softem_aug/extra_recovery_ledger.jsonl, train/audit/extra_recovery_summary.json",
            "- blocked_follow_up_items:",
            "  - G7 formal closure remains a separate task",
            "  - extra recovery audit is diagnostic-only evidence",
        ],
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
