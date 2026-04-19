from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from videocutler.ext_stageb_ovvis.audit.dropped_gt_attribution_audit import run_dropped_gt_attribution_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G7 dropped-GT attribution audit across prealign / softem_base / softem_aug stages."
    )
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset_name", default="lvvis_train_base", choices=("lvvis_train_base",))
    parser.add_argument("--trajectory_source_branch", default="mainline", choices=("mainline",))
    parser.add_argument("--smoke_max_trajectories", type=int, default=128)
    parser.add_argument("--stage", default="all", choices=("prealign", "softem_base", "softem_aug", "all"))
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _summary_path(repo_root: Path) -> Path:
    return repo_root / "codex" / "outputs" / "G7_training" / "g7_dropped_gt_attribution_latest.json"


def _summary_md_path(repo_root: Path) -> Path:
    return repo_root / "codex" / "outputs" / "G7_training" / "g7_dropped_gt_attribution_latest.md"


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

    payload = run_dropped_gt_attribution_audit(
        output_root=output_root,
        dataset_name=str(args.dataset_name),
        trajectory_source_branch=str(args.trajectory_source_branch),
        smoke=bool(args.smoke),
        smoke_max_trajectories=int(args.smoke_max_trajectories),
        stage=str(args.stage),
    )
    payload.update(
        {
            "gate_id": "G7_training",
            "phase_scope": "dropped_gt_attribution_audit",
            "exp_name": str(args.exp_name),
            "device": str(args.device),
            "seed": int(args.seed),
            "formal_training_ready": False,
            "audit_only": True,
            "training_semantics_changed": False,
            "artifacts": {
                "summary": "train/audit/dropped_gt_attribution_summary.json",
                "ledgers": [
                    "train/prealign/dropped_gt_attribution_ledger.jsonl",
                    "train/softem_base/dropped_gt_attribution_ledger.jsonl",
                    "train/softem_aug/dropped_gt_attribution_ledger.jsonl",
                ],
                "hand_off_md": "codex/outputs/G7_training/g7_dropped_gt_attribution_latest.md",
                "hand_off_json": "codex/outputs/G7_training/g7_dropped_gt_attribution_latest.json",
            },
        }
    )
    _write_json(_summary_path(repo_root), payload)
    lines = [
        "# G7 Dropped GT Attribution Audit",
        "",
        f"- status: {payload.get('status', 'EMPTY')}",
        f"- requested_stage: {payload.get('requested_stage', 'all')}",
        "- audit_only: true",
        "- training_semantics_changed: false",
        "",
        "## Stage Summaries",
    ]
    stage_summaries = dict(payload.get("stage_summaries", {}))
    for stage_id in ("prealign", "softem_base", "softem_aug"):
        summary = stage_summaries.get(stage_id)
        if not summary:
            continue
        lines.extend(
            [
                f"### {stage_id}",
                f"- dropped_gt_count: {summary.get('dropped_gt_count')}",
                f"- dropped_gt_mean_rank: {summary.get('dropped_gt_mean_rank')}",
                f"- dropped_gt_top1_hit_rate: {summary.get('dropped_gt_top1_hit_rate')}",
                f"- dropped_gt_top5_hit_rate: {summary.get('dropped_gt_top5_hit_rate')}",
                f"- dropped_gt_top10_hit_rate: {summary.get('dropped_gt_top10_hit_rate')}",
                f"- dropped_gt_mrr: {summary.get('dropped_gt_mrr')}",
                f"- wrong_top1_is_base_rate: {summary.get('wrong_top1_is_base_rate')}",
                f"- dropped_gt_in_stage_domain_rate: {summary.get('dropped_gt_in_stage_domain_rate')}",
            ]
        )
    _write_md(_summary_md_path(repo_root), lines)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
