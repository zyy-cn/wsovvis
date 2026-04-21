from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_root_for_direct_cli()

import torch

from videocutler.ext_stageb_ovvis.audit.g8_minimal_split_audit import (
    MinimalSplitAuditConfig,
    run_minimal_split_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal split audit for train/val semantic protocol.")
    parser.add_argument("--dataset_name", required=True, choices=("lvvis_train_base", "lvvis_val", "ytvis_2019_val"))
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--stage", required=True, choices=("prealign", "softem_base", "all"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trajectory_source_branch", default="mainline")
    parser.add_argument("--all_gt_generate_sidecars_if_missing", action="store_true")
    parser.add_argument("--heartbeat_every_rows", type=int, default=256)
    parser.add_argument("--batch_size_rows", type=int, default=128)
    parser.add_argument("--candidate_chunk_size", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = MinimalSplitAuditConfig(
        dataset_name=str(args.dataset_name),
        output_root=Path(args.output_root).expanduser().resolve(),
        stage=str(args.stage),
        device=torch.device(args.device),
        trajectory_source_branch=str(args.trajectory_source_branch),
        all_gt_generate_sidecars_if_missing=bool(args.all_gt_generate_sidecars_if_missing),
        heartbeat_every_rows=int(args.heartbeat_every_rows),
        batch_size_rows=int(args.batch_size_rows),
        candidate_chunk_size=int(args.candidate_chunk_size),
    )
    summary = run_minimal_split_audit(config)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
