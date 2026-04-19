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

from videocutler.ext_stageb_ovvis.audit.gt_attribution_rank_audit import (
    GTAttributionRankAuditConfig,
    run_gt_attribution_rank_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-wise matched GT attribution rank audit over full dataset class space.")
    parser.add_argument("--dataset_name", required=True, choices=("lvvis_val", "ytvis_2019_val"))
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--stage", required=True, choices=("prealign", "softem_base", "softem_aug", "all"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--logit_chunk_size", type=int, default=256)
    parser.add_argument("--trajectory_source_branch", default="mainline")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = GTAttributionRankAuditConfig(
        dataset_name=args.dataset_name,
        output_root=Path(args.output_root).expanduser().resolve(),
        stage=args.stage,
        device=torch.device(args.device),
        logit_chunk_size=int(args.logit_chunk_size),
        trajectory_source_branch=str(args.trajectory_source_branch),
    )
    summary = run_gt_attribution_rank_audit(config)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
