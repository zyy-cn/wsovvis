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

from videocutler.ext_stageb_ovvis.audit.g8_softem_aug_extra_audit import (
    SoftemAugExtraAuditConfig,
    run_softem_aug_extra_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional softem_aug extra correctness audit.")
    parser.add_argument("--dataset_name", required=True, choices=("lvvis_train_base",))
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--trajectory_source_branch", default="mainline")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--generate_sidecars_if_missing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SoftemAugExtraAuditConfig(
        dataset_name=str(args.dataset_name),
        output_root=Path(args.output_root).expanduser().resolve(),
        trajectory_source_branch=str(args.trajectory_source_branch),
        topk=int(args.topk),
        generate_sidecars_if_missing=bool(args.generate_sidecars_if_missing),
    )
    summary = run_softem_aug_extra_audit(config)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
