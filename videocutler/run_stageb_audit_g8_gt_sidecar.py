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

from videocutler.ext_stageb_ovvis.audit.g8_gt_sidecar_generation import (
    G8GTSidecarGenerationConfig,
    run_g8_gt_sidecar_generation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G8 GT sidecar-only generation. Generates only trajectory→GT sidecar artifacts and no audit metrics.")
    parser.add_argument("--dataset_name", required=True, choices=("lvvis_train_base", "lvvis_val"))
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--gt_sidecar_dir", default="audit")
    parser.add_argument("--rewrite_existing", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = G8GTSidecarGenerationConfig(
        output_root=Path(args.output_root),
        dataset_name=str(args.dataset_name),
        gt_sidecar_dir=str(args.gt_sidecar_dir),
        rewrite_existing=bool(args.rewrite_existing),
    )
    summary = run_g8_gt_sidecar_generation(config)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
