#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wsovvis.track_feature_export import (  # noqa: E402
    ExportContractError,
    build_normalized_bridge_input_from_real_stageb_sidecar_to_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build normalized Stage-B bridge-input JSON from real Stage B prediction artifacts and "
            "feature-export sidecar v1 artifacts."
        )
    )
    parser.add_argument("--run-root", type=Path, required=False, help="Run root path (e.g., runs/wsovvis_seqformer/18)")
    parser.add_argument(
        "--sidecar-root",
        type=Path,
        required=False,
        help="Sidecar root path (e.g., runs/.../d2/inference/feature_export_v1)",
    )
    parser.add_argument(
        "--inference-root",
        type=Path,
        required=False,
        help="Inference root path (defaults from run-root or sidecar-root)",
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        required=False,
        help="Path to run config.json used to resolve split-domain val_json",
    )
    parser.add_argument(
        "--sample-video-limit",
        type=int,
        required=False,
        help="Optional deterministic limit over sorted split-domain video IDs",
    )
    parser.add_argument("--output-json", type=Path, required=True, help="Output normalized bridge-input JSON")
    parser.add_argument(
        "--join-summary-json",
        type=Path,
        dest="summary_json",
        required=False,
        help="Optional output path for QA/join summary counters JSON",
    )
    parser.add_argument(
        "--qa-summary-json",
        type=Path,
        dest="summary_json",
        required=False,
        help="Alias of --join-summary-json. Optional output path for QA summary JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = build_normalized_bridge_input_from_real_stageb_sidecar_to_json(
            output_json_path=args.output_json,
            run_root=args.run_root,
            sidecar_root=args.sidecar_root,
            inference_root=args.inference_root,
            config_json_path=args.config_json,
            sample_video_limit=args.sample_video_limit,
            summary_json_path=args.summary_json,
        )
    except ExportContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote normalized bridge-input JSON: {args.output_json}")
    if args.summary_json is not None:
        print(f"Wrote QA summary JSON: {args.summary_json}")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
