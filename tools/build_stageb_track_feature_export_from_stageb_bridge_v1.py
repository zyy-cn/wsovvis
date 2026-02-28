#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wsovvis.track_feature_export import (
    ExportContractError,
    build_track_feature_export_v1_from_stageb_bridge_input_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Stage B track feature export v1 artifacts from normalized bridge-input JSON. "
            "Bridge input requires: split, producer, split_domain_video_ids, stageb_video_results, "
            "embedding_pooling='track_pooled'."
        )
    )
    parser.add_argument("--input-json", type=Path, required=True, help="Normalized bridge-input JSON path")
    parser.add_argument(
        "--output-split-root",
        type=Path,
        required=True,
        help="Output split root directory (manifest.v1.json and videos/* payloads)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it exists")
    parser.add_argument(
        "--bridge-summary-json",
        type=Path,
        required=False,
        help="Optional output path for machine-readable bridge summary counters JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output_path, summary = build_track_feature_export_v1_from_stageb_bridge_input_json(
            input_json_path=args.input_json,
            output_split_root=args.output_split_root,
            overwrite=args.overwrite,
        )
    except ExportContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.bridge_summary_json is not None:
        args.bridge_summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.bridge_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote Stage B track feature export v1 split artifact: {output_path}")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
