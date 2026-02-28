#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wsovvis.track_feature_export import (
    ExportContractError,
    build_track_feature_export_v1,
    load_task_local_input,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Stage B track feature export v1 artifacts from task-local synthetic JSON input. "
            "Input requires top-level fields: split, embedding_dim, embedding_dtype='float32', "
            "embedding_pooling='track_pooled', embedding_normalization, producer, videos. "
            "Processed tracks must include inline embedding vectors."
        )
    )
    parser.add_argument("--input-json", type=Path, required=True, help="Task-local producer input JSON path")
    parser.add_argument(
        "--output-split-root",
        type=Path,
        required=True,
        help="Output split root directory (contains manifest.v1.json and videos/* payloads)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it exists")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = load_task_local_input(args.input_json)
        output_path = build_track_feature_export_v1(
            input_payload=payload,
            output_split_root=args.output_split_root,
            overwrite=args.overwrite,
        )
    except ExportContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote Stage B track feature export v1 split artifact: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
